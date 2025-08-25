# src/train.py
import argparse
import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import joblib

from .mlflow_setup import setup_mlflow
from .config import RAW_TRAIN, PROC_PATH, DATE_COL, ARTIFACTS_DIR, TARGET, MODELS_DIR
from .data import save_processed, split_xy, load_processed
from .clean import coerce_and_impute, winsorize, parse_dates_and_features
from .features import fill_categoricals, ordinal_encode, encode_nominals_and_binaries
from .models import build_model, rmsle
from .splits import stratified_regression_folds


def preprocess(raw_path=RAW_TRAIN):
    df = pd.read_csv(raw_path)
    df = coerce_and_impute(df)
    df = winsorize(df)
    df = parse_dates_and_features(df)
    df = fill_categoricals(df)
    df = ordinal_encode(df)
    df = encode_nominals_and_binaries(df)

    # timestamp para FEAST

    df["event_timestamp"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def _load_best_params_map(models, best_params_dir):
    """Lee los archivos JSON de Optuna y devuelve un mapa de parámetros."""
    best = {}
    if not best_params_dir:
        return {m: {} for m in models} # Devuelve dicts vacíos si no hay dir
        
    for m in models:
        p = Path(best_params_dir) / f"{m}_best_params.json"
        if p.exists():
            best[m] = json.loads(p.read_text())["params"]
        else:
            print(f"[WARN] No se encontró el archivo de parámetros para {m}. Usando defaults.")
            best[m] = {}
    return best

class StackingEnsemble:
    """Clase para empaquetar los modelos base y el meta-modelo para inferencia."""
    def __init__(self, bases, meta):
        self.bases = bases
        self.meta = meta
        
    def predict(self, X_new: pd.DataFrame):
        # Asegurarse de que X_new solo contenga columnas numéricas/booleanas
        Xn = X_new.select_dtypes(include=["number","bool"]).copy()
        # Generar predicciones de los modelos base (meta-features)
        Znew = np.column_stack([b.predict(Xn) for b in self.bases])
        # Predecir con el meta-modelo
        return self.meta.predict(Znew)

def train_and_evaluate_stacking(
    df: pd.DataFrame,
    base_models=("hgb", "xgb", "lgbm"),
    n_splits=5,
    best_params_dir=None
):
    """
    Realiza el entrenamiento y evaluación del ensamblaje de stacking.
    
    1. Genera predicciones Out-of-Fold (OOF) para cada modelo base.
    2. Entrena un meta-modelo (RidgeCV) sobre estas predicciones OOF.
    3. Evalúa el rendimiento del ensamblaje completo usando las predicciones OOF.
    4. Re-entrena los modelos base en todos los datos.
    5. Re-entrena el meta-modelo en todas las predicciones OOF.
    6. Devuelve las métricas de CV y el ensamblaje final entrenado.
    """
    X, y = split_xy(df)
    folds = stratified_regression_folds(y, n_splits=n_splits, seed=42)
    best_map = _load_best_params_map(base_models, best_params_dir)

    oof_preds_df = pd.DataFrame(index=X.index)
    cv_metrics = {}

    print("--- Generando predicciones Out-of-Fold para modelos base ---")
    for name in base_models:
        oof_col_name = f"{name}_oof"
        oof_preds_df[oof_col_name] = np.nan
        with mlflow.start_run(run_name=f"{name}-oof-generation", nested=True):
            fold_scores = []
            for i, (train_idx, val_idx) in enumerate(folds, start=1):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model = build_model(name, best_map.get(name, {}))
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                oof_preds_df.loc[X.index[val_idx], oof_col_name] = preds
                
                score = rmsle(y_val, preds)
                fold_scores.append(score)
                mlflow.log_metric(f"rmsle_fold_{i}", score)
                
            mean_score = float(np.mean(fold_scores))
            mlflow.log_metric("rmsle_cv_mean", mean_score)
            cv_metrics[name] = mean_score
    
    print("--- Entrenando y evaluando el meta-modelo de Stacking ---")
    # Meta-dataset: predicciones OOF de los modelos base
    X_meta = oof_preds_df.values

    # OOF del meta (evitar leakage)
    meta_model_eval = RidgeCV(alphas=np.logspace(-3, 2, 20))
    stack_oof = np.full(len(y), np.nan, dtype=float)

    for train_idx, val_idx in folds:
        meta_model_eval.fit(X_meta[train_idx], y.iloc[train_idx])
        stack_oof[val_idx] = meta_model_eval.predict(X_meta[val_idx])

    # Métrica CV del stacking usando OOF completo
    stacking_cv_score = float(rmsle(y, stack_oof))
    cv_metrics["stacking"] = stacking_cv_score
    print(f"--- Score de Stacking (RMSLE CV): {stacking_cv_score:.5f} ---")

    # Guardar OOF del stack para análisis
    oof_preds_df["stacking_oof"] = stack_oof


    # --- Re-entrenamiento final para producción ---
    print("--- Re-entrenando modelos finales en todos los datos ---")
    # 1. Re-entrenar modelos base en todos los datos (X, y)
    fitted_bases = []
    for name in base_models:
        model_full = build_model(name, best_map.get(name, {}))
        model_full.fit(X, y)
        fitted_bases.append(model_full)

    # 2. Re-entrenar meta-modelo en todas las predicciones OOF (X_meta, y)
    meta_model_full = RidgeCV(alphas=np.logspace(-3, 2, 20))
    meta_model_full.fit(X_meta, y)

    # 3. Empaquetar el ensamblaje final
    final_ensemble = StackingEnsemble(bases=fitted_bases, meta=meta_model_full)

    return cv_metrics, final_ensemble, oof_preds_df


if __name__ == "__main__":
    # --- Añadimos el parser de argumentos ---
    parser = argparse.ArgumentParser(description="Pipeline de ML con Stacking")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--only-preprocess",
        action="store_true",
        help="Ejecuta solo el paso de preprocesamiento."
    )
    group.add_argument(
        "--from-processed",
        type=str,
        help="Ruta al parquet para iniciar el entrenamiento."
    )
    #parser.add_argument("--only-preprocess", action="store_true", help="Ejecuta solo el paso de preprocesamiento.")
    #parser.add_argument("--from-processed", type=str, default=None, help="Ruta al parquet procesado para iniciar el entrenamiento desde allí.")
    parser.add_argument("--use-best-params", type=str, default=None, help="Ruta al directorio con los JSON de Optuna para entrenar con los mejores parámetros.")
    
    args = parser.parse_args()

    # --- Lógica de control basada en los argumentos ---
    setup_mlflow("ml-pipeline")

    if args.only_preprocess:
        with mlflow.start_run(run_name="preprocess"):
            print("--- Ejecutando solo el preprocesamiento ---")
            dfp = preprocess()
            save_processed(dfp, PROC_PATH)
            mlflow.log_artifact(PROC_PATH)
            print(f"--- Datos procesados y guardados en {PROC_PATH} ---")

    elif args.from_processed:
        with mlflow.start_run(run_name="final-training-and-stacking"):
            print(f"--- Iniciando entrenamiento y stacking desde: {args.from_processed} ---")
            dfp = load_processed(args.from_processed)
            metrics, ensemble_model, oof_predictions = train_and_evaluate_stacking(
                dfp, best_params_dir=args.use_best_params
            )

            # Guardando OOF
            print("--- Guardando predicciones Out-of-Fold para análisis ---")
            oof_predictions[TARGET] = dfp[TARGET] # Añadir el target real
            oof_path = ARTIFACTS_DIR / "oof_predictions.csv"
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            oof_predictions.to_csv(oof_path, index=False)

            # 3. Guardar el artefacto del MODELO para inferencia
            print("--- Guardando modelo de Stacking final ---")
            model_path = MODELS_DIR / "stacking_ensemble.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(ensemble_model, model_path)
            
            # 4. Registrar todo en MLflow
            print("--- Registrando artefactos y métricas en MLflow ---")
            mlflow.log_dict(metrics, "final_cv_metrics_with_stacking.json")
            mlflow.log_metric("rmsle_cv_stacking", metrics.get('stacking', -1))
            mlflow.log_artifact(str(oof_path))
            mlflow.log_artifact(str(model_path))
            
            print("--- Proceso completado ---")