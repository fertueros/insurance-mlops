# src/train.py
import argparse
import mlflow, numpy as np, pandas as pd
from .mlflow_setup import setup_mlflow
from .config import RAW_TRAIN, PROC_PATH, DATE_COL
from .data import save_processed, split_xy, load_processed
from .clean import coerce_and_impute, winsorize, parse_dates_and_features
from .features import fill_categoricals, ordinal_encode, encode_nominals_and_binaries
from .models import build_model, rmsle
from .splits import stratified_regression_folds
import json

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

def cv_train(df, models=("hgb","xgb","lgbm"), n_splits=5, best_params_dir=None):
    """Entrena y evalúa múltiples modelos usando validación cruzada estratificada.

    Para cada modelo especificado:
    1.  Inicia un 'run' anidado en MLflow para registrar los resultados.
    2.  Itera a través de los folds de validación cruzada.
    3.  Entrena un modelo en los datos de entrenamiento del fold.
    4.  Evalúa en los datos de validación del fold y registra la métrica (RMSLE).
    5.  Calcula y registra la métrica promedio de CV para el modelo.

    Args:
        df (pd.DataFrame): El DataFrame procesado.
        models (Tuple[str], optional): Nombres de los modelos a entrenar.
                                       Defaults to ("hgb", "xgb", "lgbm").
        n_splits (int, optional): Número de folds para la CV. Defaults to 5.

    Returns:
        Dict[str, float]: Un diccionario con el score de CV promedio para cada modelo.
    """
    X, y = split_xy(df)
    folds = stratified_regression_folds(y, n_splits=n_splits, seed=42)
    results = {}

    all_best_params = {}
    if best_params_dir:
        for m in models:
            try:
                with open(f'{best_params_dir}/{m}_best_params.json') as f:
                    all_best_params[m] = json.load(f)['params']
            except FileNotFoundError:
                all_best_params[m] = {}

    for name in models:
        oof = np.zeros(len(y))
        with mlflow.start_run(run_name=f"{name}-cv", nested=True):
            for i,(tr,va) in enumerate(folds, start=1):
                parametros = all_best_params.get(name, {})
                m = build_model(name, params=parametros)
                m.fit(X.iloc[tr], y.iloc[tr])
                pred = m.predict(X.iloc[va])
                score = rmsle(y.iloc[va], pred)
                mlflow.log_metric(f"rmsle_fold_{i}", score)
                oof[va] = pred
            mean_score = float(np.mean([mlflow.active_run().data.metrics[k] for k in mlflow.active_run().data.metrics if k.startswith("rmsle_fold_")]))
            mlflow.log_metric("rmsle_cv", mean_score)
            results[name] = mean_score
    return results

if __name__ == "__main__":
    # --- Añadimos el parser de argumentos ---
    parser = argparse.ArgumentParser(description="Pipeline de Preprocesamiento y Entrenamiento")
    parser.add_argument("--only-preprocess", action="store_true", help="Ejecuta solo el paso de preprocesamiento.")
    parser.add_argument("--from-processed", type=str, default=None, help="Ruta al parquet procesado para iniciar el entrenamiento desde allí.")
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
        with mlflow.start_run(run_name="train-cv"):
            print(f"--- Iniciando entrenamiento desde: {args.from_processed} ---")
            dfp = load_processed(args.from_processed)
            metrics = cv_train(dfp, n_splits=5, best_params_dir=args.use_best_params)
            mlflow.log_dict(metrics, "cv_metrics.json")
            print("--- Entrenamiento completado ---")
            
    else: # Comportamiento por defecto: ejecutar todo el pipeline
        with mlflow.start_run(run_name="full-pipeline-run"):
            print("--- Ejecutando pipeline completo: preprocess + train ---")
            dfp = preprocess()
            save_processed(dfp, PROC_PATH)
            mlflow.log_artifact(PROC_PATH)
            metrics = cv_train(dfp, n_splits=5)
            mlflow.log_dict(metrics, "cv_metrics.json")
            print("--- Pipeline completo ejecutado ---")