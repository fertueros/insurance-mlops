# src/tune.py
import mlflow
import optuna
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from .config import PROC_PATH, ARTIFACTS_DIR
from .data import split_xy
from .models import build_model # Tu build_model sigue siendo útil
from .mlflow_setup import setup_mlflow


def rmsle(y_true, y_pred):
    safe_y_pred = np.maximum(0, y_pred)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(safe_y_pred))))

def space(trial, name):
    if name=="hgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    max_iter=trial.suggest_int("max_iter", 200, 2000),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    max_bins=trial.suggest_int("max_bins", 64, 255),
                    l2_regularization=trial.suggest_float("l2_regularization", 0.0, 1.0))
    if name=="xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 400, 1500),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    max_depth=trial.suggest_int("max_depth", 4, 10),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    tree_method="gpu_hist")
    if name=="lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 400, 1500),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 31, 255),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0))
    raise ValueError("Modelo no soportado")

def objective_factory(name: str, X: pd.DataFrame, y: pd.Series, folds):
    """Crea la función 'objective' para un modelo específico."""
    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(nested=True):
            # Tu función space() se puede integrar aquí o llamarla
            params = space(trial, name) # Asumiendo que 'space' existe como antes
            mlflow.log_params(params)

            fold_scores = []
            for tr_idx, val_idx in folds:
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                model = build_model(name, params)
                model.fit(X_tr, np.log1p(y_tr)) # Entrenar en log
                
                preds_log = model.predict(X_val)
                preds_original = np.expm1(preds_log) # Revertir predicción

                score = rmsle(y_val, preds_original)
                fold_scores.append(score)

            avg_score = float(np.mean(fold_scores))
            mlflow.log_metric("rmsle_cv", avg_score)
        
        return avg_score
    return objective

def run_tuning(model: str, data_path: str, trials: int):
    """Orquesta la sesión de tuning completa para un modelo."""
    df = pd.read_parquet(data_path)
    X, y = split_xy(df)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, pd.qcut(np.log1p(y), 10, labels=False, duplicates="drop"))

    study_name = f"{model}-tuning-study"
    storage_path = f"sqlite:///artifacts/optuna/study.db"
    
    with mlflow.start_run(run_name=f"Optuna_Study_{model}"):
        study = optuna.create_study(
            direction="minimize", study_name=study_name,
            storage=storage_path, load_if_exists=True
            )
        objective_func = objective_factory(model, X, y, list(folds))
        study.optimize(objective_func, n_trials=trials)

        # Guardar resultados
        best_results = {"model": model, "value": study.best_value, "params": study.best_params}
        
        # Loguear en el run padre de MLflow
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmsle_cv", study.best_value)
        
        # Guardar artefacto JSON
        out_path = ARTIFACTS_DIR / "optuna" / f"{model}_best_params.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(best_results, f, indent=2)
        
        mlflow.log_artifact(str(out_path))

        print(f"Búsqueda para {model} completada. Mejor score: {study.best_value}")
        return best_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optimización de Hiperparámetros con Optuna")
    parser.add_argument("--model", type=str, required=True, choices=["hgb", "xgb", "lgbm"], help="Nombre del modelo a optimizar.")
    parser.add_argument("--data", type=str, default=str(PROC_PATH), help="Ruta al dataset de entrenamiento (parquet).") # Convertir a str
    parser.add_argument("--trials", type=int, default=50, help="Número de trials de Optuna a ejecutar.")
    
    args = parser.parse_args()

    # Configura el experimento de MLflow
    setup_mlflow("hyperparameter-tuning")
    
    # Llama a la función principal que orquesta todo
    run_tuning(model=args.model, data_path=args.data, trials=args.trials)