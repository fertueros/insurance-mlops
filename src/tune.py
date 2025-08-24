# src/tune.py
import argparse
import json, os, optuna, numpy as np, pandas as pd, mlflow
from .mlflow_setup import setup_mlflow
from optuna.pruners import MedianPruner
from optuna.integration.mlflow import MLflowCallback
from .config import PROC_PATH, TARGET
from .splits import stratified_regression_folds
from .models import build_model, rmsle

def load_df(path=PROC_PATH): 
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

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
                    tree_method="hist")
    if name=="lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 400, 1500),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 31, 255),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0))
    raise ValueError("Modelo no soportado")

def objective_factory(name, X, y, folds):
    def obj(trial):
        params = space(trial, name)
        fold_scores = []
        for tr,va in folds:
            m = build_model(name, params)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[va])
            fold_scores.append(rmsle(y.iloc[va], pred))
        return float(np.mean(fold_scores))
    return obj

def run(model="hgb", data_path=PROC_PATH, trials=50):
    df = load_df(data_path); X = df.drop(columns=[TARGET]); y = df[TARGET]
    folds = stratified_regression_folds(y, n_splits=5, seed=42)
    os.makedirs("artifacts/optuna", exist_ok=True)
    mlcb = MLflowCallback(metric_name="rmsle_cv")
    study = optuna.create_study(direction="minimize", study_name=f"{model}-rmsle",
                                storage="sqlite:///artifacts/optuna/study.db", load_if_exists=True,
                                pruner=MedianPruner(n_startup_trials=5))
    study.optimize(objective_factory(model, X, y, folds), n_trials=trials, callbacks=[mlcb])
    best = {"model": model, "value": study.best_value, "params": study.best_params}
    with open(f"artifacts/optuna/{model}_best_params.json","w") as f: json.dump(best, f, indent=2)
    mlflow.log_dict(best, f"optuna_{model}_best.json")
    return best

if __name__ == "__main__":
    # --- Añadimos el parser de argumentos ---
    parser = argparse.ArgumentParser(description="Optimización de Hiperparámetros con Optuna")
    parser.add_argument("--model", type=str, required=True, choices=["hgb", "xgb", "lgbm"], help="Nombre del modelo a optimizar.")
    parser.add_argument("--data", type=str, default=PROC_PATH, help="Ruta al dataset de entrenamiento (parquet).")
    parser.add_argument("--trials", type=int, default=50, help="Número de trials de Optuna a ejecutar.")
    
    args = parser.parse_args()

    # --- Lógica de ejecución ---
    setup_mlflow("hyperparameter-tuning")
    with mlflow.start_run(run_name=f"optuna-tune-{args.model}"):
        mlflow.log_params({"model_to_tune": args.model, "num_trials": args.trials, "data_path": args.data})
        print(f"--- Iniciando búsqueda de Optuna para el modelo: {args.model} ---")
        best_results = run(model=args.model, data_path=args.data, trials=args.trials)
        print("--- Búsqueda completada ---")
        print(f"Mejor score: {best_results['value']}")
        print(f"Mejores parámetros: {best_results['params']}")