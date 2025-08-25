import argparse
from pathlib import Path
import pandas as pd
import mlflow
import joblib

# Importaciones de tu proyecto
import sys
sys.path.append('.') # Añade la raíz del proyecto al path
from src.mlflow_setup import setup_mlflow
from src.config import RAW_TEST, FEAST_TRAINSET, MODELS_DIR, REPORTS_DIR, TARGET
from src.train import preprocess # Reutilizamos la función de preprocesamiento
from src.data import split_xy # Reutilizamos la función para obtener la estructura de features
from src.train import preprocess, StackingEnsemble

def align_columns(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    """Asegura que X_new tenga exactamente las mismas columnas que X_ref."""
    ref_cols = X_ref.columns
    X_new_aligned = X_new.reindex(columns=ref_cols, fill_value=0)
    return X_new_aligned[ref_cols]

def main(model_path: Path, train_ref_path: Path, test_data_path: Path, out_pred_path: Path):
    """Genera predicciones en datos nuevos (holdout) usando un modelo entrenado."""
    setup_mlflow("ml-pipeline")

    print("--- 1. Preprocesando el conjunto de test (sin target) ---")
    df_test_raw = pd.read_csv(test_data_path)
    df_test_processed = preprocess(test_data_path)

    print("--- 2. Alineando columnas con la estructura del conjunto de entrenamiento ---")
    df_train_ref = pd.read_parquet(train_ref_path)
    # Usamos split_xy para obtener la lista exacta de features que el modelo espera
    X_train_ref, _ = split_xy(df_train_ref) 
    
    X_test_aligned = align_columns(X_train_ref, df_test_processed)
    
    print(f"--- 3. Cargando el modelo desde: {model_path} ---")
    try:
        ensemble_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo del modelo. Asegúrate de haber ejecutado el stage 'train'.")
        sys.exit(1)
        
    print("--- 4. Realizando predicciones ---")
    predictions = ensemble_model.predict(X_test_aligned)

    print("--- 5. Guardando las predicciones ---")
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame({
        "id": df_test_raw["id"], # Usamos el 'id' del archivo original
        TARGET: predictions
    })
    submission_df.to_csv(out_pred_path, index=False)
    print(f"Predicciones guardadas en: {out_pred_path} ({len(predictions):,} filas)")

    # --- 6. Loguear artefacto en MLflow ---
    with mlflow.start_run(run_name="predict-holdout"):
        mlflow.log_artifact(str(out_pred_path))
        mlflow.log_param("model_used", str(model_path))
        print("--- Artefacto de predicción registrado en MLflow ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar predicciones en el conjunto de test.")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "stacking_ensemble.joblib")
    parser.add_argument("--train-ref-path", type=Path, default=FEAST_TRAINSET)
    parser.add_argument("--test-data-path", type=Path, default=RAW_TEST)
    parser.add_argument("--out-pred-path", type=Path, default=REPORTS_DIR / "predictions_test.csv")
    args = parser.parse_args()
    
    main(args.model_path, args.train_ref_path, args.test_data_path, args.out_pred_path)