# scripts/register_model.py
import argparse, tempfile, time
from pathlib import Path
import mlflow, mlflow.pyfunc, joblib
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

import sys
sys.path.append(".")
from src.train import StackingEnsemble
from src.mlflow_setup import setup_mlflow
from src.config import MODELS_DIR

class StackingPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        X = model_input.select_dtypes(include=["number","bool"]).copy()
        return self.model.predict(X)

def main(model_name: str, joblib_path: str, artifact_name: str = "stacking-model"):
    setup_mlflow("ml-pipeline")
    client = MlflowClient()
    client.set_registered_model_alias("insurance-premium-stacking", "champion", 1)  # v1

    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        run_id = run.info.run_id
        print(f"Iniciando registro '{model_name}' desde '{joblib_path}' (run_id={run_id})")

        # 1) Construir un MLflow model (PyFunc) LOCALMENTE
        tmpdir = tempfile.mkdtemp()
        model_dir = Path(tmpdir) / artifact_name

        mlflow.pyfunc.save_model(
            path=str(model_dir),
            python_model=StackingPyFunc(),
            artifacts={"model": str(joblib_path)},  # tu .joblib
            # opcional: fija dependencias si quieres entornos reproducibles
            # pip_requirements=["numpy","pandas","scikit-learn","lightgbm","xgboost","joblib","mlflow"],
        )

        # 2) Subir esa carpeta como artefactos del run (NO usamos log_model)
        mlflow.log_artifacts(str(model_dir), artifact_path=artifact_name)

        # 3) Registrar usando la URI del run (flujo soportado por DagsHub)
        model_uri = f"runs:/{run_id}/{artifact_name}"
        print(f"Registrando desde URI: {model_uri}")
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Modelo registrado: name={mv.name} version={mv.version}")

        # 4) Transicionar a Staging (clásico)
        time.sleep(3)
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Staging", archive_existing_versions=True
        )
        print(f"'{model_name}' v{mv.version} -> Staging OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="insurance-premium-stacking")
    ap.add_argument("--model-path", default=str(MODELS_DIR / "stacking_ensemble.joblib"))
    args = ap.parse_args()
    main(args.model_name, args.model_path)
