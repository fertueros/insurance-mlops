# src/mlflow_setup.py
from __future__ import annotations

import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(experiment_name: str) -> str:
    """
    Configura MLflow:
      - Usa el TRACKING_URI de env (o uno por defecto si no existe).
      - Si el experimento existe en estado 'deleted', lo restaura.
      - Si no existe, lo crea.
      - Deja el experimento activo para que mlflow.start_run() lo use.
    Devuelve: experiment_id (str).
    """
    # 1) Tracking URI
    uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/fertueros/insurance-mlops.mlflow")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()

    # 2) Localiza el experimento por nombre (incluye borrados según backend)
    exp = client.get_experiment_by_name(experiment_name)

    # 3) Si no lo encontró, podría estar en 'deleted' en algunos backends → búscalo entre borrados
    if exp is None:
        try:
            from mlflow.entities import ViewType
            deleted_exps = client.search_experiments(view_type=ViewType.DELETED_ONLY)
            for e in deleted_exps:
                if e.name == experiment_name:
                    client.restore_experiment(e.experiment_id)
                    exp = client.get_experiment(e.experiment_id)
                    break
        except Exception:
            # Si el backend/version no soporta ViewType, seguimos sin exp y crearemos uno nuevo
            pass

    # 4) Si existe pero está marcado como 'deleted', restáuralo
    if exp is not None and getattr(exp, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(exp.experiment_id)
        exp = client.get_experiment(exp.experiment_id)

    # 5) Créalo si sigue sin existir
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
        exp = client.get_experiment(exp_id)

    # 6) Deja el experimento activo por NOMBRE (ya restaurado/creado)
    mlflow.set_experiment(experiment_name)

    return exp.experiment_id
