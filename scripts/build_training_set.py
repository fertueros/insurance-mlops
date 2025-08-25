"""
Construye el training set histórico con Feast (point-in-time correct).

- Lee el parquet procesado (de tu preprocess) que contiene:
  * id                (join key)
  * event_timestamp   (timestamp para el join temporal)
  * Premium Amount    (target)

- Recupera TODAS las features declaradas en el FeatureView `premium_features`
  desde el registry de Feast, y ejecuta el point-in-time join contra el entity_df.

- Guarda el resultado en data/feast/training_set.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore


def _infer_feature_refs(store: FeatureStore, fv_name: str) -> list[str]:
    """
    Obtiene la lista de referencias 'feature_view:feature' a partir del FeatureView registrado.
    Soporta tanto atributo `features` como `schema` (según versión de Feast).
    """
    # Leer desde registry
    try:
        fv = store.get_feature_view(fv_name)
    except Exception as e:
        raise RuntimeError(
            f"No se encontró el FeatureView '{fv_name}' en el registry. "
            f"¿Ejecutaste `feast apply` en feature_repo/?"
        ) from e

    # Algunas versiones exponen .features, otras .schema con Field(name=...)
    fields = getattr(fv, "features", None)
    if not fields:
        fields = getattr(fv, "schema", None)
    if not fields:
        raise RuntimeError(
            f"El FeatureView '{fv_name}' no expone `.features` ni `.schema`. "
            "Revisa tu versión de Feast o tu definición de FeatureView."
        )

    feature_names = [getattr(f, "name", None) for f in fields]
    feature_names = [n for n in feature_names if n]  # limpia Nones
    if not feature_names:
        raise RuntimeError(f"'{fv_name}' no contiene features declaradas.")

    # Construye refs con el prefijo del FeatureView
    return [f"{fv_name}:{n}" for n in feature_names]


def build_training_set(
    repo_path: str = "feature_repo",
    processed_path: str = "data/processed/train_proc.parquet",
    out_path: str = "data/feast/training_set.parquet",
    fv_name: str = "premium_features",
    id_col: str = "id",
    ts_col: str = "event_timestamp",
    target_col: str = "Premium Amount",
    full_feature_names: bool = False,
) -> Path:
    """
    Genera el training set consistente temporalmente usando Feast.

    Parámetros
    ----------
    repo_path : str
        Ruta al feature repo (directorio que contiene feature_store.yaml).
    processed_path : str
        Parquet procesado con columnas [id, event_timestamp, target] y todas las features calculadas.
    out_path : str
        Parquet de salida con features + target.
    fv_name : str
        Nombre del FeatureView a consultar (por defecto 'premium_features').
    id_col : str
        Nombre de la join key (tu columna 'id').
    ts_col : str
        Nombre de la columna timestamp (para point-in-time join).
    target_col : str
        Nombre del target en el processed (lo renombraremos temporalmente a 'label' para el join).
    full_feature_names : bool
        Si True, Feast devolverá columnas prefijadas con el nombre del FeatureView.
    """
    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"No existe el parquet procesado: {processed_path}")

    dfp = pd.read_parquet(processed_path)

    # Validaciones mínimas
    for col in (id_col, ts_col, target_col):
        if col not in dfp.columns:
            raise ValueError(
                f"Falta la columna obligatoria '{col}' en {processed_path}. "
                "Asegúrate de que tu preprocess la genere."
            )

    # Entity DF: id + timestamp + label (el target se pasa como 'label' para conservarlo)
    entity_df = dfp[[id_col, ts_col, target_col]].rename(columns={target_col: "label"})

    # Conectar al FeatureStore del repo_path
    store = FeatureStore(repo_path=repo_path)

    # Descubrir automáticamente todas las features de tu FeatureView
    feature_refs = _infer_feature_refs(store, fv_name)

    # Recuperar features históricas (point-in-time correct) y convertir a pandas
    retrieval_job = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
        full_feature_names=full_feature_names,  # útil para evitar colisiones si unes varios FVs
    )
    training_df = retrieval_job.to_df()

    # Volver a llamar al target como en config
    training_df = training_df.rename(columns={"label": target_col})

    # Persistir
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(out_path, index=False)

    # Log de resumen simple
    print(
        f"[OK] Training set guardado en: {out_path}\n"
        f"Filas: {len(training_df):,} | Columnas: {len(training_df.columns):,}\n"
        f"Features recuperadas ({len(feature_refs)}): {', '.join([r.split(':',1)[1] for r in feature_refs])}"
    )

    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Construir training set histórico con Feast")
    p.add_argument("--repo-path", default="feature_repo")
    p.add_argument("--processed-path", default="data/processed/train_proc.parquet")
    p.add_argument("--out-path", default="data/feast/training_set.parquet")
    p.add_argument("--fv-name", default="premium_features")
    p.add_argument("--id-col", default="id")
    p.add_argument("--ts-col", default="event_timestamp")
    p.add_argument("--target-col", default="Premium Amount")
    p.add_argument("--full-feature-names", action="store_true",
                   help="Prefijar nombres con el FeatureView (evita colisiones)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        build_training_set(
            repo_path=args.repo_path,
            processed_path=args.processed_path,
            out_path=args.out_path,
            fv_name=args.fv_name,
            id_col=args.id_col,
            ts_col=args.ts_col,
            target_col=args.target_col,
            full_feature_names=args.full_feature_names,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
