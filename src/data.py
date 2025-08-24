import pandas as pd
from .config import TARGET, PROC_PATH, DATE_COL

def save_processed(df: pd.DataFrame, path=PROC_PATH): df.to_parquet(path, index=False)
def load_processed(path=PROC_PATH): return pd.read_parquet(path)
def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en features (X) y target (y).

    Esta función elimina todas las columnas que no son características predictivas
    para el modelo. Esto incluye:
    - La variable objetivo (TARGET).
    - La columna de fecha original (DATE_COL).
    - Columnas de metadatos como IDs y timestamps de eventos.
    """
    # Lista de columnas que no son features
    metadata_cols = [TARGET, DATE_COL, "event_timestamp", "id"]
    
    # Nos aseguramos de solo intentar eliminar las columnas que realmente existen
    cols_to_drop = [col for col in metadata_cols if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    
    return X, y