import numpy as np, pandas as pd
from .config import NUM_COLS, DATE_COL

# limpiando numericas y features a partir de fecha
def coerce_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Coacciona columnas numéricas y realiza imputación de nulos con la mediana.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame con las columnas numéricas limpias e imputadas.
    """
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df

def winsorize(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica winsorización para acotar valores extremos (outliers).

    Args:
        df (pd.DataFrame): El DataFrame con columnas numéricas.

    Returns:
        pd.DataFrame: El DataFrame con los outliers de columnas específicas acotados.
    """
    for c in ["Annual Income","Previous Claims"]:
        if c in df.columns:
            lo, hi = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(lo, hi)
    return df

def parse_dates_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna de fecha y crea features de tiempo.

    Esta función realiza tres tareas principales:
    1.  Convierte la columna de fecha a formato datetime de pandas.
    2.  Imputa fechas faltantes con una fecha por defecto (1ro del mes).
    3.  Crea nuevas características: año, mes, día de la semana, y
        características cíclicas (seno/coseno) para el mes.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame con la columna de fecha parseada y 
        nuevas features de tiempo.
    """
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        # Imputa fechas faltantes a 1ro del mes más común.
        cm = int(df[DATE_COL].dt.month.mode(dropna=True).iloc[0]) if df[DATE_COL].notna().any() else 1
        df[DATE_COL] = df[DATE_COL].fillna(pd.Timestamp(year=2022, month=cm, day=1))
        df["psd_year"]  = df[DATE_COL].dt.year.astype("float64")
        df["psd_month"] = df[DATE_COL].dt.month.astype("float64")
        df["psd_dow"]   = df[DATE_COL].dt.dayofweek.astype("float64")
        df["psd_month_sin"] = np.sin(2*np.pi*df["psd_month"]/12.0)
        df["psd_month_cos"] = np.cos(2*np.pi*df["psd_month"]/12.0)
    return df
