import pandas as pd
from pandas.api.types import CategoricalDtype
from .config import ORDINAL_COLS, BINARY_COLS, NOMINAL_COLS

def fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores nulos en columnas categóricas con la etiqueta 'Unknown'.

    Aseguramos que las columnas sean de tipo 'category'en pandas.
    Luego, añade 'Unknown' a las categorías si no existe y rellena
    los valores faltantes (NaN) con esta etiqueta.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame con las columnas categóricas imputadas.
    """
    cat_cols = list(ORDINAL_COLS.keys()) + BINARY_COLS + NOMINAL_COLS
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
            if "Unknown" not in df[c].cat.categories:
                df[c] = df[c].cat.add_categories(["Unknown"])
            df[c] = df[c].fillna("Unknown")
    return df

def ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificación ordinal a las columnas definidas en ORDINAL_COLS.

    Convierte las categorías de texto de las columnas ordinales a códigos
    numéricos (enteros) basados en el orden explícito definido en el archivo
    de configuración. La categoría 'Unknown' se añade al final de la ordenación,
    recibiendo el código numérico más alto.

    Args:
        df (pd.DataFrame): El DataFrame con las columnas categóricas ya imputadas.

    Returns:
        pd.DataFrame: El DataFrame con las columnas ordinales codificadas numéricamente.
    """
    for col, order in ORDINAL_COLS.items():
        if col in df.columns:
            cat_type = CategoricalDtype(categories=order + ["Unknown"], ordered=True)
            df[col] = df[col].astype(cat_type).cat.codes.astype("int16")
    return df

def encode_nominals_and_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding (OHE) a las columnas nominales.

    Utiliza `pandas.get_dummies` para crear columnas binarias (0/1) para cada
    categoría en las columnas nominales. Se utiliza `drop_first=True` para
    eliminar una de las categorías y evitar la multicolinealidad.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame con las columnas nominales transformadas a formato OHE.
    """
    small = [c for c in BINARY_COLS + NOMINAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=small, drop_first=True)