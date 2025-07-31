import pandas as pd, numpy as np

CATEGORICAL_ORDERED = {
    "Policy Type": ["Basic", "Comprehensive", "Premium"],
    "Education Level": ["High School", "Bachelor's", "Master's", "PhD"],
}
CATEGORICAL_BINARY = ["Gender", "Smoking Status"]
CATEGORICAL_NOMINAL = ["Marital Status","Occupation","Location","Exercise Frequency","Property Type", "Customer Feedback"]
DATE_LIKE = ["Policy Start Date"]
TARGET = "Premium Amount"

def read_raw(path="data/raw/train.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["Age","Annual Income","Number of Dependents","Health Score",
                "Previous Claims","Vehicle Age","Credit Score","Insurance Duration", TARGET]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_BINARY + CATEGORICAL_NOMINAL + list(CATEGORICAL_ORDERED.keys()):
        if c in df.columns:
            df[c] = df[c].astype("category")
            if c in CATEGORICAL_ORDERED:
                df[c] = df[c].cat.set_categories(CATEGORICAL_ORDERED[c], ordered=True)
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_LIKE:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_LIKE:
        if c in df.columns and np.issubdtype(df[c].dtype, np.datetime64):
            df[f"{c}_year"]  = df[c].dt.year.astype("float64")
            df[f"{c}_month"] = df[c].dt.month.astype("float64")
            df[f"{c}_dow"]   = df[c].dt.dayofweek.astype("float64")
            df[f"{c}_m_sin"] = np.sin(2*np.pi*df[f"{c}_month"]/12.0)
            df[f"{c}_m_cos"] = np.cos(2*np.pi*df[f"{c}_month"]/12.0)
    return df

def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un reporte básico de calidad de datos (tipos, nulos, únicos)."""
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean()*100).round(2),
    })
    rep = rep.reset_index().rename(columns={"index": "feature"})
    return rep.sort_values("pct_missing", ascending=False)

def rmsle_safe_check(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Verifica si la columna objetivo contiene valores que romperían el cálculo de RMSLE."""
    info = {"hay_negativos": False, "no_finito": 0}
    if target in df.columns:
        s = df[target]
        info["hay_negativos"] = bool((s < 0).any())
        info["no_finito"] = int((~np.isfinite(s)).sum())
    return info
