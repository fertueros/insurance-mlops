# src/splits.py
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

def stratified_regression_folds(y, n_splits=5, seed=42, n_bins=10):
    """Crea folds de validación cruzada estratificados.

    Asegurar que la distribución de la variable objetivo (y) sea similar en 
    cada fold de entrenamiento y validación.

    Args:
        y (pd.Series): La variable objetivo (continua) del dataset.
        n_splits (int, optional): El número de folds para la validación cruzada.
                                  Defaults to 5.
        seed (int, optional): Semilla aleatoria. Defaults to 42.
        n_bins (int, optional): El número de bins para discretizar la variable 'y'.
                                Defaults to 10.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]:
            Una lista de tuplas. Cada tupla contiene dos arrays de NumPy:
            el primero con los índices para el conjunto de entrenamiento y el
            segundo con los índices para el conjunto de validación de ese fold.
    """
    yb = pd.qcut(np.log1p(y), q=n_bins, duplicates="drop").astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(y, yb))
