# Changelog
Todas las notas de cambios de este proyecto siguen [SemVer](https://semver.org) y el formato de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).

## [Unreleased]

## [1.0.0] - 2025-08-25
### Added

* **Pipeline productivo de ML** (reproducible con DVC):

  * Módulos principales en `src/`: `clean.py`, `features.py`, `data.py`, `models.py`, `train.py`, `tune.py`, `splits.py`, `mlflow_setup.py`.
  * Stages de DVC: `preprocess`, `feast_build`, `tune_hgb`, `tune_xgb`, `tune_lgbm`, `train` (stacking), `predict_test`, `register_model`.
* **Feature Store (Feast)** para *point-in-time correctness*:

  * `feature_repo/` con `feature_store.yaml`, `data_sources.py`, `features.py`.
  * **Entity**: `id`; **timestamp**: `event_timestamp`.
  * Generación de `data/feast/training_set.parquet` para entrenamiento consistente.
* **Tuning de hiperparámetros con Optuna** (HGB/XGB/LGBM):

  * Mejores parámetros guardados en `artifacts/optuna/*_best_params.json`.
  * Experimentos rastreados en MLflow (DagsHub).
* **Stacking (HGB + XGB + LGBM)** con meta-modelo `RidgeCV` y predicciones **OOF**:

  * Validación cruzada (K=5). **RMSLE (CV)** del ensamble: **1.1289**.
  * Artefactos: `models/stacking_ensemble.joblib` y `reports/oof_predictions.csv`.
* **Inferencia sobre test sin etiquetas**:

  * Script `scripts/predict_test.py` → `reports/predictions_test.csv` (`id,y_pred`).
  * Alineación de columnas respecto al set de entrenamiento servido por Feast.
* **Registro del modelo** en **MLflow Model Registry (DagsHub)**:

  * Script `scripts/register_model.py` (PyFunc) y transición a *Staging*.
* **Documentación**:

  * `README.md` completo con diagrama Mermaid, instrucciones de reproducción, enlaces a DagsHub/MLflow.
  * `docs/DATA_DICTIONARY.md` y `docs/MODEL_CARD.md`.

### Changed

* Estandarización del preprocesamiento y *feature engineering* para consumo por GBMs:

  * Filtrado estricto de **solo numéricas/bool** en el *feature matrix* de entrenamiento.
  * Alineación de columnas entre *train* (Feast) y *test* en inferencia.
* Entrenamiento basado en **Feast** (training set materializado) para evitar fugas temporales.
* Consolidación de métrica **RMSLE** y utilidades de modelos en `src/models.py`.

### Fixed

* Error de LightGBM por `event_timestamp` con dtype `datetime` en la matriz de features → ahora se excluye del set de entrenamiento/predicción.

### Notes

* `test.csv` **no** contiene `target`; el desempeño oficial del proyecto se reporta con **CV/OOF**.
* Primera versión **estable** del pipeline end-to-end. A partir de aquí, cambios incompatibles se reflejarán en la parte **MAJOR** de SemVer.

## [0.2.0] - 2025-08-02
### Added
- EDA inicial completo en `notebooks/01_eda.ipynb` con decisiones documentadas.
- Utilidades de EDA en `src/eda_utils.py` (coerción de tipos, parsing de fechas, features de calendario, reporte de calidad).
- Configuración de tracking remoto de MLflow para el EDA (experimento `eda-baseline`), con logging de artefactos y parámetros.
- `src/logging_utils.py` con **loguru** para trazabilidad (logs rotados en `logs/` y subida del log a MLflow).
- Gráficos y reportes en `reports/`: calidad, faltantes (missingno), distribución del target, descriptivos numéricos y outliers. Subida de reportes a mlflow.
- Exploraciones de **PCA / UMAP** (no productivas aún).
- Chequeos para RMSLE (no-negatividad del target) y propuesta de entrenar sobre `log1p(y)`.

### Changed
- Categorización y orden de variables categóricas (e.g., `Policy Type`, `Education Level`) para análisis consistente.
- Ajustes de `.gitignore`/placeholders para garantizar persistencia de estructura en `reports/` y `logs/`.

### Decisions (guía para etapas siguientes)
- **Imputación**: medianas en numéricos sesgados; `Unknown` en categóricas con NA.
- **Tratamiento de outliers**: winsorize p99 en `Annual Income` y `Previous Claims`.
- **Métrica**: optimizar **RMSLE** (entrenar con `log1p(y)` y retransformar al evaluar).

## [0.1.0] - 2025-07-29
### Added
- Estructura base del proyecto con Cookiecutter Data Science.
- Flujo de ramas: `main`, `develop` y convenciones de `feature/*` y `chore/*`.
- Inicialización de DVC (`dvc init`) y versionado de datos crudos.
- Tracking con DVC del dataset `data/raw/train.csv`.
- Configuración de remoto de datos en DagsHub y primer `dvc push` exitoso.
- Configuración de tracking remoto de MLflow apuntando a DagsHub (URI definida y autenticación documentada mediante variables de entorno).

### Changed
- `.gitignore` ajustado para permitir punteros `*.dvc` bajo `data/` y mantener datasets fuera de Git.
- Placeholders en `data/processed/` para preservar la estructura del proyecto.

[Unreleased]: https://github.com/fertueros/insurance-mlops/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/fertueros/insurance-mlops/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/fertueros/insurance-mlops/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fertueros/insurance-mlops/tree/v0.1.0

