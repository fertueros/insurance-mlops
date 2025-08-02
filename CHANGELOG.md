# Changelog
Todas las notas de cambios de este proyecto siguen [SemVer](https://semver.org) y el formato de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/fertueros/insurance-mlops/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fertueros/insurance-mlops/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fertueros/insurance-mlops/tree/v0.1.0

