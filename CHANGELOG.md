# Changelog
Todas las notas de cambios de este proyecto siguen [SemVer](https://semver.org) y el formato de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/fertueros/insurance-mlops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fertueros/insurance-mlops/tree/v0.1.0

