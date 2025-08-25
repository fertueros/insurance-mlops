# Model Card: `insurance-premium-stacking-v1.0`

Esta Model Card proporciona información detallada sobre el modelo de ensamblaje de Stacking desarrollado para predecir primas de seguros.

---

### Detalles del Modelo

-   **Desarrollado por:** [Tu Nombre]
-   **Fecha:** 25 de Agosto de 2025
-   **Versión del Modelo:** 1.0.0
-   **Tipo de Modelo:** Ensamblaje de Regresión (Stacking).
    -   **Modelos Base (Nivel 0):** HistGradientBoostingRegressor, XGBRegressor, LGBMRegressor.
    -   **Meta-Modelo (Nivel 1):** RidgeCV (Regresión Lineal con Regularización).

---

### Uso Previsto

-   **Uso Primario:** Estimar el monto de la prima de un seguro basándose en las características del cliente y de la póliza. El objetivo es asistir a los analistas de tarificación y mejorar la precisión de las cotizaciones automáticas.
-   **Usuarios Previstos:** Analistas de riesgo, equipos de producto en compañías de seguros, científicos de datos.
-   **Fuera de Alcance:** Este modelo no debe ser utilizado como la única herramienta para la aprobación o denegación final de una póliza. No debe usarse para tomar decisiones críticas de negocio sin la supervisión de un experto humano. El modelo no considera factores macroeconómicos o eventos catastróficos.

---

### Factores de Evaluación

-   **Grupos de Datos Relevantes:** El modelo fue entrenado en un dataset sintético que simula una población de clientes diversa en términos de edad, ingresos y ubicación.
-   **Métrica Principal:** Raíz del Error Logarítmico Cuadrático Medio (RMSLE). Esta métrica fue elegida por su robustez ante distribuciones de precios sesgadas y porque penaliza más las subestimaciones que las sobreestimaciones en una escala relativa.
-   **Resultados de la Métrica:**
    -   **RMSLE en Validación Cruzada (5-folds):** **1.1289**
    -   **RMSLE en Conjunto de Test (Kaggle):** **1.1383**

---

### Datos de Entrenamiento

-   **Fuente:** El modelo fue entrenado utilizando el archivo `train.csv` de la competencia [Kaggle Playground Series - S4E12](https://www.kaggle.com/competitions/playground-series-s4e12).
-   **Preprocesamiento:** El pipeline de entrenamiento incluye:
    -   Imputación de valores numéricos faltantes con la mediana.
    -   Winsorización de outliers en columnas de ingresos y reclamos.
    -   Ingeniería de características a partir de fechas (año, mes, representaciones cíclicas).
    -   Imputación de valores categóricos faltantes con la etiqueta "Unknown".
    -   Codificación Ordinal y One-Hot Encoding para variables categóricas.
-   **Toda la lógica de preprocesamiento está encapsulada y es reproducible** a través del pipeline de DVC (`dvc repro`).

---

### Consideraciones Éticas y Limitaciones

-   **Sesgos Potenciales:** Dado que el dataset es sintético, los sesgos inherentes a los datos del mundo real pueden no estar completamente representados. Sin embargo, el modelo podría aprender correlaciones espurias si existieran en los datos de entrenamiento. Un análisis de equidad (fairness) debería realizarse antes de un despliegue en producción para asegurar que el modelo no discrimina en base a características sensibles como `Gender` o `Location`.
-   **Dependencia de la Calidad de los Datos:** El rendimiento del modelo depende críticamente de que los datos de entrada tengan la misma estructura y distribución que los datos de entrenamiento. Un monitoreo de la deriva de datos (data drift) sería necesario en un entorno productivo.
-   **Interpretabilidad:** Si bien los modelos base (GBMs) son complejos, el meta-modelo lineal (`RidgeCV`) podría ofrecer cierta interpretabilidad sobre cómo se ponderan las "opiniones" de los modelos base. Para una interpretabilidad más profunda a nivel de característica, se requerirían técnicas como SHAP.