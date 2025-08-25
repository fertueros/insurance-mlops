# Diccionario de Datos: Predicción de Primas de Seguro

Este documento describe en detalle cada una de las características presentes en el conjunto de datos utilizado para el entrenamiento y la evaluación del modelo.

### Información General

| Ítem                  | Detalle                                                                                                                                                                                                                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fuente Original**   | [Insurance Premium Prediction Dataset](https://www.kaggle.com/datasets/schran/insurance-premium-prediction)                                                                                                                                                                                      |
| **Fuente del Proyecto** | [Kaggle Playground Series - S4E12](https://www.kaggle.com/competitions/playground-series-s4e12). Los datos de esta competencia fueron generados sintéticamente basándose en el dataset original para crear un problema de regresión más desafiante.                                                  |
| **Citación**            | Reade, W., & Park, E. (2024). *Regression with an Insurance Dataset*. Kaggle. https://kaggle.com/competitions/playground-series-s4e12                                                                                                                                                             |
| **Filas Totales**     | 1,200,000 (`train.csv`) + 800,000 (`test.csv`)                                                                                                                                                                                                                                                      |
| **Variable Objetivo**   | `Premium Amount` – Monto de la prima del seguro.                                                                                                                                                                                                                                                  |
| **Tipos de Variables**  | Numéricas, Categóricas y de Fecha. El dataset presenta desafíos del mundo real como valores faltantes, tipos de datos incorrectos y distribuciones sesgadas.                                                                                                                                            |

### Descripción de las Columnas

| Columna | Tipo de Dato | Descripción en Español |
| :--- | :--- | :--- |
| `id` | Numérico | Identificador único para cada registro de póliza. |
| `Age` | Numérico | Edad del individuo asegurado en años. |
| `Gender` | Categórico | Género del individuo asegurado (Ej: Male, Female). |
| `Annual Income` | Numérico | Ingresos anuales del individuo asegurado. Presenta una distribución sesgada. |
| `Marital Status` | Categórico | Estado civil del individuo asegurado (Ej: Single, Married, Divorced). |
| `Number of Dependents` | Numérico | Número de dependientes del asegurado. Contiene valores faltantes. |
| `Education Level` | Categórico | Nivel educativo más alto alcanzado (Ej: High School, Bachelor's, Master's, PhD). |
| `Occupation` | Categórico | Ocupación del individuo asegurado (Ej: Employed, Self-Employed, Unemployed). |
| `Health Score` | Numérico | Puntuación que representa el estado de salud. Presenta una distribución sesgada. |
| `Location` | Categórico | Tipo de ubicación de la residencia (Ej: Urban, Suburban, Rural). |
| `Policy Type` | Categórico | Tipo de póliza de seguro contratada (Ej: Basic, Comprehensive, Premium). |
| `Previous Claims` | Numérico | Número de reclamos previos realizados. Contiene valores atípicos (outliers). |
| `Vehicle Age` | Numérico | Antigüedad del vehículo asegurado en años. |
| `Credit Score` | Numérico | Puntuación de crédito del individuo asegurado. Contiene valores faltantes. |
| `Insurance Duration` | Numérico | Duración de la póliza de seguro en años. |
| `Policy Start Date` | Fecha | Fecha de inicio de la póliza de seguro. Originalmente como texto con formato incorrecto. |
| `Customer Feedback` | Categórico | Comentarios/valoración del cliente sobre el servicio. |
| `Smoking Status` | Categórico | Indica si el individuo es fumador (Ej: Yes, No). |
| `Exercise Frequency` | Categórico | Frecuencia con la que el individuo hace ejercicio (Ej: Daily, Weekly, Monthly, Rarely). |
| `Property Type` | Categórico | Tipo de propiedad que posee el asegurado (Ej: House, Apartment, Condo). |
| **`Premium Amount`** | **Numérico** | **(TARGET)** Monto de la prima del seguro a predecir. |