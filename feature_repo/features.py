from feast import Entity, Field, FeatureView, ValueType
from feast.types import Float32, Float64, Int32
from datetime import timedelta
from data_sources import train_proc

# Usa tu columna existente 'id' como join key (no hace falta renombrar)
record = Entity(name="record", join_keys=["id"], value_type=ValueType.INT32)

schema = [
    # --- Features Numéricas ---
    Field(name="Age", dtype=Float32),
    Field(name="Annual Income", dtype=Float64),
    Field(name="Number of Dependents", dtype=Float64),
    Field(name="Health Score", dtype=Float64),
    Field(name="Previous Claims", dtype=Float32), # Es float por la winsorización
    Field(name="Vehicle Age", dtype=Float32),
    Field(name="Credit Score", dtype=Float32),
    Field(name="Insurance Duration", dtype=Float32),

    # --- Features de Fecha ---
    Field(name="psd_year", dtype=Float32),
    Field(name="psd_month", dtype=Float32),
    Field(name="psd_dow", dtype=Float32),
    Field(name="psd_month_sin", dtype=Float32),
    Field(name="psd_month_cos", dtype=Float32),

    # --- Features Categóricas Codificadas ---
    # Ordinales (ahora son números)
    Field(name="Policy Type", dtype=Int32),
    Field(name="Education Level", dtype=Int32),
    Field(name="Customer Feedback", dtype=Int32),

    # Binarias y Nominales (después de OHE con drop_first=True)
    # Nota: El nombre de la columna será `NombreOriginal_ValorPositivo`
    Field(name="Gender_Male", dtype=Int32),
    Field(name="Gender_Unknown", dtype=Int32),
    Field(name="Smoking Status_Yes", dtype=Int32),
    Field(name="Smoking Status_Unknown", dtype=Int32),
    Field(name="Marital Status_Married", dtype=Int32),
    Field(name="Marital Status_Single", dtype=Int32),
    Field(name="Marital Status_Unknown", dtype=Int32),
    
    Field(name="Occupation_Self-Employed", dtype=Int32),
    Field(name="Occupation_Unemployed", dtype=Int32),
    Field(name="Occupation_Unknown", dtype=Int32),
    Field(name="Location_Suburban", dtype=Int32),
    Field(name="Location_Urban", dtype=Int32),
    Field(name="Location_Unknown", dtype=Int32),
    
    Field(name="Exercise Frequency_Monthly", dtype=Int32),
    Field(name="Exercise Frequency_Rarely", dtype=Int32),
    Field(name="Exercise Frequency_Weekly", dtype=Int32),
    Field(name="Exercise Frequency_Unknown", dtype=Int32),
    Field(name="Property Type_Condo", dtype=Int32),
    Field(name="Property Type_House", dtype=Int32),
    Field(name="Property Type_Unknown", dtype=Int32),

]

premium_view = FeatureView(
    name="premium_features",
    entities=[record],
    ttl=timedelta(days=365),
    schema=schema,
    source=train_proc,
)
