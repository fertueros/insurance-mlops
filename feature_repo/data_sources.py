from feast import FileSource

train_proc = FileSource(
    path="../data/processed/train_proc.parquet",  # relativo a feature_repo/
    timestamp_field="event_timestamp",            # necesario para point-in-time
)
