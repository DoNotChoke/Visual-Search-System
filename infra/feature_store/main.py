from __future__ import annotations

from datetime import datetime

from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Array, Float32, UnixTimestamp
from feast.value_type import ValueType

image = Entity(
    name="image",
    join_keys=["img_path"],
    value_type=ValueType.STRING,
    description="Unique image identifier (e.g., s3 uri or canonical image path).",
)

image_embeddings_source = FileSource(
    name="image_embeddings_source",
    path="image_embeddings.parquet",
    timestamp_field="event_timestamp",
)

image_embeddings = FeatureView(
    name="image_embeddings",
    entities=[image],
    schema=[
        Field(
            name="vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="L2",
        ),
        Field(name="s3_uri", dtype=String),
        Field(name="category", dtype=String),
        Field(name="event_timestamp", dtype=UnixTimestamp),
    ],
    source=image_embeddings_source,
)