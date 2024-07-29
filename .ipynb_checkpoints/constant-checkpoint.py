# import os 
# import chromadb
# from chromadb.config import Settings 


# CHROMA_SETTINGS = Settings(
#         chroma_db_impl='duckdb+parquet',
#         persist_directory='db',
#         anonymized_telemetry=False
# )

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);
