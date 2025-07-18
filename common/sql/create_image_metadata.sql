CREATE TABLE IF NOT EXISTS image_metadata (
    folder TEXT,
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    is_stored BOOL
);