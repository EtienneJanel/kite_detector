CREATE TABLE IF NOT EXISTS serving_prediction (
    id TEXT PRIMARY KEY,
    created_date TIMESTAMP,
    predictions JSON
);