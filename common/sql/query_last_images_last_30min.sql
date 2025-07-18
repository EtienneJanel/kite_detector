SELECT 
    id AS image_id,
    timestamp,
    folder
FROM image_metadata
WHERE timestamp >= datetime('now', '-30 minutes')
    AND is_stored = TRUE
ORDER BY timestamp DESC