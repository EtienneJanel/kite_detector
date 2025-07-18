SELECT 
    id as image_id,
    timestamp,
    folder,
    is_stored
FROM image_metadata
ORDER BY 2 DESC
LIMIT 2