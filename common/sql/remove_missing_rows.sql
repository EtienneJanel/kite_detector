-- Delete from serving_prediction where id is not in image_metadata
DELETE FROM serving_prediction
WHERE id NOT IN (
    SELECT id FROM image_metadata
);

-- Delete from image_metadata where id is not in serving_prediction
DELETE FROM image_metadata
WHERE id NOT IN (
    SELECT id FROM serving_prediction
);