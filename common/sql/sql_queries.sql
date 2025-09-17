-- SQLite
SELECT
    count(*) as nb_row,
    min(timestamp) as start,
    max(timestamp) as END
FROM image_metadata
;
-- SELECT
--     *
-- FROM image_metadata_archive
-- limit 10
-- ;
SELECT
    date(timestamp) as date,
    folder,
    count(*) as nb_row,
    min(timestamp) as start,
    max(timestamp) as END
FROM image_metadata
group by 1,2
;

SELECT
    count(*) as nb_row,
    min(created_date) as start,
    max(created_date) as END
FROM serving_prediction