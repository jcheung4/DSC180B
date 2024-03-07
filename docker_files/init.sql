\c mydatabase;

CREATE TABLE poles (
    id SERIAL PRIMARY KEY,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    type VARCHAR(6)
);

CREATE TABLE poles_sample (
    id SERIAL PRIMARY KEY,
    "GIS_LATITUDE" DOUBLE PRECISION,
    "GIS_LONGITUDE" DOUBLE PRECISION,
    "GIS_OH_MATERIAL" VARCHAR(6)
);

COPY poles_sample("GIS_LATITUDE", "GIS_LONGITUDE", "GIS_OH_MATERIAL")
FROM '/docker-entrypoint-initdb.d/poles_sample.csv' DELIMITER ',' CSV HEADER;

INSERT INTO poles (latitude, longitude, type)
SELECT "GIS_LATITUDE"::DOUBLE PRECISION, 
       "GIS_LONGITUDE"::DOUBLE PRECISION, 
       CASE 
           WHEN "GIS_OH_MATERIAL" = 'WOOD' THEN 'Wooden'
           WHEN "GIS_OH_MATERIAL" = 'STEEL' THEN 'Metal'
           ELSE "GIS_OH_MATERIAL"
       END
FROM poles_sample;

INSERT INTO poles (latitude, longitude, type) VALUES
    (32.8517336, -117.1965509, 'Wooden'),
    (32.8516457, -117.1961353, 'Wooden'),
    (32.8515214, -117.1954883, 'Wooden'),
    (32.8515214, -117.1954883, 'Wooden'),
    (32.8208604, -117.1861909, 'Metal'),
    (32.8204763, -117.1861693, 'Wooden'),
    (32.8203362, -117.1861627, 'Wooden'),
    (32.8201549, -117.1861559, 'Wooden'),
    (32.820064, -117.1861533, 'Metal'),
    (32.6781706, -117.0986694, 'Wooden'),
    (32.6782575, -117.0983719, 'Wooden'),
    (32.6783658, -117.0979953, 'Wooden'),
    (32.6783413, -117.097868, 'Wooden'),
    (32.6783692, -117.0977638, 'Wooden'),
    (32.6784841, -117.0973523, 'Wooden'),
    (32.6784841, -117.0973523, 'Wooden'),
    (32.6784841, -117.0973523, 'Wooden'),
    (32.6784841, -117.0973523, 'Wooden'),
    (32.6784841, -117.0973523, 'Wooden'),
    (32.7077154, -117.084068, 'Wooden'),
    (32.7077132, -117.0836056, 'Wooden'),
    (32.70771, -117.0832791, 'Wooden')