\c mydatabase;

CREATE TABLE poles (
    id SERIAL PRIMARY KEY,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    type VARCHAR(6)
);

INSERT INTO poles (latitude, longitude, type) VALUES
    (32.8517336, -117.1965509, 'Wooden'),
    (32.8516457, -117.1961353, 'Wooden'),
    (32.8515214, -117.1954883, 'Wooden'),
    (32.8515214, -117.1954883, 'Wooden'),