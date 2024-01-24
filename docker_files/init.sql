-- Switch to the newly created database
\c mydatabase;

CREATE TABLE poles (
    id SERIAL PRIMARY KEY,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

INSERT INTO poles (latitude, longitude) VALUES
    (32.7455211, -117.0242656),
    (32.745515, -117.023937);