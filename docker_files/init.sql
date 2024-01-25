\c mydatabase;

CREATE TABLE poles (
    id SERIAL PRIMARY KEY,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    type VARCHAR(6)
);

INSERT INTO poles (latitude, longitude, type) VALUES
    (32.7455211, -117.0242656, 'Metal'),
    (32.745515, -117.023937, 'Wooden');