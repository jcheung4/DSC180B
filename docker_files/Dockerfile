# Use an official PostgreSQL image as the base image
FROM postgres:latest

# Set environment variables
ENV POSTGRES_DB mydatabase
ENV POSTGRES_USER dsc180b
ENV POSTGRES_PASSWORD dsc180b

# Copy SQL script to initialize the database and table
COPY init.sql /docker-entrypoint-initdb.d/

# Expose the PostgreSQL port
EXPOSE 5432