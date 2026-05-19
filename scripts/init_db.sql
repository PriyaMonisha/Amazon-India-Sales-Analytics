-- PostgreSQL initialization script
-- Runs automatically on first postgres container start
-- (mounted to /docker-entrypoint-initdb.d/)

-- Create mlflow metadata database
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

-- Create airflow metadata database
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO postgres;
