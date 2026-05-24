import sys

# Makes both `config` and `src.*` importable inside Airflow containers.
# docker-compose mounts ./src → /opt/airflow/src and ./config.py → /opt/airflow/config.py.
# Import this module as the very first import in every DAG file, before any src.* import.
sys.path.insert(0, "/opt/airflow")
