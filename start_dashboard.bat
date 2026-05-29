@echo off
title Amazon India Sales Analytics — Dashboard Launcher
color 0A

echo.
echo ============================================================
echo   Amazon India Sales Analytics — Evaluation Launcher
echo ============================================================
echo.

REM ── Check PostgreSQL is reachable ───────────────────────────────
echo [1/2] Checking PostgreSQL connection...
python -c "import psycopg2; psycopg2.connect(dbname='amazon_sales',user='postgres',password='REDACTED',host='localhost',port=5432); print('OK')" 2>nul
if errorlevel 1 (
    echo.
    echo  WARNING: Cannot reach PostgreSQL on localhost:5432
    echo  The service should auto-start with Windows.
    echo  If not, open Services (services.msc) and start:
    echo    postgresql-x64-18
    echo.
    pause
)
echo  PostgreSQL is ready.
echo.

REM ── Launch Streamlit ────────────────────────────────────────────
echo [2/2] Launching Streamlit dashboard...
echo.
echo  Dashboard opening at: http://localhost:8501
echo.
echo  TO STOP: Press Ctrl+C in this window
echo ============================================================
echo.

cd /d "%~dp0"
python -m streamlit run streamlit_app/app.py --server.port 8501
