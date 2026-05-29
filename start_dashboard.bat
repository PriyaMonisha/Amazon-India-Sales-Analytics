@echo off
title Amazon India Sales Analytics — Dashboard Launcher
color 0A

echo.
echo ============================================================
echo   Amazon India Sales Analytics — Evaluation Launcher
echo ============================================================
echo.

REM ── Step 1: Check Docker is running ────────────────────────────
echo [1/4] Checking Docker Desktop...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Docker is NOT running.
    echo  Please open Docker Desktop, wait for the green icon,
    echo  then run this script again.
    echo.
    pause
    exit /b 1
)
echo  Docker is running.
echo.

REM ── Step 2: Start PostgreSQL + Redis ───────────────────────────
echo [2/4] Starting PostgreSQL + Redis containers...
docker compose up postgres redis -d
if errorlevel 1 (
    echo  Failed to start containers. Check docker-compose.yml.
    pause
    exit /b 1
)
echo  Containers started.
echo.

REM ── Step 3: Wait for PostgreSQL to be ready ────────────────────
echo [3/4] Waiting for PostgreSQL to be ready...
:wait_loop
docker exec amazon_postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_loop
)
echo  PostgreSQL is ready.
echo.

REM ── Step 4: Launch Streamlit dashboard ─────────────────────────
echo [4/4] Launching Streamlit dashboard...
echo.
echo  Dashboard will open at: http://localhost:8501
echo.
echo  TO STOP: Close this window, then run stop_dashboard.bat
echo.
echo ============================================================
echo.

python -m streamlit run streamlit_app/app.py --server.port 8501 --server.headless false
