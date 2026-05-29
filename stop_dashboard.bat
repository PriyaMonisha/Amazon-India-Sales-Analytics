@echo off
title Stopping Dashboard
color 0C

echo.
echo ============================================================
echo   Stopping Amazon India Sales Analytics
echo ============================================================
echo.

echo Stopping containers (data is preserved)...
docker compose down
echo.
echo Done. You can now close Docker Desktop to free RAM.
echo.
echo NOTE: Your data is safe. Next time just run start_dashboard.bat
echo ============================================================
echo.
pause
