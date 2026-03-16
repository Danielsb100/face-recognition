@echo off
echo Starting Facial Recognition System...
cd /d "%~dp0"
.\venv\Scripts\python.exe gui.py
if %errorlevel% neq 0 (
    echo.
    echo Error: The application crashed.
    pause
)
