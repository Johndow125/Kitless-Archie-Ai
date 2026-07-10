@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw -B app.py
) else (
    start "" python -B app.py
)
exit /b
