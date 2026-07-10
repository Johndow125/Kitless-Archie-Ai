@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
echo Starting standalone Kitless Client...
python -B app.py
pause
