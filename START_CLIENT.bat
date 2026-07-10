@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set LOG=%~dp0client_data\client_startup.log
if not exist "%~dp0client_data" mkdir "%~dp0client_data"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3 and try again.
    pause
    exit /b 1
)
python -B -c "import importlib.util; spec=importlib.util.spec_from_file_location('kitless_client', r'%~dp0app.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)" 1>"%LOG%" 2>&1
if errorlevel 1 (
    echo Kitless Client failed to start:
    type "%LOG%"
    pause
    exit /b 1
)
where pythonw >nul 2>nul
if errorlevel 1 (
    start "" python -B app.py
) else (
    start "" pythonw -B app.py
)
exit /b
