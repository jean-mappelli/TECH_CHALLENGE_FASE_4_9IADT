@echo off
setlocal
cd /d "%~dp0"

echo Preparando ambiente Python...
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (python -m venv .venv) else (py -3 -m venv .venv)
    if errorlevel 1 exit /b 1
)

echo Instalando dependencias...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist "outputs" mkdir outputs
echo Aplicacao disponivel em http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn medmonitor.api:app --host 127.0.0.1 --port 8000