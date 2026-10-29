@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv_sam3\Scripts\python.exe" (
    python -m venv --system-site-packages "%~dp0.venv_sam3"
    if errorlevel 1 exit /b 1
)
"%~dp0.venv_sam3\Scripts\python.exe" -m pip install -r "%~dp0requirements_sam3.txt"
if errorlevel 1 pause
