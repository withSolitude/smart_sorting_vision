@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
    echo Drag one image file onto this script, or run: run_test.cmd "image.jpg"
    pause
    exit /b 1
)
"%~dp0.venv_sam3\Scripts\python.exe" -u "%~dp0test_sam3.py" "%~1" --show
if errorlevel 1 pause
