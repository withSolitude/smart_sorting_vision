@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0.venv_sam3\Scripts\python.exe" -u "%~dp0interactive_sam3.py" --config "%~dp0sam3_side_config.yaml" %*
if errorlevel 1 pause
