@echo off
cd /d "%~dp0" || exit /b 1
python -X utf8 "%~dp0compare.py" --open
if errorlevel 1 pause
