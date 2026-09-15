@echo off
chcp 65001 >nul
cd /d "%~dp0"
"C:\Users\fengy\anaconda3\envs\webot\python.exe" scripts/results_pipeline.py full-report --open
if errorlevel 1 echo Full report failed. See the error above.
pause
