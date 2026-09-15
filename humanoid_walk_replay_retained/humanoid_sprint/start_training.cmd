@echo off
chcp 65001 >nul
cd /d "%~dp0"
"C:\Users\fengy\anaconda3\envs\webot\python.exe" scripts/results_pipeline.py train-report --open
if errorlevel 1 echo Training or quick report failed. See the error above.
pause
