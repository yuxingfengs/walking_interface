@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\launch.ps1" -Config "%~dp0..\experiments\cycle_mean_gaussian\config.json" -Mode train
if errorlevel 1 pause
