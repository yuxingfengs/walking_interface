@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\launch.ps1" -Config "%~dp0..\experiments\step_wide_gaussian\config.json" -Mode check
if errorlevel 1 pause
