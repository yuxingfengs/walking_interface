@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\launch.ps1" -Config "%~dp0..\experiments\step_narrow_gaussian\config.json" -Mode pretrain
set "run_exit=%errorlevel%"
if not "%run_exit%"=="0" pause
exit /b %run_exit%
