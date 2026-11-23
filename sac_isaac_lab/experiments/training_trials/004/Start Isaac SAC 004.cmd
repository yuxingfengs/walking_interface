@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\..\launch.ps1" -Mode train -Experiment 004 %*
set "run_exit=%errorlevel%"
if not "%run_exit%"=="0" pause
exit /b %run_exit%
