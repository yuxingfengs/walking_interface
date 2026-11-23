@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_runtime_check.ps1"
set "check_exit=%errorlevel%"
pause
exit /b %check_exit%
