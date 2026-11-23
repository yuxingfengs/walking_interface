@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_dependencies.ps1"
set "setup_exit=%errorlevel%"
pause
exit /b %setup_exit%
