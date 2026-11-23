@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Move Isaac Runtime.ps1" %*
exit /b %errorlevel%
