@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner\module.ps1" -Mode check
set "rc=%errorlevel%"
if not "%rc%"=="0" pause
exit /b %rc%
