@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner\module.ps1" -Mode continue
set "rc=%errorlevel%"
if not "%rc%"=="0" pause
exit /b %rc%
