@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\launch.ps1" -Config "%~dp0..\experiments\walk_only\config.json" -Mode train
set "entryExit=%errorlevel%"
if not "%entryExit%"=="0" pause
exit /b %entryExit%
