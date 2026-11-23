@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_standing.ps1" -StandingComparison ddpg %*
set "check_exit=%errorlevel%"
if not "%check_exit%"=="0" pause
exit /b %check_exit%
