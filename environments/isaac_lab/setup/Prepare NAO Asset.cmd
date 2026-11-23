@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\..\sac_isaac_lab\launch.ps1" -Mode prepare
if errorlevel 1 pause