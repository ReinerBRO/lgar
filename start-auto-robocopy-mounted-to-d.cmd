@echo off
setlocal EnableExtensions

rem Run this from Tianyi Windows inside the mounted Mac project directory.
rem It repeatedly calls the one-shot robocopy launcher every N seconds, default 5.

set "INTERVAL=%~1"
if "%INTERVAL%"=="" set "INTERVAL=5"

cd /d "%~dp0"

if not exist "%~dp0robocopy-mounted-to-d.cmd" (
  echo missing launcher: %~dp0robocopy-mounted-to-d.cmd
  exit /b 1
)

echo auto robocopy loop started: %CD% -^> D:\lgar
echo interval=%INTERVAL%s, press Ctrl+C to stop

:loop
echo [%DATE% %TIME%] scan start
call "%~dp0robocopy-mounted-to-d.cmd" quiet
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo [%DATE% %TIME%] robocopy failed: exit=%RC%
  echo rerun robocopy-mounted-to-d.cmd once for details
) else if not "%RC%"=="0" (
  echo [%DATE% %TIME%] copied or changed: robocopy exit=%RC%
) else (
  echo [%DATE% %TIME%] idle
)
timeout /t %INTERVAL% /nobreak >nul
goto loop
