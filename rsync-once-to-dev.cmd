@echo off
setlocal
cd /d "%~dp0"
title Rsync lgar to dev

set "PROJECT=lgar"
set "LOCAL_WIN=%~dp0"
set "LOCAL_CYG=/cygdrive/d/lgar/"
set "REMOTE_DIR=/gemini/space/private/zjc/goals/lgar/"
set "RSYNC=C:\cwrsync\bin\rsync.exe"
set "SSH=C:\cwrsync\bin\ssh.exe"
set "SSH_CYG=C:/cwrsync/bin/ssh.exe"
set "SSH_CONFIG=/cygdrive/c/Users/Administrator/.ssh/config"
set "WATCH_INTERVAL=5"

echo Project: %PROJECT%
echo Local:   %LOCAL_CYG%
echo Remote:  dev:%REMOTE_DIR%
echo.

if not exist "%LOCAL_WIN%" (
  echo Missing local project directory: %LOCAL_WIN%
  set "STATUS=1"
  goto finish
)

if not exist "%RSYNC%" (
  echo Missing %RSYNC%. Run D:\rsync-offline\install-windows-cwrsync.cmd first.
  set "STATUS=1"
  goto finish
)

if not "%~2"=="" set "WATCH_INTERVAL=%~2"

call :ensure_remote
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto finish
)

if /I "%~1"=="watch" goto watch

call :sync_once
set "STATUS=%ERRORLEVEL%"
goto finish

:watch
echo Watching %LOCAL_WIN% and syncing to dev:%REMOTE_DIR%
echo Press Ctrl+C to stop. Interval: %WATCH_INTERVAL%s
echo.

:watch_loop
call :sync_once
if errorlevel 1 echo rsync failed with %ERRORLEVEL%, retrying after %WATCH_INTERVAL%s
timeout /t %WATCH_INTERVAL% /nobreak >nul
goto watch_loop

:ensure_remote
"%SSH%" -T -F %SSH_CONFIG% dev "mkdir -p /gemini/space/private/zjc/goals/lgar/; if command -v rsync >/dev/null 2>&1; then :; elif [ -x /root/.local/bin/rsync ]; then mkdir -p /usr/local/bin && cp /root/.local/bin/rsync /usr/local/bin/rsync && chmod 755 /usr/local/bin/rsync; else echo 'dev rsync missing: install rsync first'; exit 10; fi; command -v rsync && rsync --version | head -n 1"
exit /b %ERRORLEVEL%

:sync_once
"%RSYNC%" -av --itemize-changes --progress -s --rsync-path=/usr/local/bin/rsync -e "%SSH_CYG% -T -F %SSH_CONFIG%" "%LOCAL_CYG%" dev:%REMOTE_DIR%
exit /b %ERRORLEVEL%

:finish
echo.
if "%STATUS%"=="0" (
  echo Rsync completed successfully.
) else (
  echo Rsync failed with exit code %STATUS%.
)
echo.
pause
exit /b %STATUS%
