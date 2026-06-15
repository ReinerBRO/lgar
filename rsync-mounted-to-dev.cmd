@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run this from Tianyi Windows inside the mounted Mac project directory.
rem Default mode is sync-only. Use quiet for auto loops, verify for a remote summary.
rem Optional file args sync only those project-relative paths.

set "MODE=sync"
if /i "%~1"=="help" goto usage
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--help" goto usage
if /i "%~1"=="sync" (
  set "MODE=sync"
  shift
) else if /i "%~1"=="quiet" (
  set "MODE=quiet"
  shift
) else if /i "%~1"=="verify" (
  set "MODE=verify"
  shift
)

set "RSYNC=C:\cwrsync\bin\rsync.exe"
set "SSH=C:\cwrsync\bin\ssh.exe"
set "SSH_RSYNC=C:/cwrsync/bin/ssh.exe"
set "SSH_CONFIG=/cygdrive/c/Users/Administrator/.ssh/config"
set "DST_ROOT=/gemini/space/private/zjc/goals/lgar"

if not exist "%RSYNC%" (
  echo missing rsync: %RSYNC%
  exit /b 1
)
if not exist "%SSH%" (
  echo missing ssh: %SSH%
  exit /b 1
)

set "DRIVE=%~d0"
set "DRIVE=%DRIVE::=%"
set "REST=%~p0"
set "REST=%REST:\=/%"
if "%REST:~-1%"=="/" set "REST=%REST:~0,-1%"
set "SRC_ROOT=/cygdrive/%DRIVE%%REST%"

"%SSH%" -T -F "%SSH_CONFIG%" dev "mkdir -p '%DST_ROOT%'" >nul
if errorlevel 1 exit /b 1

set "SOURCES="
if "%~1"=="" (
  call :add_source "curcpt/eval_long_context_public.py"
  if errorlevel 1 exit /b %ERRORLEVEL%
  call :add_source "scripts/zjc_eval_qwen25_3b_32k_public_multilen.sh"
  if errorlevel 1 exit /b %ERRORLEVEL%
  call :add_source "sync-m-to-dev-then-dev-to-d.cmd"
  if errorlevel 1 exit /b %ERRORLEVEL%
) else (
  :arg_loop
  if "%~1"=="" goto args_done
  call :add_source "%~1"
  if errorlevel 1 exit /b %ERRORLEVEL%
  shift
  goto arg_loop
)
:args_done

if /i "%MODE%"=="quiet" goto rsync_quiet

echo sync %SRC_ROOT%/ -^> dev:%DST_ROOT%/
echo mode=%MODE%
echo sources:
for %%F in (!SOURCES!) do echo   %%~F
echo.
"%RSYNC%" -rtv --checksum -s --modify-window=2 --no-perms --no-owner --no-group --omit-dir-times -R --rsync-path=/usr/local/bin/rsync -e "%SSH_RSYNC% -T -F %SSH_CONFIG%" ^
  --exclude=".git/" ^
  --exclude="__pycache__/" ^
  --exclude="*.pyc" ^
  --exclude=".pytest_cache/" ^
  --exclude=".mypy_cache/" ^
  --exclude=".ruff_cache/" ^
  --exclude=".venv/" ^
  --exclude="venv/" ^
  --exclude="env/" ^
  --exclude="build/" ^
  --exclude="dist/" ^
  --exclude="*.egg-info/" ^
  --exclude=".DS_Store" ^
  --exclude="data/" ^
  --exclude="logs/" ^
  --exclude="outputs/" ^
  --exclude="output/" ^
  --exclude="runs/" ^
  --exclude="tmp/" ^
  --exclude=".cache/" ^
  --exclude="wandb/" ^
  !SOURCES! ^
  "dev:%DST_ROOT%/"
if errorlevel 1 exit /b 1

if /i not "%MODE%"=="verify" (
  echo done
  exit /b 0
)

echo === verify remote summary ===
"%SSH%" -T -F "%SSH_CONFIG%" dev "cd '%DST_ROOT%' && pwd && find . -maxdepth 2 -type f | head -40"
if errorlevel 1 exit /b 1

echo verify ok
exit /b 0

:rsync_quiet
"%RSYNC%" -rtq --checksum -s --modify-window=2 --no-perms --no-owner --no-group --omit-dir-times -R --rsync-path=/usr/local/bin/rsync -e "%SSH_RSYNC% -T -F %SSH_CONFIG%" ^
  --exclude=".git/" ^
  --exclude="__pycache__/" ^
  --exclude="*.pyc" ^
  --exclude=".pytest_cache/" ^
  --exclude=".mypy_cache/" ^
  --exclude=".ruff_cache/" ^
  --exclude=".venv/" ^
  --exclude="venv/" ^
  --exclude="env/" ^
  --exclude="build/" ^
  --exclude="dist/" ^
  --exclude="*.egg-info/" ^
  --exclude=".DS_Store" ^
  --exclude="data/" ^
  --exclude="logs/" ^
  --exclude="outputs/" ^
  --exclude="output/" ^
  --exclude="runs/" ^
  --exclude="tmp/" ^
  --exclude=".cache/" ^
  --exclude="wandb/" ^
  !SOURCES! ^
  "dev:%DST_ROOT%/"
exit /b %ERRORLEVEL%

:add_source
set "REL=%~1"
set "REL=!REL:\=/!"
if "!REL!"=="" exit /b 0
if "!REL:~0,1!"=="/" (
  echo use project-relative paths, got: %~1
  exit /b 2
)
if "!REL:~1,1!"==":" (
  echo use project-relative paths, got: %~1
  exit /b 2
)
if not exist "!REL!" (
  echo missing local path: !REL!
  exit /b 2
)
set "SOURCES=!SOURCES! %SRC_ROOT%/./!REL!"
exit /b 0

:usage
echo Usage:
echo   rsync-mounted-to-dev.cmd [sync^|quiet^|verify] [relative-path ...]
echo.
echo Examples:
echo   rsync-mounted-to-dev.cmd verify
echo   rsync-mounted-to-dev.cmd verify curcpt/eval_long_context_public.py
exit /b 0
