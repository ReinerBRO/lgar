@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run this from Tianyi Windows on the mounted Mac project path:
rem   M:\h1syu1\PythonProjects\goals\lgar
rem It pushes M: -> dev first, then pulls dev -> D:\goals\lgar.
rem Optional args are project-relative paths. With no args, it syncs the current eval hotfix files.

set "RSYNC=C:\cwrsync\bin\rsync.exe"
set "SSH=C:\cwrsync\bin\ssh.exe"
set "SSH_RSYNC=C:/cwrsync/bin/ssh.exe"
set "SSH_CONFIG=/cygdrive/c/Users/Administrator/.ssh/config"
set "DEV_ROOT=/gemini/space/private/zjc/goals/lgar"
set "D_DST=D:\goals\lgar"
set "D_DST_CYG=/cygdrive/d/goals/lgar"

if not exist "%RSYNC%" (
  echo missing rsync: %RSYNC%
  exit /b 1
)
if not exist "%SSH%" (
  echo missing ssh: %SSH%
  exit /b 1
)
if not exist "%~dp0rsync-mounted-to-dev.cmd" (
  echo missing launcher: %~dp0rsync-mounted-to-dev.cmd
  exit /b 1
)
if not exist "%D_DST%" mkdir "%D_DST%"
if errorlevel 1 exit /b 1

set "PULL_SOURCES="
set "SYNC_ARGS="
if "%~1"=="" (
  call :add_pull_source "curcpt/eval_long_context_public.py"
  if errorlevel 1 exit /b %ERRORLEVEL%
  call :add_pull_source "scripts/zjc_eval_qwen25_3b_32k_public_multilen.sh"
  if errorlevel 1 exit /b %ERRORLEVEL%
  call :add_pull_source "sync-m-to-dev-then-dev-to-d.cmd"
  if errorlevel 1 exit /b %ERRORLEVEL%
) else (
  :arg_loop
  if "%~1"=="" goto args_done
  call :add_pull_source "%~1"
  if errorlevel 1 exit /b %ERRORLEVEL%
  shift
  goto arg_loop
)
:args_done

echo === step 1/2: M -^> dev ===
call "%~dp0rsync-mounted-to-dev.cmd" sync !SYNC_ARGS!
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo M -^> dev failed: exit=%RC%
  exit /b %RC%
)

echo.
echo === step 2/2: dev -^> D ===
echo pull dev:%DEV_ROOT%/ -^> %D_DST%\
"%RSYNC%" -rtv --checksum -s --modify-window=2 --no-perms --no-owner --no-group --omit-dir-times -R --rsync-path=/usr/local/bin/rsync -e "%SSH_RSYNC% -T -F %SSH_CONFIG%" ^
  !PULL_SOURCES! ^
  "%D_DST_CYG%/"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo dev -^> D failed: exit=%RC%
  exit /b %RC%
)

echo.
echo === verify D files ===
dir /b "%D_DST%\curcpt\eval_long_context_public.py" "%D_DST%\scripts\zjc_eval_qwen25_3b_32k_public_multilen.sh" 2>nul
echo done
exit /b 0

:add_pull_source
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
set "PULL_SOURCES=!PULL_SOURCES! dev:%DEV_ROOT%/./!REL!"
set "SYNC_ARGS=!SYNC_ARGS! !REL!"
exit /b 0
