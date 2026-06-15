@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run this from Tianyi Windows inside the mounted Mac project directory.
rem No args: copy the project tree. With args: copy only those relative files.

set "MODE=%~1"
set "DST=D:\lgar"
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "SRC_PREFIX=%SRC%\"
set "EXTRA="
set "LOG_FLAGS="
set "QUIET="
if /i "%MODE%"=="dry" set "EXTRA=/L"
if /i "%MODE%"=="quiet" (
  set "QUIET=1"
  set "LOG_FLAGS=/NDL /NJH /NJS /NC /NS"
)

if not "%~1"=="" (
  if /i not "%~1"=="dry" (
    if /i not "%~1"=="quiet" goto only_files
  )
)

if not defined QUIET echo robocopy %SRC% -^> %DST%
if /i "%MODE%"=="dry" echo dry run: no files will be copied
if not defined QUIET echo.
robocopy "%SRC%" "%DST%" /E /FFT /Z /MT:8 /R:2 /W:2 /NP /XJ /COPY:DAT /DCOPY:DAT %LOG_FLAGS% %EXTRA% ^
  /XD ^
  ".git" ^
  "__pycache__" ^
  ".pytest_cache" ^
  ".mypy_cache" ^
  ".ruff_cache" ^
  ".venv" ^
  "venv" ^
  "env" ^
  "build" ^
  "dist" ^
  "data" ^
  "logs" ^
  "outputs" ^
  "output" ^
  "runs" ^
  "tmp" ^
  ".cache" ^
  "wandb" ^
  /XF ^
  "*.pyc" ^
  ".DS_Store"
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo robocopy failed: exit=%RC%
  if defined QUIET echo rerun robocopy-mounted-to-d.cmd without quiet for details
  exit /b %RC%
)
if not defined QUIET echo robocopy ok: exit=%RC%
exit /b %RC%

:only_files
echo selected copy %SRC% -^> %DST%
set "ANY_FAIL=0"

:only_loop
if "%~1"=="" goto only_done
set "REL=%~1"
set "REL=!REL:/=\!"

if "!REL!"=="" (
  shift
  goto only_loop
)
if "!REL:~1,1!"==":" (
  echo skip absolute path: !REL!
  set "ANY_FAIL=1"
  shift
  goto only_loop
)
if "!REL:~0,1!"=="\" (
  echo skip absolute path: !REL!
  set "ANY_FAIL=1"
  shift
  goto only_loop
)
if not "!REL:..=!"=="!REL!" (
  echo skip parent path: !REL!
  set "ANY_FAIL=1"
  shift
  goto only_loop
)
if not exist "!SRC!\!REL!" (
  echo missing source: !REL!
  set "ANY_FAIL=1"
  shift
  goto only_loop
)

for %%I in ("!SRC!\!REL!") do (
  set "SRC_DIR=%%~dpI"
  set "NAME=%%~nxI"
)
if "!SRC_DIR:~-1!"=="\" set "SRC_DIR=!SRC_DIR:~0,-1!"
set "RELDIR=!SRC_DIR:%SRC_PREFIX%=!"
if "!RELDIR!"=="!SRC_DIR!" set "RELDIR="
if "!RELDIR!"=="" (
  set "DST_DIR=!DST!"
) else (
  set "DST_DIR=!DST!\!RELDIR!"
)

robocopy "!SRC_DIR!" "!DST_DIR!" "!NAME!" /FFT /Z /R:2 /W:2 /NP /XJ /COPY:DAT /DCOPY:DAT /NJH /NJS /NDL /NC /NS /NFL >nul
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 (
  echo failed: !REL! exit=!RC!
  set "ANY_FAIL=1"
) else if "!RC!"=="0" (
  echo up-to-date: !REL!
) else (
  echo copied/updated: !REL! exit=!RC!
)
shift
goto only_loop

:only_done
if "%ANY_FAIL%"=="0" (
  echo selected copy ok
  exit /b 0
)
echo selected copy had errors
exit /b 1
