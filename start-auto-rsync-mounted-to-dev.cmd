@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Run this from Tianyi Windows inside the mounted Mac project directory.
rem Sync once, then watch local file hashes. Optional args override watched files.

set "INTERVAL=%~1"
if "%INTERVAL%"=="" (
  set "INTERVAL=5"
) else (
  echo(%INTERVAL%| findstr /r "^[0-9][0-9]*$" >nul
  if errorlevel 1 (
    set "INTERVAL=5"
  ) else (
    shift
  )
)
set "DEFAULT_WATCH_FILES=curcpt/eval_long_context_public.py|scripts/zjc_eval_qwen25_3b_32k_public_multilen.sh|sync-m-to-dev-then-dev-to-d.cmd"
set "WATCH_EXCLUDE_DIRS=.git|__pycache__|.pytest_cache|.mypy_cache|.ruff_cache|.venv|venv|env|build|dist|data|logs|outputs|output|runs|tmp|.cache|wandb"
set "WATCH_EXCLUDE_FILES=*.pyc|.DS_Store"
set "WATCH_FILES="
set "SYNC_ARGS="
set "LAST_SIG="

if "%~1"=="" goto default_watch
goto watch_arg_loop

:default_watch
set "WATCH_FILES=%DEFAULT_WATCH_FILES%"
set "SYNC_ARGS="
goto watch_args_done

:watch_arg_loop
if "%~1"=="" goto watch_args_done
set "REL=%~1"
set "REL=!REL:\=/!"
if "!REL!"=="" (
  shift
  goto watch_arg_loop
)
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
if defined WATCH_FILES (
  set "WATCH_FILES=!WATCH_FILES!|!REL!"
) else (
  set "WATCH_FILES=!REL!"
)
set "SYNC_ARGS=!SYNC_ARGS! !REL!"
shift
goto watch_arg_loop

:watch_args_done

cd /d "%~dp0"

if not exist "%~dp0rsync-mounted-to-dev.cmd" (
  echo missing launcher: %~dp0rsync-mounted-to-dev.cmd
  exit /b 1
)

echo auto rsync loop started
echo project=%CD%
echo interval=%INTERVAL%s
echo watch=%WATCH_FILES%
echo press Ctrl+C to stop

echo.
echo [%DATE% %TIME%] initial rsync start
call "%~dp0rsync-mounted-to-dev.cmd" sync !SYNC_ARGS!
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [%DATE% %TIME%] initial rsync failed: exit=%RC%
  exit /b %RC%
)
call :compute_sig
set "LAST_SIG=%CUR_SIG%"
echo [%DATE% %TIME%] baseline recorded

:loop
echo.
call :compute_sig
if defined LAST_SIG if "!CUR_SIG!"=="!LAST_SIG!" (
  echo [%DATE% %TIME%] no local changes
  echo waiting %INTERVAL%s...
  timeout /t %INTERVAL% /nobreak >nul
  goto loop
)
echo [%DATE% %TIME%] rsync start
call "%~dp0rsync-mounted-to-dev.cmd" sync !SYNC_ARGS!
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [%DATE% %TIME%] rsync failed: exit=%RC%
) else (
  if defined CUR_SIG set "LAST_SIG=!CUR_SIG!"
  echo [%DATE% %TIME%] rsync ok
)
echo waiting %INTERVAL%s...
timeout /t %INTERVAL% /nobreak >nul
goto loop

:compute_sig
set "CUR_SIG="
for /f "usebackq delims=" %%S in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $files=$env:WATCH_FILES -split '\|' | Where-Object { $_ }; $excludeDirs=$env:WATCH_EXCLUDE_DIRS -split '\|' | Where-Object { $_ }; $excludeFiles=$env:WATCH_EXCLUDE_FILES -split '\|' | Where-Object { $_ }; function IsExcludedDir($p){$parts=$p -split '[\\/]'; foreach($d in $excludeDirs){if($parts -contains $d){return $true}} return $false}; function IsExcludedFile($p){$name=[IO.Path]::GetFileName($p); foreach($pat in $excludeFiles){$wc=[System.Management.Automation.WildcardPattern]::new($pat,[System.Management.Automation.WildcardOptions]::IgnoreCase); if($wc.IsMatch($name)){return $true}} return $false}; $items=foreach($f in $files){if(Test-Path -LiteralPath $f -PathType Leaf){if(-not (IsExcludedDir $f) -and -not (IsExcludedFile $f)){$h=(Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash; $f+':'+$h}}elseif(Test-Path -LiteralPath $f -PathType Container){Get-ChildItem -LiteralPath $f -Recurse -File | Where-Object {-not (IsExcludedDir $_.FullName) -and -not (IsExcludedFile $_.FullName)} | Sort-Object FullName | ForEach-Object {$_.FullName+':'+(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash}}else{'MISSING:'+$f}}; $joined=$items -join '|'; $sha=[Security.Cryptography.SHA256]::Create(); $bytes=[Text.Encoding]::UTF8.GetBytes($joined); [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-','')"`) do set "CUR_SIG=%%S"
exit /b 0
