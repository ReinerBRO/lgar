@echo off
cd /d "%~dp0"
title Auto rsync lgar to dev
echo Auto rsync lgar to dev:/gemini/space/private/zjc/goals/lgar/
echo Press Ctrl+C to stop. Default interval: 5s.
echo.
call "%~dp0rsync-once-to-dev.cmd" watch %1
