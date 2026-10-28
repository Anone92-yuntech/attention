@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
if not exist ".venv\Scripts\python.exe" (
    echo Missing attention\.venv\Scripts\python.exe. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u video-save.py %*
set "attention_exit_code=%ERRORLEVEL%"
if not "%attention_exit_code%"=="0" pause
exit /b %attention_exit_code%
