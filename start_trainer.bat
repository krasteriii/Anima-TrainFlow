@echo off
setlocal
cd /d %~dp0

where uv >nul 2>nul
if not errorlevel 1 (
    echo Starting Anima TrainFlow ^(uv^)...
    echo.
    rem uv run installs/updates .venv from uv.lock first when needed
    uv run app.py
    goto :done
)

rem Fallback: the portable build ships its own interpreter with the packages
if exist "%~dp0python_embeded\python.exe" (
    echo [!] uv not found - using the portable Python in python_embeded.
    echo.
    "%~dp0python_embeded\python.exe" app.py
    goto :done
)

echo [ERROR] uv not found. Install it with:
echo.
echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
echo.
echo then run Install_Requirements.bat.
pause
exit /b 1

:done
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script crashed. Check the error message above.
    pause
)
