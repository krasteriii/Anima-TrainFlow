@echo off
setlocal
cd /d %~dp0

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed. Install it with:
    echo.
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo then open a new terminal and run this file again.
    pause
    exit /b 1
)

echo Installing Python 3.10 + PyTorch ^(CUDA 12.8^) + sd-scripts dependencies into .venv ...
echo ^(versions pinned in uv.lock; already downloaded packages come from the uv cache^)
uv sync
if errorlevel 1 (
    echo.
    echo [ERROR] uv sync failed. Check the message above.
    pause
    exit /b 1
)

echo.
echo Installation Complete!
echo You can now run the trainer via start_trainer.bat
pause
