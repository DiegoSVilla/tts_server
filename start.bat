@echo off
REM XTTS-v2 Streaming TTS Server Launcher
REM Uses auto-select GPU mode with FP16 optimization enabled

REM Check for test mode
if "%1"=="test" (
    echo.
    echo ========================================
    echo   XTTS-v2 TTS Server - Test Suite
    echo ========================================
    echo.
    echo Running all tests...
    echo.
    python main_test.py
    goto :end
)

echo.
echo ========================================
echo   XTTS-v2 Streaming TTS Server
echo ========================================
echo.
echo Starting server with optimizations:
echo   - Auto GPU selection (most free memory)
echo   - FP16 enabled (2x faster inference)
echo   - Speed multiplier: 1.3 (30%% faster speech)
echo.

uv run python main.py --auto-select-gpu --fp16 --speed 1.3

REM Keep window open if script fails
if errorlevel 1 (
    echo.
    echo Server stopped with errors. Press any key to exit...
    pause >nul
)

:end