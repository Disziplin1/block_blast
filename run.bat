@echo off
REM Block Blast AI Assistant 실행
cd /d "%~dp0"

if not exist .venv (
    echo 가상환경이 없습니다. 먼저 setup.bat 을 실행해주세요.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py

if errorlevel 1 (
    echo.
    echo 오류가 발생했습니다. 위 메시지를 확인해주세요.
    pause
)
