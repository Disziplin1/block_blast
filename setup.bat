@echo off
REM 최초 1회만 실행: 가상환경 생성 + 필요한 패키지 설치
cd /d "%~dp0"

if not exist .venv (
    echo [1/2] 가상환경(.venv) 생성 중...
    python -m venv .venv
)

echo [2/2] 패키지 설치 중... (시간이 걸릴 수 있습니다)
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo 설치가 완료되었습니다. 이제 run.bat 을 더블클릭하여 실행하세요.
pause
