@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Movie Agent - Web UI
echo   Vue 3 frontend + Python standard-library HTTP API
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found: .venv
    echo.
    echo Please run the following first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "config\config.yaml" (
    echo [SETUP] config\config.yaml not found, copying from template...
    copy "config\config.example.yaml" "config\config.yaml" >nul
    echo.
    echo [NEXT] Open config\config.yaml and fill in your DashScope API Key
    echo        Apply at: https://dashscope.console.aliyun.com/
    echo        Then run this script again.
    echo.
    pause
    exit /b 1
)

echo Server URL : http://127.0.0.1:8765
echo Note       : the page waits until model and knowledge base are ready
echo Stop       : close this window
echo.

rem 本地向量模型的缓存锁定在项目内的 models_cache\，并默认走国内镜像
rem （src\utils.py 里也有同样设置，这里再设一次是为了让外部工具/调试也一致）
if not exist "models_cache" mkdir "models_cache"
set "HF_HOME=%CD%\models_cache"
set "HF_HUB_CACHE=%CD%\models_cache"
set "SENTENCE_TRANSFORMERS_HOME=%CD%\models_cache\sentence_transformers"
set "TORCH_HOME=%CD%\models_cache\torch"
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"

rem open the browser after the server has started listening
start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8765"

".venv\Scripts\python.exe" "server\app.py"

echo.
echo [server] stopped
pause
