@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 使用 conda fall 环境（xgboost 3.2 + opencv 5 + mediapipe 0.10）
set PYTHON=d:\Anaconda3\envs\fall\python.exe
if not exist "%PYTHON%" (
    echo [错误] conda 环境 fall 未找到: %PYTHON%
    echo 请先创建环境: conda create -n fall python=3.10
    pause
    exit /b 1
)

echo ============================================================
echo   AI 智能居家监护系统 — 实时演示启动器
echo   环境: conda fall (xgboost 3.2 + opencv 5)
echo ============================================================
echo.
echo   选择启动模式:
echo.
echo   [1] 完整演示 (摄像头 + 步态推送 + Mock后端)
echo       — 适合独立演示，无需真实后端
echo.
echo   [2] 纯摄像头演示 (无后端推送)
echo       — 最简模式，只看摄像头推理
echo.
echo   [3] 连接真实后端 (需要先启动 FastAPI)
echo       — 配合田靖宇的后端服务
echo.
echo   [4] 只启动 Mock API 服务器
echo       — 用于前端/后端联调测试
echo.
echo   [5] 视频文件推理 (输入视频路径)
echo       — 离线分析已有视频
echo.
set /p mode="请输入编号 [1-5]: "

if "%mode%"=="1" goto demo_full
if "%mode%"=="2" goto demo_camera
if "%mode%"=="3" goto demo_backend
if "%mode%"=="4" goto mock_only
if "%mode%"=="5" goto video_mode

echo 无效选择，退出。
pause
exit /b

:demo_full
echo.
echo 🚀 启动完整演示模式...
echo    - Mock API 服务器 (localhost:8000)
echo    - 摄像头实时推理
echo    - 步态数据推送
echo.
echo   按 'q' 退出 | 按 's' 截图
echo ============================================================
%PYTHON% cv_client.py --mock-server --mock-port 8000
goto end

:demo_camera
echo.
echo 🚀 启动纯摄像头演示模式...
echo    - 仅本地推理，不推送后端
echo    - 按 'q' 退出 | 按 's' 截图
echo ============================================================
%PYTHON% cv_client.py --no-push
goto end

:demo_backend
echo.
echo 🚀 连接真实后端...
set /p backend_url="后端地址 (默认 http://127.0.0.1:8000): "
if "%backend_url%"=="" set backend_url=http://127.0.0.1:8000
echo    - 后端: %backend_url%
echo    - 按 'q' 退出 | 按 's' 截图
echo ============================================================
%PYTHON% cv_client.py --api-url %backend_url%
goto end

:mock_only
echo.
echo 🚀 启动 Mock API 服务器 (localhost:8000)...
echo    POST http://localhost:8000/api/v1/gait-data/
echo    按 Ctrl+C 停止
echo ============================================================
%PYTHON% -c "from cv_client import run_mock_server; run_mock_server(8000)"
goto end

:video_mode
echo.
set /p video_path="视频文件路径: "
echo 🚀 视频推理模式: %video_path%
echo ============================================================
%PYTHON% fall_inference.py "%video_path%" --save-video
goto end

:end
echo.
echo 演示结束。
pause
