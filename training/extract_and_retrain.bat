@echo off
cd /d E:\老人跌倒
call conda activate fall
echo ===== 提取三个摔倒视频关键点 =====
echo.
python training\extract_fall_videos.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===== 提取完成! 开始训练六分类模型 =====
    echo.
    python training\train_6class_v2.py
    echo.
    echo ===== All done! =====
) else (
    echo.
    echo [FAIL] 提取失败，请查看上面的错误信息
)
pause
