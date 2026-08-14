@echo off
setlocal
set PYTHON=D:\Anaconda3\envs\fall\python.exe
set SCRIPT=E:\老人跌倒\training\extract_one_subject.py
echo ===== START %date% %time% =====
for /L %%s in (1,1,9) do (
    echo.
    echo ===== Subject.%%s =====
    %PYTHON% -u "%SCRIPT%" %%s
    echo Exit code: %ERRORLEVEL%
)
echo ===== DONE %date% %time% =====
