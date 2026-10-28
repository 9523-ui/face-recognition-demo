@echo off
cd /d "%~dp0"
setlocal

rem ============================================================
rem  自动挑选「真正装了 opencv」的 Python
rem  本机情况：PATH 里的 D:\python (3.14) 没有 cv2，
rem            装了 cv2 的是 WorkBuddy 自带的 3.13.12
rem ============================================================
set "PY="

python -c "import cv2" >nul 2>&1
if not errorlevel 1 goto :got_sys

set "CAND=%USERPROFILE%\miniconda3\python.exe"
if not exist "%CAND%" goto :no_python
"%CAND%" -c "import cv2" >nul 2>&1
if errorlevel 1 goto :no_python
set "PY=%CAND%"
goto :run

:got_sys
set "PY=python"

:run
echo 解释器: %PY%
python -c "import cv2" >nul 2>&1
if errorlevel 1 "%PY%" -c "import cv2, numpy; print('OpenCV', cv2.__version__)" 2>nul
echo.

rem ---- 模型检查 ----
if not exist "models\yunet.onnx" goto :dl
if not exist "models\sface.onnx" goto :dl
goto :go

:dl
echo 首次运行，正在下载模型...
"%PY%" download_models.py
if errorlevel 1 (
    echo.
    echo 模型下载失败，请检查网络后重试。
    pause
    exit /b 1
)

:go
echo   直接运行   = 实时识别（空格截图 / q 退出）
echo   录入人脸   = run.bat --enroll 张三
echo.
"%PY%" face_demo.py %*

if errorlevel 1 (
    echo.
    echo [程序异常退出] 上面是错误信息，按任意键关闭窗口
    pause
)
endlocal
exit /b 0

:no_python
echo [错误] 没找到能 import cv2 的 Python 解释器。
echo.
echo   当前 PATH 里的 python 缺少 opencv。任选一种修法：
echo     1) 在 cmd 里执行: pip install opencv-python numpy
echo     2) 编辑本文件，把 CAND 改成你装了 opencv 的 python.exe 路径
echo.
pause
exit /b 1
