@echo off
cd /d "%~dp0"
setlocal

rem ============================================================
rem  图形界面版启动器（本文件必须是 GBK/ANSI 编码，否则 cmd 解析中文会错行）
rem  要求解释器同时具备 opencv 与 tkinter
rem  常见位置：D:\miniconda\python.exe、%USERPROFILE%\miniconda3\python.exe
rem        D:\python (3.14) 缺 cv2
rem        WorkBuddy 自带 3.13.12 缺 tkinter
rem ============================================================
set "PY="

python -c "import cv2,tkinter" >nul 2>&1
if not errorlevel 1 goto :use_sys

set "CAND=D:\miniconda\python.exe"
if not exist "%CAND%" goto :try_wb
"%CAND%" -c "import cv2,tkinter" >nul 2>&1
if errorlevel 1 goto :try_wb
set "PY=%CAND%"
goto :run

:try_wb
set "CAND=%USERPROFILE%\miniconda3\python.exe"
if not exist "%CAND%" goto :no_py
"%CAND%" -c "import cv2,tkinter" >nul 2>&1
if errorlevel 1 goto :no_py
set "PY=%CAND%"
goto :run

:use_sys
set "PY=python"

:run
if /i "%~1"=="--check" goto :check

if not exist "models\yunet.onnx" goto :dl
if not exist "models\sface.onnx" goto :dl
goto :go

:dl
echo 首次运行，正在下载模型...
"%PY%" download_models.py
if errorlevel 1 goto :dl_fail

:go
"%PY%" gui.py %*
if errorlevel 1 goto :crashed
endlocal
exit /b 0

:dl_fail
echo.
echo 模型下载失败，请检查网络后重试。
pause
exit /b 1

:crashed
echo.
echo 界面异常退出，上面是错误信息。按任意键关闭窗口。
pause
endlocal
exit /b 1

:no_py
echo [错误] 没找到同时装了 opencv 和 tkinter 的 Python。
echo.
echo   常见安装位置: D:\miniconda\python.exe
echo   若不存在，请先安装依赖: pip install opencv-python pillow numpy
echo.
pause
exit /b 1

:check
echo 解释器: %PY%
"%PY%" -c "import sys,cv2,numpy,tkinter;print('python  ',sys.version.split()[0]);print('opencv  ',cv2.__version__);print('numpy   ',numpy.__version__);print('tkinter ',tkinter.TkVersion)"
if exist "models\yunet.onnx" if exist "models\sface.onnx" echo 模型: 两个 onnx 文件都已就位
if not exist "models\yunet.onnx" echo 模型: 缺少 yunet.onnx
if not exist "models\sface.onnx" echo 模型: 缺少 sface.onnx
exit /b 0
