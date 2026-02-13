@echo off
chcp 65001 >nul
echo ========================================
echo    考试监考系统 启动中...
echo ========================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [提示] 未检测到虚拟环境，使用系统Python
)

echo.
echo [提示] 按 Q 键退出系统
echo.

python main.py --display

echo.
echo ========================================
echo    系统已退出
echo ========================================
pause
