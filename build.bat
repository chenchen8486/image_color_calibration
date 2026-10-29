@echo off
chcp 65001 >nul
title Camera Color Calibration - Build Tool
echo ========================================
echo   Camera Color Calibration - Build Tool
echo ========================================
echo.

echo [1/4] Checking PyInstaller...
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [ERROR] venv not found. Please create the virtual environment first.
    echo.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m PyInstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyInstaller not found in venv. Please install it first:
    echo   venv\Scripts\pip.exe install pyinstaller
    echo.
    pause
    exit /b 1
)
echo [1/4] PyInstaller check passed.
echo.

echo [2/4] Building executable (directory mode -D)...
echo         This may take 3~10 minutes, please wait...
venv\Scripts\python.exe -m PyInstaller ColorCalibrationTool.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed. Please check the logs above.
    pause
    exit /b 1
)
echo [2/4] Build completed.
echo.

echo [3/4] Preparing release folder...
set DIST_NAME=ColorCalibrationTool
if not exist "dist\%DIST_NAME%\" (
    echo.
    echo [ERROR] dist\%DIST_NAME% not found. Build may have failed.
    pause
    exit /b 1
)
if exist "release\%DIST_NAME%" rmdir /s /q "release\%DIST_NAME%"
if not exist "release" mkdir "release"
move "dist\%DIST_NAME%" "release\%DIST_NAME%" >nul
rmdir /s /q "dist" "build"

if not exist "release\%DIST_NAME%\data\1_template" mkdir "release\%DIST_NAME%\data\1_template"
if not exist "release\%DIST_NAME%\data\2_capture" mkdir "release\%DIST_NAME%\data\2_capture"
if not exist "release\%DIST_NAME%\data\3_result\map_data" mkdir "release\%DIST_NAME%\data\3_result\map_data"
if not exist "release\%DIST_NAME%\data\4_test_data\input" mkdir "release\%DIST_NAME%\data\4_test_data\input"
if not exist "release\%DIST_NAME%\data\4_test_data\output" mkdir "release\%DIST_NAME%\data\4_test_data\output"
if not exist "release\%DIST_NAME%\logs" mkdir "release\%DIST_NAME%\logs"

if exist "data\1_template" (
    xcopy /E /I /Y "data\1_template\*" "release\%DIST_NAME%\data\1_template\" >nul
    echo         - Copied data/1_template/
)
echo         - Created empty data/2_capture/ (user will place match.bmp here)
if exist "config.ini" (
    copy /Y "config.ini" "release\%DIST_NAME%\config.ini" >nul
    echo         - Copied config.ini
) else (
    echo         - config.ini not found, it will be auto-generated on first run
)
copy /Y "README.md" "release\%DIST_NAME%\README.md" >nul
echo         - Copied README.md

echo 相机色彩矫正标定系统 - 使用说明 > "release\%DIST_NAME%\使用说明.txt"
echo ========================================= >> "release\%DIST_NAME%\使用说明.txt"
echo. >> "release\%DIST_NAME%\使用说明.txt"
echo 使用前准备 >> "release\%DIST_NAME%\使用说明.txt"
echo 1. 确认 data\1_template\ 中的模板图和 map.xlsx 齐全（打包时已自动拷贝）。 >> "release\%DIST_NAME%\使用说明.txt"
echo 2. 将相机实拍色卡图命名为 match.bmp 后放入： >> "release\%DIST_NAME%\使用说明.txt"
echo    data\2_capture\ >> "release\%DIST_NAME%\使用说明.txt"
echo. >> "release\%DIST_NAME%\使用说明.txt"
echo 运行程序 >> "release\%DIST_NAME%\使用说明.txt"
echo 直接双击 ColorCalibrationTool.exe，按控制台菜单选择功能： >> "release\%DIST_NAME%\使用说明.txt"
echo   功能1：训练模型并生成标定文件（LUT） >> "release\%DIST_NAME%\使用说明.txt"
echo   功能2：批量推理测试图片 >> "release\%DIST_NAME%\使用说明.txt"
echo   功能3：重新加载配置文件 >> "release\%DIST_NAME%\使用说明.txt"
echo. >> "release\%DIST_NAME%\使用说明.txt"
echo 输出位置 >> "release\%DIST_NAME%\使用说明.txt"
echo 1. 标定文件和训练结果保存在 data\3_result\ 目录下。 >> "release\%DIST_NAME%\使用说明.txt"
echo 2. 功能2 的待测图片放入 data\4_test_data\input\， >> "release\%DIST_NAME%\使用说明.txt"
echo    矫正结果输出到 data\4_test_data\output\。 >> "release\%DIST_NAME%\使用说明.txt"
echo ========================================= >> "release\%DIST_NAME%\使用说明.txt"
echo         - Created 使用说明.txt
echo [3/4] Preparation completed.
echo.

echo [4/4] Verifying build result...
if exist "release\%DIST_NAME%\ColorCalibrationTool.exe" (
    echo ========================================
    echo   Build Success!
    echo ========================================
    echo.
    echo Output directory: release\%DIST_NAME%\
    echo.
    echo You can copy the entire "release\%DIST_NAME%\" folder
    echo to any Windows user. No Python installation required.
    echo.
) else (
    echo [WARNING] .exe file not found. Please check the build logs.
)
echo.
pause
