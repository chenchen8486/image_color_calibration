@echo off
chcp 65001 >nul
title Camera Color Calibration - Build Tool
echo ========================================
echo   Camera Color Calibration - Build Tool
echo ========================================
echo.

echo [1/4] Checking PyInstaller...
pyinstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyInstaller not found. Please install it first:
    echo   pip install pyinstaller
    echo.
    pause
    exit /b 1
)
echo [1/4] PyInstaller check passed.
echo.

echo [2/4] Building executable (directory mode -D)...
echo         This may take 3~10 minutes, please wait...
pyinstaller ColorCalibrationTool.spec
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed. Please check the logs above.
    pause
    exit /b 1
)
echo [2/4] Build completed.
echo.

echo [3/4] Preparing required folders and files...
if not exist "dist\ColorCalibrationTool\1_template" mkdir "dist\ColorCalibrationTool\1_template"
if not exist "dist\ColorCalibrationTool\2_capture" mkdir "dist\ColorCalibrationTool\2_capture"
if not exist "dist\ColorCalibrationTool\3_result\map_data" mkdir "dist\ColorCalibrationTool\3_result\map_data"
if not exist "dist\ColorCalibrationTool\4_test_data\input" mkdir "dist\ColorCalibrationTool\4_test_data\input"
if not exist "dist\ColorCalibrationTool\4_test_data\output" mkdir "dist\ColorCalibrationTool\4_test_data\output"
if not exist "dist\ColorCalibrationTool\logs" mkdir "dist\ColorCalibrationTool\logs"

if exist "1_template" (
    xcopy /E /I /Y "1_template\*" "dist\ColorCalibrationTool\1_template\" >nul
    echo         - Copied 1_template/
)
if exist "2_capture" (
    xcopy /E /I /Y "2_capture\*" "dist\ColorCalibrationTool\2_capture\" >nul
    echo         - Copied 2_capture/
)
if exist "config.ini" (
    copy /Y "config.ini" "dist\ColorCalibrationTool\config.ini" >nul
    echo         - Copied config.ini
) else (
    echo         - config.ini not found, it will be auto-generated on first run
)
copy /Y "readme.md" "dist\ColorCalibrationTool\readme.md" >nul
echo         - Copied readme.md
echo [3/4] Preparation completed.
echo.

echo [4/4] Verifying build result...
if exist "dist\ColorCalibrationTool\ColorCalibrationTool.exe" (
    echo ========================================
    echo   Build Success!
    echo ========================================
    echo.
    echo Output directory: dist\ColorCalibrationTool\
    echo.
    echo You can copy the entire "dist\ColorCalibrationTool\" folder
    echo to any Windows user. No Python installation required.
    echo.
) else (
    echo [WARNING] .exe file not found. Please check the build logs.
)
echo.
pause
