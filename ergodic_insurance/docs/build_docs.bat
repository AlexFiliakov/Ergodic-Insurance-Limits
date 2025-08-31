@echo off
REM Build script for Sphinx documentation

echo Building Sphinx documentation...
cd /d "%~dp0"

REM Clean build to avoid the ....api issue
if exist "..\..\api" (
    echo Cleaning existing API documentation...
    rmdir /s /q "..\..\api" 2>nul
)

REM Build the documentation with correct output path
echo Running Sphinx build...
python -m sphinx -b html . ..\..\api

echo.
echo Documentation build complete!
echo Output directory: ..\..\api
echo View at: https://alexfiliakov.github.io/Ergodic-Insurance-Limits/api/
