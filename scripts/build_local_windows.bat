@echo off
cls
echo Building CUDA code locally...

:: Get the GPU name using nvidia-smi
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do set gpu_name=%%i

:: Check the GPU model and set the architecture
if "%gpu_name%"=="RTX 3050 Ti" (
    set ARCH=sm_86
) else (
    echo Warning: GPU "%gpu_name%" not recognized.
    echo Please add your GPU and its architecture to the script.
    echo Falling back to default ARCH=sm_86
    set ARCH=sm_86
)

echo Detected GPU: %gpu_name%
echo Using ARCH=%ARCH%

:: Navigate to the CUDA directory and build the project
cd cuda || exit /b 1
call make clean
call make ARCH=%ARCH%
cd ..\scripts || exit /b 1

echo Build complete.
pause
