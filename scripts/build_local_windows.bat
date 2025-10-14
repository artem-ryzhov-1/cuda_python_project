
@echo off
setlocal enabledelayedexpansion

echo Building CUDA code on Windows...

REM Detect GPU and set architecture
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format^=csv^,noheader') do set GPU_NAME=%%g
echo Detected GPU: %GPU_NAME%

REM Map GPU to CUDA architecture
set ARCH=sm_86
if not defined GPU_NAME (
    echo Warning: Could not detect GPU. Using default ARCH=%ARCH%
) else (
    if /i "%GPU_NAME:RTX 3050 Ti=%" neq "%GPU_NAME%" set ARCH=sm_86
    if /i "%GPU_NAME:RTX 30=%" neq "%GPU_NAME%" set ARCH=sm_86
    if /i "%GPU_NAME:RTX 40=%" neq "%GPU_NAME%" set ARCH=sm_89
    if /i "%GPU_NAME:T4=%"   neq "%GPU_NAME%" set ARCH=sm_75
    if /i "%GPU_NAME:V100=%" neq "%GPU_NAME%" set ARCH=sm_70
    if /i "%GPU_NAME:A100=%" neq "%GPU_NAME%" set ARCH=sm_80
    if /i "%GPU_NAME:P100=%" neq "%GPU_NAME%" set ARCH=sm_60
)
echo Using CUDA architecture: %ARCH%

REM Find NVCC
set NVCC_PATH=
if defined CUDA_PATH (
    set NVCC_PATH=%CUDA_PATH%\bin\nvcc.exe
) else if defined CUDA_HOME (
    set NVCC_PATH=%CUDA_HOME%\bin\nvcc.exe
) else (
    REM Try common installation paths
    for %%v in (12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8) do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe" (
            set NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe
            goto :found_nvcc
        )
    )
)

:found_nvcc
if not exist "%NVCC_PATH%" (
    echo Error: Could not find nvcc.exe
    echo Please set CUDA_PATH or CUDA_HOME environment variable
    exit /b 1
)
echo Found NVCC at: %NVCC_PATH%

REM Navigate to cuda directory
cd /d "%~dp0\..\cuda"
if errorlevel 1 (
    echo Error: Could not navigate to cuda directory
    exit /b 1
)

REM Set up MSVC environment by calling vcvars64.bat (Visual Studio environment setup)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Add MSVC tools to PATH explicitly (optional, if vcvars64.bat doesn’t already do this)
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64;%PATH%

set OUTPUT = bin\lindblad_cuda.exe

REM Compile CUDA code
echo Compiling CUDA code...
"%NVCC_PATH%" -O3 -std=c++17 -arch=%ARCH% -Isrc src\main.cu -o %OUTPUT%

if errorlevel 1 (
    echo Error: Compilation failed
    exit /b 1
)

echo Build complete. Executable: %OUTPUT%
exit /b 0