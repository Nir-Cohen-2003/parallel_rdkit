@echo off
setlocal enabledelayedexpansion

set "CMAKE_GENERATOR=Visual Studio 17 2022"

python -m build --wheel --outdir "%SRC_DIR%\dist\wheels" --no-isolation --skip-dependency-check
if %ERRORLEVEL% neq 0 exit 1

rem pixi-build runs this script in a temporary source directory, so %SRC_DIR%\..
rem is *not* the project workspace. Walk up until we find pixi.toml so the wheel
rem lands in a persistent location that CI can upload to PyPI.
set "PROJECT_ROOT=%SRC_DIR%"
:findroot
if exist "%PROJECT_ROOT%\pixi.toml" goto :foundroot
for %%I in ("%PROJECT_ROOT%\..") do set "PARENT=%%~fI"
if "%PARENT%"=="%PROJECT_ROOT%" goto :noroot
set "PROJECT_ROOT=%PARENT%"
goto :findroot

:noroot
echo WARNING: could not find project root ^(no pixi.toml^); skipping wheel export
:install
for %%f in ("%SRC_DIR%\dist\wheels\*.whl") do (
    python -m pip install "%%f" --no-deps --no-build-isolation --prefix "%PREFIX%" -vv
    if !ERRORLEVEL! neq 0 exit 1
)
goto :eof

:foundroot
if not exist "%PROJECT_ROOT%\dist" mkdir "%PROJECT_ROOT%\dist"
copy "%SRC_DIR%\dist\wheels\*.whl" "%PROJECT_ROOT%\dist\"
if %ERRORLEVEL% neq 0 exit 1
goto :install
