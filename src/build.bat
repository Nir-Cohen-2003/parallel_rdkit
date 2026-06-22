@echo off
setlocal enabledelayedexpansion

rem CMake 4.x dropped the VS 2017 generator; use Ninja instead (conda-forge standard for scikit-build-core)
set "CMAKE_GENERATOR=Ninja"

python -m build --wheel --outdir "%SRC_DIR%\dist\wheels" --no-isolation --skip-dependency-check
if %ERRORLEVEL% neq 0 exit 1

for %%f in ("%SRC_DIR%\dist\wheels\*.whl") do (
    python -m pip install "%%f" --no-deps --no-build-isolation --prefix "%PREFIX%" -vv
    if !ERRORLEVEL! neq 0 exit 1
)
