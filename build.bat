@echo off
setlocal enabledelayedexpansion

python -m pip install nanobind --quiet
python -m build --wheel --outdir "%SRC_DIR%\dist\wheels" --no-isolation --skip-dependency-check
for %%f in ("%SRC_DIR%\dist\wheels\*.whl") do (
    python -m pip install "%%f" --no-deps --no-build-isolation --prefix "%PREFIX%" -vv
)
