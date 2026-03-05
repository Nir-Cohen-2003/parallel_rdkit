#!/bin/bash
set -ex

python -m build --wheel --outdir "$SRC_DIR/dist/wheels" --no-isolation --skip-dependency-check
mkdir -p "$RECIPE_DIR/dist"
cp "$SRC_DIR/dist/wheels/"*.whl "$RECIPE_DIR/dist/"
python -m pip install "$SRC_DIR/dist/wheels/"*.whl --no-deps --no-build-isolation -vv
