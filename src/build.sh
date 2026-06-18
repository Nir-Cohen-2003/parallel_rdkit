#!/bin/bash
set -ex

python -m build --wheel --outdir "$SRC_DIR/dist/wheels" --no-isolation --skip-dependency-check

# pixi-build runs this script in a temporary source directory, so $SRC_DIR/../
# is *not* the project workspace. Walk up until we find pixi.toml so the wheel
# lands in a persistent location that CI can upload to PyPI.
find_project_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/pixi.toml" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo "WARNING: could not find project root (no pixi.toml); skipping wheel export" >&2
    return 1
}

if PROJECT_ROOT=$(find_project_root "$SRC_DIR"); then
    mkdir -p "$PROJECT_ROOT/dist"

    # Optional: repair the Linux wheel to a manylinux tag so PyPI accepts it.
    # Add 'auditwheel' to build requirements on linux if you want this step.
    if [ "$(uname -s)" = "Linux" ] && command -v auditwheel >/dev/null 2>&1; then
        auditwheel repair "$SRC_DIR/dist/wheels/"*.whl --wheel-dir "$PROJECT_ROOT/dist/"
    else
        cp "$SRC_DIR/dist/wheels/"*.whl "$PROJECT_ROOT/dist/"
    fi
fi

python -m pip install "$SRC_DIR/dist/wheels/"*.whl --no-deps --no-build-isolation --prefix "$PREFIX" -vv
