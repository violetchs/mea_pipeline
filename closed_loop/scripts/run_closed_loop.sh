#!/usr/bin/env sh
set -e
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR/.."
cmake -S closed_loop/cpp -B closed_loop/cpp/build
cmake --build closed_loop/cpp/build
python -m closed_loop.closed_loop_setup --config-dir closed_loop/config --runner closed_loop/cpp/build/closed_loop_runner --record "$@"
