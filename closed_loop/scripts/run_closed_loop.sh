#!/usr/bin/env sh
set -e
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR/.."

if [ -z "${CXX:-}" ]; then
    for candidate in g++-13; do
        if command -v "$candidate" >/dev/null 2>&1; then
            export CXX="$(command -v "$candidate")"
            cc_candidate="$(printf '%s\n' "$candidate" | sed 's/g++/gcc/')"
            if command -v "$cc_candidate" >/dev/null 2>&1; then
                export CC="$(command -v "$cc_candidate")"
            fi
            echo "Using C++ compiler: $CXX"
            break
        fi
    done
fi
if [ -z "${CXX:-}" ]; then
    echo "ERROR: g++-13 or newer is required to link MaxLab libmaxlab.a." >&2
    echo "Install it or set CXX/CC to a compatible compiler before running this script." >&2
    exit 1
fi

rm -f closed_loop/cpp/build/CMakeCache.txt
rm -rf closed_loop/cpp/build/CMakeFiles
cmake -S closed_loop/cpp -B closed_loop/cpp/build
cmake --build closed_loop/cpp/build
python -m closed_loop.closed_loop_setup --config-dir closed_loop/config --runner closed_loop/cpp/build/closed_loop_runner --record "$@"
