#!/usr/bin/env bash
# Run the Kokoro TTS client kokoro_client.py with the project venv.
#
# Drop-in replacement for kokoro-hello's kokoro.sh. Resolves symlinks, so it
# can be linked anywhere (e.g. into /usr/local/bin as `kokoro`).
set -euo pipefail

script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
script_dir="$(dirname -- "$script_path")"

python_bin="${script_dir}/.venv/bin/python"
client_py="${script_dir}/kokoro_client.py"

if [[ ! -x "$python_bin" ]]; then
    printf 'kokoro: venv python not found at %s\nrun "uv sync" first\n' "$python_bin" >&2
    exit 1
fi

exec "$python_bin" "$client_py" "$@"
