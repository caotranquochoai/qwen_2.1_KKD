#!/usr/bin/env bash
# Linux x86_64, Bash. CUDA wheels match the Windows installer.
set -Eeuo pipefail
trap 'printf "Setup failed at line %s. Installation stopped.\n" "$LINENO" >&2' ERR

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_command='python3.12'
launch='none'
install_prerequisites=true
apt_updated=false
constraint_file=''

usage() {
    cat <<'HELP'
Usage: bash install-linux.sh [--launch ui|api|none] [--python /path/to/python3.12]
                             [--skip-prerequisites]
Creates/reuses env, installs CUDA PyTorch and project/UI/API dependencies.
Missing prerequisites are installed through apt on Ubuntu/Debian when available.
HELP
}
fail() { printf '%s\n' "$*" >&2; exit 1; }
cleanup() {
    if [[ -n "$constraint_file" && -f "$constraint_file" ]]; then
        rm -- "$constraint_file"
    fi
}
trap cleanup EXIT

while (( $# > 0 )); do
    case "$1" in
        --launch)
            (( $# >= 2 )) || fail '--launch requires ui, api or none.'
            launch="$2"; shift 2 ;;
        --python)
            (( $# >= 2 )) || fail '--python requires an executable path or command.'
            python_command="$2"; shift 2 ;;
        --skip-prerequisites) install_prerequisites=false; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage; fail "Unknown option: $1" ;;
    esac
done
case "$launch" in ui|api|none) ;; *) fail '--launch must be ui, api or none.' ;; esac
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This installer requires Linux x86_64.'
for required_file in app.py api_server.py requirements.txt requirements-api.txt; do
    [[ -f "$project_dir/$required_file" ]] || fail "Missing $required_file. Download/clone the complete repository first."
done

install_apt_packages() {
    [[ "$install_prerequisites" == true ]] || fail "Missing prerequisites: $*. Install manually and rerun."
    command -v apt-get >/dev/null 2>&1 || fail "Install Python 3.12, its venv package and Git manually on this distribution, then rerun with --python."
    local -a elevation=()
    if (( EUID != 0 )); then
        command -v sudo >/dev/null 2>&1 || fail 'Installing prerequisites requires root or sudo. Install manually or ask the host administrator.'
        elevation=(sudo)
    fi
    if [[ "$apt_updated" == false ]]; then
        "${elevation[@]}" apt-get update
        apt_updated=true
    fi
    "${elevation[@]}" apt-get install -y "$@"
}

if ! command -v git >/dev/null 2>&1; then
    install_apt_packages git ca-certificates
fi
venv_dir="$project_dir/env"
venv_python="$venv_dir/bin/python"
if [[ ! -x "$venv_python" ]]; then
    [[ ! -e "$venv_dir" ]] || fail 'env exists without a usable Linux Python. Rename it and rerun; no environment files were deleted.'
    if ! command -v "$python_command" >/dev/null 2>&1; then
        [[ "$python_command" == python3.12 ]] || fail "Python executable not found: $python_command"
        install_apt_packages python3.12 python3.12-venv ca-certificates
    fi
    "$python_command" - <<'PY'
import struct, sys
assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8, 'Python 3.12 x64 is required'
PY
    # Ubuntu 24.04 provides these packages in its standard repositories.
    if ! "$python_command" -c 'import venv, ensurepip' >/dev/null 2>&1; then
        if [[ "$python_command" == python3.12 ]]; then
            install_apt_packages python3.12-venv
        else
            fail 'Selected Python lacks venv/ensurepip. Install those modules for this interpreter and rerun.'
        fi
    fi
    printf 'Creating env...\n'
    "$python_command" -m venv "$venv_dir"
fi
"$venv_python" - <<'PY'
import struct, sys
assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8, 'Existing env must use Python 3.12 x64; rename it to preserve it and rerun'
PY
"$venv_python" -m pip install --upgrade pip setuptools wheel
printf 'Installing PyTorch 2.14.0 CUDA 13.2 (large download)...\n'
"$venv_python" -m pip install --upgrade 'torch==2.14.0+cu132' torchvision \
    --index-url https://download.pytorch.org/whl/cu132

# Preserve the installed CUDA builds while resolving the other requirements.
constraint_file="$(mktemp)"
"$venv_python" - "$constraint_file" <<'PY'
import importlib.metadata as m
import pathlib, sys
pathlib.Path(sys.argv[1]).write_text(
    '\n'.join(n + '==' + m.version(n) for n in ('torch', 'torchvision')) + '\n', encoding='utf-8')
PY
"$venv_python" -m pip install -c "$constraint_file" \
    -r "$project_dir/requirements.txt" -r "$project_dir/requirements-api.txt"
# Installation check only; does not load model weights or generate an image.
"$venv_python" - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: install/update the NVIDIA driver; containers also need GPU passthrough'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM: %.2f GiB' % (torch.cuda.get_device_properties(0).total_memory / 2**30))
PY
printf 'Installation completed. Model weights download on first load.\n'
printf 'UI:  %q %q\n' "$venv_python" "$project_dir/app.py"
printf 'API: %q %q\n' "$venv_python" "$project_dir/api_server.py"
if [[ "$launch" != none ]]; then
    cd -- "$project_dir"
    if [[ "$launch" == ui ]]; then
        "$venv_python" "$project_dir/app.py"
    else
        "$venv_python" "$project_dir/api_server.py"
    fi
fi
