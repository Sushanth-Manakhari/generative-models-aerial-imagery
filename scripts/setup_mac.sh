#!/usr/bin/env bash
# Laptop setup for the generative-models lecture, on macOS.
#
#   bash scripts/setup_mac.sh
#
# What it does, all inside this project folder:
#   1. finds Python 3.10 or newer (no version is hard-coded; the newest one found is used)
#   2. creates .venv here, or reuses it if it is already there and new enough
#   3. upgrades pip inside .venv
#   4. installs requirements.txt and the genroofs package (editable) into .venv
#   5. runs the self-check: python -m genroofs.check
#
# It never uses sudo, never installs anything globally, and changes nothing outside this folder
# (pip keeps its usual download cache in your user cache folder).
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
VENV="$PROJECT/.venv"

say()   { printf '\n==> %s\n' "$*"; }
fail()  { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }
tidy() {                                           # show paths with ~ instead of the home folder
    case "$1" in
        "$HOME"*) printf '~%s' "${1#"$HOME"}" ;;
        *) printf '%s' "$1" ;;
    esac
}

new_enough() {                                     # is this Python 3.10 or newer?
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

pick_python() {                                    # newest python3.N on PATH, then plain python3
    local candidates found
    candidates="$(compgen -c python3. 2>/dev/null | grep -E '^python3\.[0-9]+$' | sort -t. -k2 -n -r -u || true)"
    for found in $candidates python3; do
        if command -v "$found" >/dev/null 2>&1 && new_enough "$found"; then
            command -v "$found"
            return 0
        fi
    done
    return 1
}

say "Project folder: $(tidy "$PROJECT")"
if [ "$(uname -s)" != "Darwin" ]; then
    printf 'Note: this script is written for macOS; it may still work on Linux.\n'
fi
if [ "$(uname -m)" = "x86_64" ] && [ "$(uname -s)" = "Darwin" ]; then
    printf 'Note: this is an Intel Mac. PyTorch stopped publishing new builds for Intel Macs, so the install may\n'
    printf '      fail or pick an old version. If it does, use Google Colab for the lab instead.\n'
fi

if [ -x "$VENV/bin/python" ] && new_enough "$VENV/bin/python"; then
    say "Reusing $(tidy "$VENV") ($("$VENV/bin/python" --version 2>&1))"
else
    if [ -e "$VENV" ]; then
        fail "$(tidy "$VENV") exists but has no Python 3.10 or newer. Rename or delete that folder, then run this again."
    fi
    PY="$(pick_python)" || fail "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/macos/ (see docs/SETUP.md), then run this again."
    say "Creating $(tidy "$VENV") with $("$PY" --version 2>&1) ($(tidy "$PY"))"
    "$PY" -m venv "$VENV" || fail "Could not create the virtual environment."
fi

PYV="$VENV/bin/python"

say "Upgrading pip inside .venv"
"$PYV" -m pip install --upgrade pip || fail "pip could not upgrade itself. Check the internet connection."

say "Installing the lab's packages into .venv (torch is large; the first time can take several minutes)"
"$PYV" -m pip install -r requirements.txt || fail "Installing requirements.txt failed. Scroll up for pip's message."

say "Installing genroofs from this folder (editable)"
"$PYV" -m pip install -e . --no-deps || fail "Installing genroofs failed. Scroll up for pip's message."

say "Running the self-check"
if "$PYV" -m genroofs.check; then
    printf '\nDone. To use this setup later, open Terminal in this folder and run:\n'
    printf '    source .venv/bin/activate\n'
    printf 'Then, for example:  python -m pytest -q tests   or   jupyter lab\n'
else
    fail "The self-check listed something to fix (see above). Fix it, then run this script again."
fi
