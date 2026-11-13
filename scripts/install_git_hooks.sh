#!/usr/bin/env bash
# Installs the repository's git hooks into .git/hooks (hooks are not versioned
# by git itself, so every clone must install them once).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for hook in "$ROOT"/tools/git-hooks/*; do
  name="$(basename "$hook")"
  cp "$hook" "$ROOT/.git/hooks/$name"
  chmod +x "$ROOT/.git/hooks/$name"
  echo "Installed $name hook -> .git/hooks/$name"
done
