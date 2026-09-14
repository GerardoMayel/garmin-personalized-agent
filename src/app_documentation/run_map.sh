#!/usr/bin/env bash
# ==============================================================================
# Garmin Personal Insight Agent - Project Tree & README Generator Runner
# ==============================================================================
# This shell script executes the Python project structure mapper to scan
# all folders, files, and generate newreadme.md and project_structure.txt.
#
# Usage:
#   ./src/app_documentation/run_map.sh [options]
#
# Options forwarded to update_readme.py:
#   --with-descriptions   Include annotations/comments from current README
#   --apply-to-root       Directly overwrite root README.md
#   --no-spacers          Omit empty spacer branches between top-level folders
#   --help                Show detailed options
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " 📁 Project Structure Mapper & README Generator"
echo "=========================================================="
echo "📍 Repo root: ${REPO_ROOT}"
echo "📍 Script dir: ${SCRIPT_DIR}"

# Detect Python interpreter
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "❌ Error: Python 3 is required but not found in PATH." >&2
    exit 1
fi

echo "🐍 Using Python: $("${PYTHON_BIN}" --version)"

# Execute the mapping script
"${PYTHON_BIN}" "${SCRIPT_DIR}/update_readme.py" --root "${REPO_ROOT}" "$@"

echo ""
echo "✅ Done! Files generated in ${SCRIPT_DIR}:"
echo "   - newreadme.md          (Full updated README for manual review)"
echo "   - project_structure.txt (Tree representation only)"
echo "=========================================================="
