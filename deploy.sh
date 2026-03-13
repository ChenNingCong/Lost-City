#!/usr/bin/env bash
# Deploy Lost Cities to a Hugging Face Space.
#
# Usage:
#   ./deploy.sh              # space name defaults to "lost-cities"
#   ./deploy.sh my-space

set -euo pipefail

NAME="${1:-lost-cities}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# Read username from logged-in credentials
USERNAME=$(hf auth whoami 2>/dev/null | head -1)
if [[ -z "$USERNAME" ]]; then
    echo "Error: not logged in. Run: hf login"
    exit 1
fi

SPACE="${USERNAME}/${NAME}"
echo "Deploying to ${SPACE} ..."

# ── Create space ───────────────────────────────────────────────────────────────
python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('${SPACE}', repo_type='space', space_sdk='docker', exist_ok=True)
"

# ── Upload source files ────────────────────────────────────────────────────────
for f in game.py agent.py requirements.txt README.md Dockerfile; do
    echo "  uploading: ${f}"
    hf upload "${SPACE}" "${DIR}/${f}" "${f}" --repo-type space
done

# main_hf.py is uploaded as main.py
echo "  uploading: main_hf.py -> main.py"
hf upload "${SPACE}" "${DIR}/main_hf.py" "main.py" --repo-type space

# ── Patch API base URL in HTML before uploading ────────────────────────────────
HTML_TMP=$(mktemp --suffix=.html)
trap 'rm -f "${HTML_TMP}"' EXIT
sed 's|const API_BASE_URL = "http://127.0.0.1:8000";|const API_BASE_URL = "";|' \
    "${DIR}/lost-cities.html" > "${HTML_TMP}"
echo "  uploading: lost-cities.html (patched API URL)"
hf upload "${SPACE}" "${HTML_TMP}" "lost-cities.html" --repo-type space

# ── Newest model checkpoint ────────────────────────────────────────────────────
NEWEST=$(find "${DIR}/model" -name "*.pkt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -n "$NEWEST" ]]; then
    echo "  uploading model: $(basename "$NEWEST")"
    hf upload "${SPACE}" "$NEWEST" "model/$(basename "$NEWEST")" --repo-type space
else
    echo "  warning: no .pkt model files found, trained agent will not be available"
fi

echo ""
echo "Done! https://huggingface.co/spaces/${SPACE}"
