#!/usr/bin/env bash
# Fetch X-IoMT v1.1 ("OriginalDatasetUpdated") at the pinned commit and verify every file
# against data/MANIFEST.sha256. Only the 109 original CSVs (~364 MB) are downloaded; the
# adversarially augmented part of the upstream repository is not needed.
set -euo pipefail

REPO_URL="https://github.com/RuiPintoUBI/X-IoMTDataset"
COMMIT="2e25990dc414b4666585d9a91a945379e86dc0fc"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/raw/X-IoMTDataset"

if [ ! -d "${DEST}/.git" ]; then
  git clone --filter=blob:none --sparse --no-checkout "${REPO_URL}" "${DEST}"
fi
cd "${DEST}"
git sparse-checkout set OriginalDatasetUpdated
git fetch --depth 1 origin "${COMMIT}" 2>/dev/null || true
git checkout --quiet "${COMMIT}"

cd "${DEST}/OriginalDatasetUpdated"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --quiet -c "${ROOT}/data/MANIFEST.sha256"
else
  shasum -a 256 --quiet -c "${ROOT}/data/MANIFEST.sha256"
fi
echo "X-IoMT v1.1 verified: $(ls -1 *.csv | wc -l) CSV files at commit ${COMMIT}"
