#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! compgen -G 'third_party/wheels/*.whl' > /dev/null; then
  exec bash scripts/setup_ml.sh
fi
# Reuse the installed Python 3.10 PyTorch stack without altering .venv or user packages.
python3 -m venv --system-site-packages .venv-v2e
(cd third_party/wheels && sha256sum -c SHA256SUMS)
.venv-v2e/bin/python -m pip install --no-index --find-links third_party/wheels \
  screeninfo==0.8.1 engineering-notation==0.10.0 easygui==0.98.3
mkdir -p artifacts
.venv-v2e/bin/python scripts/v2e_worker.py --check | tee artifacts/v2e_environment.json
