#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv-v2e/bin/python ]; then
  echo 'Environnement ML existant : verification sans reinstallation.'
  if .venv-v2e/bin/python scripts/v2e_worker.py --check; then exit 0; fi
  echo 'Installation incomplete : reprise des dependances dans cet environnement.'
else
  python3 -m venv .venv-v2e
fi
.venv-v2e/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-v2e/bin/python -m pip install -r requirements-ml.txt
mkdir -p artifacts
.venv-v2e/bin/python scripts/v2e_worker.py --check | tee artifacts/v2e_environment.json
