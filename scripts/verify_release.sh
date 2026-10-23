#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
"${PYTHON_BIN}" examples/minimal_forward.py
"${PYTHON_BIN}" -m pytest -q
sha256sum -c results/www_validation_snapshot/SHA256SUMS.txt
echo "release verification: PASS"
