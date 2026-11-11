#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Preparando ambiente Python..."
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

echo "Instalando dependencias..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

mkdir -p outputs
echo "Aplicacao disponivel em http://127.0.0.1:8000"
exec .venv/bin/python -m uvicorn medmonitor.api:app --host 127.0.0.1 --port 8000