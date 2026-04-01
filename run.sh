#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Starting HP Quiz at http://127.0.0.1:5000"
conda run -n hpquiz flask --app app run --debug
