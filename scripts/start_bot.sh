#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate

echo $$ > /tmp/thalex_bot.pid
trap "rm -f /tmp/thalex_bot.pid" EXIT

exec python src/main.py "$@"
