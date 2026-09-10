#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root/.deps:$repo_root/icefall${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root/icefall/egs/librispeech/ASR"
exec python zipformer/train_distill.py "$@"
