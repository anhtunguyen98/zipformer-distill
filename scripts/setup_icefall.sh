#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1
if [[ ! -d "$repo_root/icefall/.git" ]]; then
  git clone https://github.com/k2-fsa/icefall.git "$repo_root/icefall"
  git -C "$repo_root/icefall" checkout "$revision"
fi
actual=$(git -C "$repo_root/icefall" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  echo "Expected icefall $revision, found $actual. Use a separate checkout at the pinned revision." >&2
  exit 1
fi
cp "$repo_root/zipformer/train_distill.py" "$repo_root/zipformer/kd_utils.py" "$repo_root/icefall/egs/librispeech/ASR/zipformer/"
echo "Installed distillation overlay into icefall $revision"
