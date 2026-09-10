#!/usr/bin/env bash
# Example architecture only: change it to match your actual checkpoints.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${BPE_MODEL:?Set BPE_MODEL to an absolute bpe.model path}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to an absolute checkpoint path}"
: "${MANIFEST_DIR:?Set MANIFEST_DIR to an absolute prepared manifests directory}"
student_args=()
if [[ -n "${STUDENT_CHECKPOINT:-}" ]]; then
  student_args=(--student-init-ckpt "$STUDENT_CHECKPOINT")
fi
exec bash "$repo_root/scripts/train.sh" \
  --world-size 1 --num-epochs 30 --use-fp16 1 \
  --exp-dir "${EXP_DIR:-$repo_root/exp/encoder_kd}" \
  --bpe-model "$BPE_MODEL" --manifest-dir "$MANIFEST_DIR" \
  --full-libri 0 --enable-musan 0 --max-duration 100 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --causal 1 --chunk-size 16 --left-context-frames 128 \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --teacher-num-encoder-layers 2,2,3,4,3,2 \
  --teacher-feedforward-dim 512,768,1024,1536,1024,768 \
  --teacher-encoder-dim 192,256,384,512,384,256 \
  --teacher-encoder-unmasked-dim 192,192,256,256,256,192 \
  --teacher-causal 0 --teacher-chunk-size=-1 \
  --kd-type encoder --kd-encoder-loss cosine \
  --kd-encoder-scale 1.0 --kd-logit-scale 1.0 \
  --kd-temperature 2.0 --kd-warmup-steps 2000 \
  "${student_args[@]}" "$@"
