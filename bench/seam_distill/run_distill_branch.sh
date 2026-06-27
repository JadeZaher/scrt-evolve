#!/usr/bin/env bash
# run_distill_branch.sh — end-to-end local validation of cross-MODEL seam
# distillation (track 29 v1.1): compress a larger TEACHER into a smaller STUDENT
# branch via the two-phase, decoupled pipeline added to scrt_evolve_train/shard.py.
#
#   Phase A (capture): teacher (Mistral-7B) streamed one layer at a time; its
#     per-seam hidden states are written to a disk cache, then the teacher is
#     freed — peak VRAM = one teacher layer, the two models are never co-resident.
#   Phase B (train):   student (TinyLlama-1.1B) trains each block's LoRA + a
#     discard-after width projection to match the cached teacher seam.
#
# Run from WSL2 (CUDA torch in ~/scrt-gpu-venv). Pair shares the Llama tokenizer.
set -euo pipefail

REPO=/mnt/c/Users/atooz/Programming/ai-utils-memory/scrt-evolve
# Self-log to a native path so progress is readable from Windows (Read tool).
exec > >(tee "$REPO/bench/seam_distill/distill_run.log") 2>&1
# TEACHER: Wizard-Vicuna-7B-Uncensored — a LLaMA-7B (32 layers, d=4096) sharing
# TinyLlama's Llama SentencePiece tokenizer (vocab 32000), so hidden states align
# position-by-position. Converted ONCE to safetensors on ext4 (the cached .bin
# can't load on torch<2.6); see convert_teacher_safetensors.py.
TEACHER=${TEACHER:-$HOME/wizard-vicuna-7b-st}
STUDENT=/mnt/c/Users/atooz/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
RUN=${RUN:-$HOME/distill-run}
STEPS=${STEPS:-300}
BLOCK_SIZE=${BLOCK_SIZE:-2}
CALIB=${CALIB:-8}
export PYTHONPATH="$REPO/python"

source ~/scrt-gpu-venv/bin/activate
mkdir -p "$RUN"

if [ "${SKIP_CAPTURE:-0}" = "1" ] && [ -f "$RUN/seams/seam_manifest.json" ]; then
  echo "=================== PHASE A: skipped (reusing cached teacher seams) ==================="
else
echo "=================== PHASE A: teacher seam capture ==================="
SECONDS=0
python3 -m scrt_evolve_train --distill-mode --distill-phase capture \
  --dataset "$RUN/calib.jsonl" --model "$STUDENT" --teacher-model "$TEACHER" \
  --out "$RUN/adapter" --teacher-cache "$RUN/seams" \
  --block-size "$BLOCK_SIZE" --calib-batches "$CALIB" --max-seq-len 256 \
  --layer-map stride --target-modules auto
echo "[capture] elapsed ${SECONDS}s"
du -sh "$RUN/seams" || true
fi

echo "=================== PHASE B: student distill train ==================="
SECONDS=0
python3 -m scrt_evolve_train --distill-mode --distill-phase train \
  --dataset "$RUN/calib.jsonl" --model "$STUDENT" --teacher-model "$TEACHER" \
  --out "$RUN/adapter" --teacher-cache "$RUN/seams" \
  --block-size "$BLOCK_SIZE" --calib-batches "$CALIB" --max-seq-len 256 \
  --steps "$STEPS" --lr 1e-3 --rank 16 --alpha 32 \
  --distill-loss cosine_mse --projection auto --target-modules auto --log-every 50
echo "[train] elapsed ${SECONDS}s"
echo "=== adapter shards ==="
ls -la "$RUN/adapter" || true

if [ "${SKIP_EXPORT:-0}" = "1" ]; then
  echo "=================== PHASE C: skipped (SKIP_EXPORT=1) ==================="
  exit 0
fi
echo "=================== PHASE C: merge + export GGUF ==================="
SECONDS=0
python3 -m scrt_evolve_gguf --model "$STUDENT" --adapter "$RUN/adapter" \
  --out "$RUN/scrt-distill-tinyllama.gguf" --quant Q4_K_M \
  --merge-shards 'adapter-shard-*.safetensors' \
  --work-dir "$RUN/export" --llama-cpp "$HOME/llama.cpp"
echo "[export] elapsed ${SECONDS}s"
echo "=== smaller GGUF artifact (student-sized, distilled from 7B teacher) ==="
ls -la "$RUN/scrt-distill-tinyllama.gguf" || true
du -h "$RUN/scrt-distill-tinyllama.gguf" 2>/dev/null || true
