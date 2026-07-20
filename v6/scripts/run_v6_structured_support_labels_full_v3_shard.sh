#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SHARD_INDEX FREEZE_JSON OUTPUT_ROOT [PROJECT_ROOT]" >&2
  exit 2
fi

SHARD_INDEX="$1"
if [[ ! "$SHARD_INDEX" =~ ^[0-3]$ ]]; then
  echo "SHARD_INDEX must be 0, 1, 2, or 3" >&2
  exit 2
fi

FREEZE="$(realpath "$2")"
OUTPUT_ROOT="$(realpath -m "$3")"
PROJECT_ROOT="${4:-$(pwd)}"
OUTPUT="${OUTPUT_ROOT}/structured_support_labels_full_v3_shard${SHARD_INDEX}"
LOG="${OUTPUT_ROOT}/structured_support_labels_full_v3_shard${SHARD_INDEX}.log"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the process environment}"
mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

RESUME=()
if [[ -d "$OUTPUT" && ! -f "$OUTPUT/SHARD_MANIFEST.json" ]]; then
  RESUME=(--resume)
fi

python v6/scripts/838_score_v6_structured_support_labels_full_v3.py \
  --freeze "$FREEZE" \
  --shard_index "$SHARD_INDEX" \
  --num_shards 4 \
  --output_dir "$OUTPUT" \
  "${RESUME[@]}" 2>&1 | tee -a "$LOG"
