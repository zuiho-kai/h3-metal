#!/bin/bash
# usage: bench/run_fox2.sh <model_dir> <outdir> [extra h3 flags, e.g. --fasth3]
# Runs the reference prompt twice in one process: clip 1 = cold, clip 2 = warm.
# Timing = the "Done -> ... [Xs]" wall clock printed around each h3_generate().
set -u
MODEL="$1"; OUT="$2"; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT/output"
FOX="A red fox picks its way along a fallen mossy log through a misty forest at dawn, pauses to sniff the air, then leaps down into the ferns. Cinematic, shallow depth of field, drifting fog, birdsong."
export H3_REUSE_STEPS="0,2,4,6,8,10,13,16,19" H3_VAE_BF16=1
printf '!open off\n!show off\n!output %s\n%s\n%s\n' "$OUT/output" "$FOX" "$FOX" | \
  /usr/bin/time -l "$ROOT/h3" --profile -d "$MODEL" \
  --width 768 --height 448 --frames 49 --steps 20 --seed 0 --layers 45 --reuse 2 "$@" \
  > "$OUT/bench.log" 2>&1
echo "rc=$?" >> "$OUT/bench.log"
grep -n "Done ->" "$OUT/bench.log"
