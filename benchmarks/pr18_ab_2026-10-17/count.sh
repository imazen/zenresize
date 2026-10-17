#!/usr/bin/env bash
# Retired user-space instructions per call, main vs this checkout.
# Usage: benchmarks/pr18_ab_2026-10-17/count.sh [new-only]  (needs perf)
# new-only counts Resizer::new alone (weight tables and buffers).
set -euo pipefail
cd "$(dirname "$0")"
cargo build --quiet --release --example count
bin="${CARGO_TARGET_DIR:-target}/release/examples/count"
for path in srgb_i16 linear_i16 linear_f32 f32_io; do
  for size in "32 32 16 16 20000" "48 48 64 64 5000" "256 256 128 128 1000" "1024 1024 512 512 20"; do
    read -r w h ow oh iters <<<"$size"
    line="$path ${w}x${h}->${ow}x${oh}"
    for arm in main pr18; do
      ins=$(perf stat -x, -e instructions:u "$bin" "$arm" "$path" $size ${1:-} 2>&1 >/dev/null | awk -F, '/instructions/{print $1}')
      line="$line $arm=$((ins / iters))"
    done
    echo "$line" | awk '{split($3,a,"=");split($4,b,"=");printf "%s %s delta=%+d (%+.3f%%)\n",$0,"",b[2]-a[2],100*(b[2]-a[2])/a[2]}'
  done
done
