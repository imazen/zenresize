# Tiled V-Filter Exploration Results

## Goal

Investigate whether tiling the fullframe V-filter reduces cache misses and improves performance for wide images.

## Theory

The fullframe V-filter (`filter_v_all_u8_i16`) processes the intermediate buffer with this loop structure:
```
for each output row:
    for each 16-byte chunk across the full row width:
        accumulate tap_count input rows → output chunk
```

Consecutive output rows share overlapping input rows (e.g., 10 of 12 rows for 2× Lanczos3 downscale). The working set per output row is `tap_count × h_row_len` bytes. For 4K (h_row_len=7680, 12 taps): 92KB — exceeds L1 (32KB). By the time we finish `out_y`, the shared rows' data may be evicted from L1.

**Column tiling** reorders the loop to process narrow column strips across all output rows:
```
for each column tile:
    for each output row:
        accumulate only this tile's chunk range
```

This keeps `tap_count × tile_width` bytes in L1 while reusing shared rows across consecutive output rows.

## Implementation

- Scalar fallback: `filter_v_all_u8_i16_tiled_scalar` (src/simd/scalar.rs)
- AVX2 kernel: `filter_v_all_u8_i16_tiled_v3` (src/simd/x86.rs)
- Dispatch: `filter_v_all_u8_i16_tiled` (src/simd/mod.rs)
- Benchmark: `benches/tiled_bench.rs`
- Cachegrind profiler: `benches/cachegrind_vfilter.rs`

All tiled variants produce bit-exact output matching the baseline.

## Benchmark Results (criterion, 4ch u8 Lanczos3)

| Scenario | Taps | h_row_len | Baseline | Best Tiled | Δ |
|----------|------|-----------|----------|------------|---|
| 1024sq 2× | 12 | 2048 | 348µs | 336µs (512c) | -3% |
| 2048sq 2× | 12 | 4096 | 2.43ms | 2.25ms (256c) | -7% |
| 4000×3000 5× | 30 | 3200 | 1.46ms | 1.41ms (256c) | -3% |
| 4K 2× | 12 | 7680 | 2.96ms | 2.81ms (256c) | -5% |
| 8K 2× | 12 | 15360 | 12.33ms | 12.33ms (512c) | 0% |
| 4000×3000 10× | 60 | 1600 | 756µs | 728µs (256c) | -4% |

Sweet spot: 256 chunks/tile (4096 bytes/tile). Too-small tiles (64c) are 5-24% slower from overhead.

## Cache Analysis

### Cachegrind (valgrind, simulated cache)

| Metric | Baseline | Tiled 256c | Change |
|--------|----------|------------|--------|
| Instructions | 270.7M | 263.3M | **-2.7%** |
| Data reads | 78.2M | 67.8M | **-13.3%** |
| D1 read misses | 4.70M | 4.81M | **+2.4%** |
| Data writes | 1.73M | 1.87M | +8.2% |

### Hardware perf counters (perf stat)

| Metric | Baseline | Tiled 256c | Change |
|--------|----------|------------|--------|
| L1 loads | 82.8M | 72.5M | **-12.4%** |
| L1 load misses | 5.80M | 6.51M | **+12.2%** |
| L1 miss rate | 7.0% | 9.0% | **worse** |
| LLC misses | 122K | 126K | +2.9% |

## Key Finding

**Column tiling does NOT reduce cache misses.** L1 misses actually increase by 12% due to scattered output writes and the tile-first iteration order disrupting prefetch patterns.

The 3-7% speedup comes from **better code generation**: the shorter inner loop (256 vs 480 iterations for 4K) produces 12% fewer memory load instructions and 3% fewer total instructions, likely due to:
- Better register allocation with smaller loop trip counts
- Different compiler unrolling decisions
- Reduced loop overhead (branch, increment, compare)

## Why Tiling Doesn't Help Cache

1. **L1 is too small**: With 12 taps, fitting the working set in L1 (32KB) requires tiles of ~170 bytes/row — too small, creating too many tiles with excessive overhead.

2. **Hardware prefetcher already handles the pattern**: The baseline's sequential chunk access across tap rows follows a regular stride pattern that modern CPUs prefetch effectively.

3. **Scattered output writes**: The tiled version writes to different output rows within each tile, disrupting write-combining and adding L1 write misses.

4. **Compute-memory balance**: The AVX2 kernel does ~24 SIMD ops per chunk, taking ~24 cycles. Memory loads take ~4 cycles from L2. The kernel is near compute-bound, limiting the benefit of faster memory access.

## Streaming V-Filter

The streaming V-filter (`filter_v_row_*`) processes one output row at a time from a ring buffer. There is no inter-output-row data reuse within the kernel — each output row's V-filter is independent. Tiling would not help because there's nothing to keep hot across calls.

## Conclusion

Column tiling is not a viable optimization for the fullframe V-filter. The modest speedup (3-7%) comes from a code generation artifact, not genuine cache improvement. This artifact is fragile and compiler-version-dependent.

Potential alternative approaches not explored:
- Fused V+H filter (eliminate intermediate buffer entirely)
- Larger SIMD widths (AVX-512 to process more columns per iteration)
- Software prefetch hints for the strided access pattern
