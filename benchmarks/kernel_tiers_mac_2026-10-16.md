# Resize row kernels on macOS ARM64, 2026-10-16

Source `e3975fb9`, M4 Pro. Command: `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-kernel-tiers-macos`. The recipe ran `cargo bench --bench kernel_tiers -- --cpu-policy=unrestricted --format=llm`. Full output is `~/tmp/zenresize-kernel-tiers.log` on this Mac. The scheduler was unrestricted and several groups reported timing drift; these are exploratory row-kernel results.

The audit covers 28 tier or fused-versus-sequence groups. No measured NEON tier was slower than forced scalar. Examples:

| Row kernel, 1920 pixels | NEON mean | Forced scalar mean |
| --- | ---: | ---: |
| `premultiply_alpha_row` | 1.16 µs | 1.28 µs |
| `filter_h_row_f32` | 7.43 µs | 58.94 µs |
| `filter_v_row_f32` | 4.83 µs | 6.09 µs |
| `f32_to_f16_row` | 418.6 ns | 12.20 µs |
| `srgb_u8_to_linear_f32` | 2.49 µs | 2.47 µs; indistinguishable |

The fused `srgb_u8_to_linear + premultiply` row was 3.13 µs versus 3.39 µs for the sequence. These are row timings, not complete resize speedups. The existing source notes that `u8_to_f32_row` and some other near-ties include dispatch boundary cost, so a scalar route is unsupported here.

`run-heavy`: peak RSS 0.26 GiB, minimum available memory 5217 MiB, peak load 3.99.
