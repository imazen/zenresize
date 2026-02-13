# zenresize TODO

## Replace guards with bytemuck cast + safe_unaligned_simd

The hoisted-bounds guard system eliminated hand-written `unsafe` from x86.rs, but has a
soundness flaw: access-time indices are only range-checked in debug mode. In release,
passing an out-of-range index to a guard method is silent UB.

**Insight:** Casting `&[u8]` to `&[[u8; 16]]` (via bytemuck or `as_chunks`) gives you a
slice of fixed-size arrays. These can be passed directly to archmage's `safe_unaligned_simd`
functions — no guards needed, no unsafe, bounds proven by Rust's normal slice indexing.

**25 of 36 guards (Pattern A/B4/C/E) can be replaced this way.** These are all cases where
stride == width (contiguous non-overlapping chunks):
- All conversion kernels (u8↔f32, premul/unpremul)
- V kernel outputs
- Weight table access (cast `&[i16]` to `&[[i16; 16]]`, index with `ew_base + g`)
- V f32 row accumulation (sequential unrolled 4x)

**11 guards (Pattern B/D/F) still need an offset mechanism.** These are the H kernels
where `left` comes from the weight table (data-dependent, not chunk-aligned) and the
batch V kernel (array-indexed row offsets). Options for these:
- Always-check guards (assert index in range even in release; LLVM elides for sequential)
- Accept the debug-only check (current approach, same risk as `get_unchecked`)

See `hoisted-bounds/unchecked-examples.md` for the full inventory of all 36 guard patterns.
