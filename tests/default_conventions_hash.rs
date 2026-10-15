//! The encoded RGBA portion of the pre-change default byte capture.
//! Integer arithmetic makes this fixture independent of floating SIMD FMA tiers.
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
#[test]
fn named_defaults_match_pre_conventions_capture() {
    let mut hash = 14695981039346656037u64;
    for &filter in Filter::all().iter().filter(|f| (**f as u8) <= 31) {
        for (w, h, ow, oh) in [
            (1, 7, 9, 3),
            (17, 13, 7, 5),
            (7, 5, 19, 11),
            (23, 19, 23, 19),
        ] {
            let input: Vec<u8> = (0..w * h * 4)
                .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
                .collect();
            let cfg = ResizeConfig::builder(w, h, ow, oh)
                .filter(filter)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
            for byte in Resizer::new(&cfg).resize(&input) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211);
            }
        }
    }
    assert_eq!(hash, 0x6bcaf1e70817a20c);
}
