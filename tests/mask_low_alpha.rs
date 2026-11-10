//! Positive coverage changes opacity, not the straight color of an opaque pixel.
use zenresize::{Filter, MaskFill, MaskSource, PixelDescriptor, ResizeConfig, StreamingResize};

struct Coverage(f32);

impl MaskSource for Coverage {
    fn fill_mask_row(&self, dst: &mut [f32], _y: u32) -> MaskFill {
        dst.fill(self.0);
        MaskFill::Partial
    }
}

#[test]
fn low_mask_coverage_preserves_straight_rgb() {
    for (width, output_height) in [(1, 1), (17, 2), (33, 4)] {
        for coverage in [1.0f32, 0.5, 1.0 / 255.0, 2.0 / 255.0] {
            let config = ResizeConfig::builder(width, 1, width, output_height)
                .format(PixelDescriptor::RGBA8_SRGB)
                .filter(Filter::Box)
                .srgb()
                .build();
            let mut resizer = StreamingResize::new(&config).with_mask(Coverage(coverage));
            let input: Vec<u8> = (0..width).flat_map(|_| [128, 64, 32, 255]).collect();
            resizer.push_row(&input).unwrap();
            resizer.finish();
            let mut rows = 0;
            while let Some(row) = resizer.next_output_row() {
                for pixel in row.as_chunks::<4>().0 {
                    assert_eq!(
                        *pixel,
                        [128, 64, 32, (255.0 * coverage).round() as u8],
                        "width={width} height={output_height} coverage={coverage} row={rows}"
                    );
                }
                rows += 1;
            }
            assert_eq!(rows, output_height);
        }
    }
}
