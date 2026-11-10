//! Compare optimized rounded-rectangle spans with the full-row mask contract.
use zenresize::{
    Filter, MaskFill, MaskSource, PixelDescriptor, ResizeConfig, RoundedRectMask, StreamingResize,
};

struct FullRow(RoundedRectMask);

impl MaskSource for FullRow {
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill {
        self.0.fill_mask_row(dst, y)
    }
}

fn render(mask: impl MaskSource + Send + 'static, width: u32, height: u32) -> Vec<u8> {
    let config = ResizeConfig::builder(width, height, width, height)
        .format(PixelDescriptor::RGBA8_SRGB)
        .filter(Filter::Box)
        .srgb()
        .build();
    let mut resize = StreamingResize::new(&config).with_mask(mask);
    let input: Vec<u8> = (0..width).flat_map(|_| [128, 64, 32, 255]).collect();
    let mut output = Vec::new();
    for _ in 0..height {
        resize.push_row(&input).unwrap();
        while let Some(row) = resize.next_output_row() {
            output.extend_from_slice(row);
        }
    }
    resize.finish();
    while let Some(row) = resize.next_output_row() {
        output.extend_from_slice(row);
    }
    assert_eq!(output.len(), width as usize * height as usize * 4);
    output
}

fn main() {
    for (width, height, radius) in [(17, 17, 5.5), (33, 21, 7.5), (64, 32, 9.0)] {
        let spans = render(
            RoundedRectMask::uniform(width, height, radius),
            width,
            height,
        );
        let full = render(
            FullRow(RoundedRectMask::uniform(width, height, radius)),
            width,
            height,
        );
        let mut mismatches = 0;
        let mut max_alpha_gap = 0;
        for (i, (actual, expected)) in spans
            .as_chunks::<4>()
            .0
            .iter()
            .zip(full.as_chunks::<4>().0)
            .enumerate()
        {
            if actual != expected {
                if mismatches == 0 {
                    println!(
                        "{width}x{height} radius={radius}: first ({}, {}): spans={actual:?} full_row={expected:?}",
                        i % width as usize,
                        i / width as usize
                    );
                }
                mismatches += 1;
                max_alpha_gap = max_alpha_gap.max(actual[3].abs_diff(expected[3]));
            }
        }
        println!(
            "{width}x{height} radius={radius}: mismatched_pixels={mismatches} max_alpha_gap={max_alpha_gap}"
        );
    }
}
