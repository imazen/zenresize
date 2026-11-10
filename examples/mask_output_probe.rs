//! PR 23 diagnostic: caller-composite outputs must apply the mask exactly once.
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenresize::{
    Background, Filter, MaskFill, MaskSource, PixelDescriptor, ResizeConfig, SolidBackground,
    StreamingResize,
};

struct Pattern;

impl MaskSource for Pattern {
    fn fill_mask_row(&self, dst: &mut [f32], y: u32) -> MaskFill {
        for (x, m) in dst.iter_mut().enumerate() {
            *m = coverage(x, y);
        }
        MaskFill::Partial
    }
}

fn coverage(x: usize, y: u32) -> f32 {
    ((x + y as usize) % 5) as f32 * 0.25
}

fn input(width: u32, height: u32) -> Vec<Vec<u8>> {
    (0..height)
        .map(|y| {
            (0..width)
                .flat_map(|x| {
                    [
                        (31 + x * 7 + y * 3) as u8,
                        (97 + y * 5) as u8,
                        (181 + x * 3) as u8,
                        (64 + (x * 13 + y * 17) % 192) as u8,
                    ]
                })
                .collect()
        })
        .collect()
}

fn premul(mut resize: StreamingResize, rows: &[Vec<u8>]) -> Vec<Vec<f32>> {
    let mut output = Vec::new();
    for row in rows {
        resize.push_row(row).unwrap();
        while let Some(out) = resize.next_output_row_premul_linear_f32() {
            output.push(out.to_vec());
        }
    }
    resize.finish();
    while let Some(out) = resize.next_output_row_premul_linear_f32() {
        output.push(out.to_vec());
    }
    output
}

fn bytes<B: Background>(
    mut resize: StreamingResize<B>,
    rows: &[Vec<u8>],
    external: Option<&[f32]>,
) -> Vec<Vec<u8>> {
    let mut output = Vec::new();
    let drain = |resize: &mut StreamingResize<B>, output: &mut Vec<Vec<u8>>| loop {
        let next = match external {
            Some(bg) => resize.next_output_row_over(bg),
            None => resize.next_output_row(),
        };
        match next {
            Some(row) => output.push(row.to_vec()),
            None => break,
        }
    };
    for row in rows {
        resize.push_row(row).unwrap();
        drain(&mut resize, &mut output);
    }
    resize.finish();
    drain(&mut resize, &mut output);
    output
}

fn main() {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        for (iw, ih, ow, oh) in [(20, 20, 13, 17), (17, 1, 33, 4), (33, 17, 17, 9)] {
            for filter in [
                Filter::Box,
                Filter::MagicKernelSharp,
                Filter::MagicKernelSharp2021,
            ] {
                let config = ResizeConfig::builder(iw, ih, ow, oh)
                    .format(PixelDescriptor::RGBA8_SRGB)
                    .filter(filter)
                    .linear()
                    .build();
                let rows = input(iw, ih);
                let raw = premul(StreamingResize::new(&config), &rows);
                let masked = premul(StreamingResize::new(&config).with_mask(Pattern), &rows);
                assert_eq!(raw.len(), oh as usize);
                assert_eq!(masked.len(), raw.len());
                let mut wrong_components = 0;
                for (y, (r, m)) in raw.iter().zip(&masked).enumerate() {
                    assert_eq!(r.len(), ow as usize * 4);
                    assert_eq!(m.len(), r.len());
                    for (i, (&r, &m)) in r.iter().zip(m).enumerate() {
                        wrong_components +=
                            usize::from((r * coverage(i / 4, y as u32)).to_bits() != m.to_bits());
                    }
                }
                let mut wrong_bytes = 0;
                for white in [false, true] {
                    let bg = if white { [1.0; 4] } else { [0.0; 4] };
                    let external = bg.repeat(ow as usize);
                    let owned = SolidBackground::from_linear(
                        bg[0],
                        bg[1],
                        bg[2],
                        bg[3],
                        PixelDescriptor::RGBA8_SRGB,
                    );
                    let reference = bytes(
                        StreamingResize::with_background(&config, owned)
                            .unwrap()
                            .with_mask(Pattern),
                        &rows,
                        None,
                    );
                    let actual = bytes(
                        StreamingResize::new(&config).with_mask(Pattern),
                        &rows,
                        Some(&external),
                    );
                    assert_eq!(actual.len(), oh as usize);
                    assert_eq!(reference.len(), actual.len());
                    wrong_bytes += reference
                        .iter()
                        .flatten()
                        .zip(actual.iter().flatten())
                        .filter(|(r, a)| r != a)
                        .count();
                }
                println!(
                    "{tier} {iw}x{ih}->{ow}x{oh} {filter:?}: premul_wrong_components={wrong_components} over_wrong_bytes={wrong_bytes}"
                );
            }
        }
    });
    println!("{report}");
}
