//! A padded resize under an output orientation (issue 25): the oriented output
//! must be the unoriented padded output turned by the orientation, sample for
//! sample. The reference is this crate's own unoriented output, so resampling
//! cancels out and only placement is judged; the turn is written out here from
//! the EXIF definitions rather than taken from the crate.

use zenresize::{Filter, OrientOutput, PixelDescriptor, ResizeConfig, StreamingResize};

const ORIENTATIONS: [OrientOutput; 8] = [
    OrientOutput::Identity,
    OrientOutput::FlipH,
    OrientOutput::Rotate180,
    OrientOutput::FlipV,
    OrientOutput::Transpose,
    OrientOutput::Rotate90,
    OrientOutput::Transverse,
    OrientOutput::Rotate270,
];

/// Where sample `(x, y)` of a `w`×`h` raster lands under `o`.
fn turn(o: OrientOutput, x: u32, y: u32, w: u32, h: u32) -> (u32, u32) {
    match o {
        OrientOutput::Identity => (x, y),
        OrientOutput::FlipH => (w - 1 - x, y),
        OrientOutput::Rotate180 => (w - 1 - x, h - 1 - y),
        OrientOutput::FlipV => (x, h - 1 - y),
        OrientOutput::Transpose => (y, x),
        OrientOutput::Rotate90 => (h - 1 - y, x),
        OrientOutput::Transverse => (h - 1 - y, w - 1 - x),
        OrientOutput::Rotate270 => (y, w - 1 - x),
        _ => unreachable!("eight EXIF orientations"),
    }
}

/// Opaque RGBA8 rows, samples unique per row and column.
fn input(w: u32, h: u32) -> Vec<Vec<u8>> {
    (0..h)
        .map(|y| {
            (0..w)
                .flat_map(|x| {
                    [
                        (x * 6) as u8,
                        (y * 10) as u8,
                        ((x * 7) ^ (y * 5)) as u8,
                        255,
                    ]
                })
                .collect()
        })
        .collect()
}

/// Uneven margins: top 1, right 5, bottom 3, left 2.
fn padded_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .filter(Filter::Robidoux)
        .format(PixelDescriptor::RGBA8_SRGB)
        .padding(1, 5, 3, 2)
        .padding_color([0.2, 0.4, 0.6, 1.0])
        .build()
}

/// Push every row, draining what the resizer offers after each push, then
/// finish and drain the rest.
fn run(resizer: &mut StreamingResize, rows: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for row in rows {
        resizer.push_row(row).unwrap();
        while let Some(r) = resizer.next_output_row() {
            out.extend_from_slice(r);
        }
    }
    resizer.finish();
    while let Some(r) = resizer.next_output_row() {
        out.extend_from_slice(r);
    }
    out
}

#[test]
fn uneven_padding_turns_with_the_output() {
    let mut wrong = Vec::new();
    // A downscale and a same-size pass: in both, the first output row needs
    // several input rows, so rows are captured while input is still arriving.
    for (in_w, in_h, out_w, out_h) in [(40u32, 24u32, 20u32, 12u32), (20, 12, 20, 12)] {
        let rows = input(in_w, in_h);
        let config = padded_config(in_w, in_h, out_w, out_h);
        let (pw, ph) = (out_w + 2 + 5, out_h + 1 + 3);
        let plain = run(&mut StreamingResize::new(&config), &rows);
        assert_eq!(plain.len(), (pw * ph * 4) as usize, "unoriented size");
        for o in ORIENTATIONS {
            let tw = if o.swaps_axes() { ph } else { pw };
            let mut expected = vec![0u8; plain.len()];
            for y in 0..ph {
                for x in 0..pw {
                    let (dx, dy) = turn(o, x, y, pw, ph);
                    let s = ((y * pw + x) * 4) as usize;
                    let d = ((dy * tw + dx) * 4) as usize;
                    expected[d..d + 4].copy_from_slice(&plain[s..s + 4]);
                }
            }
            let got = run(
                &mut StreamingResize::new(&config).with_orientation(o),
                &rows,
            );
            let differ = got
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected.as_chunks::<4>().0)
                .filter(|(a, b)| a != b)
                .count();
            if got.len() != expected.len() || differ != 0 {
                wrong.push(format!(
                    "{in_w}x{in_h} -> {out_w}x{out_h} {o:?}: {differ} of {} pixels differ, {} bytes for {}",
                    pw * ph,
                    got.len(),
                    expected.len()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn flip_h_keeps_uneven_padding_clean_on_every_row() {
    let (in_w, in_h, out_w, out_h) = (40u32, 24u32, 20u32, 12u32);
    let rows = input(in_w, in_h);
    let config = padded_config(in_w, in_h, out_w, out_h);
    let pw = out_w + 2 + 5;
    let plain = run(&mut StreamingResize::new(&config), &rows);
    let pad = &plain[..4];
    let got = run(
        &mut StreamingResize::new(&config).with_orientation(OrientOutput::FlipH),
        &rows,
    );
    // Flipped, the 5 right columns come first and the 2 left ones last.
    let row_len = (pw * 4) as usize;
    let mut stale = Vec::new();
    for (r, row) in got.chunks(row_len).enumerate().skip(1).take(out_h as usize) {
        let dirty = row
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .filter(|&(x, px)| !(5..5 + out_w as usize).contains(&x) && &px[..] != pad)
            .count();
        if dirty != 0 {
            stale.push(format!(
                "row {r}: {dirty} padding pixels are not the padding color"
            ));
        }
    }
    assert!(stale.is_empty(), "{}", stale.join("\n"));
}
