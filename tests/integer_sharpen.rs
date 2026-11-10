use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer, StreamingResize, WorkingFormat};

#[test]
fn integer_sharpen_changes_pixels_in_both_pipelines() {
    for linear in [false, true] {
        let make = |percent| {
            let builder = ResizeConfig::builder(33, 3, 65, 5)
                .format(PixelDescriptor::RGBX8_SRGB)
                .filter(Filter::Mitchell)
                .resize_sharpen(percent);
            if linear {
                builder.linear().build()
            } else {
                builder.srgb().build()
            }
        };
        let plain = make(0.0);
        let sharp = make(12.0);
        let mut stream = StreamingResize::new(&sharp);
        assert_eq!(
            stream.working_format(),
            if linear {
                WorkingFormat::I16Linear
            } else {
                WorkingFormat::I16Srgb
            }
        );
        let input: Vec<_> = (0..99)
            .flat_map(|i| {
                let v = if i % 33 < 16 { 80 } else { 160 };
                [v, v, v, 255]
            })
            .collect();
        let expected = Resizer::new(&sharp).resize(&input);
        assert_ne!(Resizer::new(&plain).resize(&input), expected);
        let mut actual = Vec::new();
        for row in input.as_chunks::<{ 33 * 4 }>().0 {
            stream.push_row(row).unwrap();
            while let Some(row) = stream.next_output_row() {
                actual.extend_from_slice(row);
            }
        }
        stream.finish();
        while let Some(row) = stream.next_output_row() {
            actual.extend_from_slice(row);
        }
        assert_eq!(actual, expected);
        for percent in [50.0, 90.0, 100.0] {
            assert_eq!(
                StreamingResize::new(&make(percent)).working_format(),
                WorkingFormat::F32
            );
        }
    }
}
