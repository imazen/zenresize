//! Raw numerical oracle driver for benchmarks/conventions_2026-10-15:
//! three channels, explicit input/output files.
use std::io::Write;
use zenresize::{
    Arithmetic, BoundaryMode, Filter, PixelDescriptor, ResizeConfig, ResizeConfigBuilder, Resizer,
    SampleMapping,
};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let [w, h, ow, oh]: [u32; 4] = core::array::from_fn(|i| args[i + 2].parse().unwrap());
    let u8_io = args[1].ends_with("-u8");
    let mode = args[1].trim_end_matches("-u8");
    let fixed_support = |b: ResizeConfigBuilder| b.antialias(false).boundary(BoundaryMode::Clamp);
    let base = ResizeConfig::builder(w, h, ow, oh);
    let builder = match mode.trim_end_matches("-align") {
        "pillow22" => base
            .filter(Filter::CatmullRom)
            .arithmetic(Arithmetic::Fixed22),
        "cubic075" => fixed_support(base.filter(Filter::CubicA075)),
        "cubic05-aa" => base.filter(Filter::CatmullRom),
        "linear" => fixed_support(base.filter(Filter::Triangle)),
        "gaussian" | "gaussian-aa" => base
            .filter(Filter::Gaussian)
            .antialias(mode == "gaussian-aa")
            .sample_mapping(SampleMapping::HalfPixelFloat32),
        "nearest-cv" => base
            .filter(Filter::Nearest)
            .sample_mapping(SampleMapping::AsymmetricFloat64),
        "nearest-torch" => base
            .filter(Filter::Nearest)
            .sample_mapping(SampleMapping::AsymmetricFloat32),
        "nearest" => base
            .filter(Filter::Nearest)
            .sample_mapping(SampleMapping::Asymmetric),
        "nearest-half" => base.filter(Filter::Nearest),
        _ => panic!("unknown convention {mode}"),
    };
    let builder = if mode.ends_with("-align") {
        builder.sample_mapping(SampleMapping::AlignCorners)
    } else {
        builder
    };
    let bytes = std::fs::read(&args[6]).unwrap();
    let mut output = std::fs::File::create(&args[7]).unwrap();
    if u8_io {
        let builder = builder.format(PixelDescriptor::RGB8_SRGB).srgb();
        let config = if mode == "pillow22" {
            builder.build()
        } else {
            builder.arithmetic(Arithmetic::F32).build()
        };
        output
            .write_all(&Resizer::new(&config).resize(&bytes))
            .unwrap();
    } else {
        let input: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let config = builder
            .format(PixelDescriptor::RGBF32_LINEAR)
            .arithmetic(Arithmetic::F32)
            .build();
        for x in Resizer::new(&config).resize_f32(&input) {
            output.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}
