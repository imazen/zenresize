//! Reproducible cross-revision differential check: hash every table field.
//! Redirect each revision's output to a separate file and compare byte-for-byte.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use zenresize::{
    Filter,
    filter::InterpolationDetails,
    weights::{F32WeightTable, I16WeightTable},
};
fn main() {
    for &f in Filter::all() {
        for n in [1, 2, 3, 7, 17, 257] {
            for out in [1, 2, 5, 19, 63, 257, 1021] {
                for modifier in 0..10 {
                    let base = InterpolationDetails::create(f);
                    let d = match modifier {
                        0 => base,
                        1 => base.with_lobe_ratio(0.0),
                        2 => base.with_lobe_ratio(0.02),
                        3 => base.with_lobe_ratio(0.12),
                        4 => base.with_lobe_ratio(0.15),
                        5 => base.with_lobe_ratio(0.5),
                        6 => base.with_lobe_ratio(0.95),
                        7 => base.with_sharpen_percent(0.0),
                        8 => base.with_sharpen_percent(12.0),
                        _ => base.with_sharpen_percent(15.0),
                    };
                    let w = F32WeightTable::new(n, out, &d);
                    let mut hash = DefaultHasher::new();
                    w.left.hash(&mut hash);
                    w.max_taps.hash(&mut hash);
                    for i in 0..out as usize {
                        for &v in w.weights(i) {
                            v.to_bits().hash(&mut hash);
                        }
                    }
                    println!(
                        "f32 {} {n} {out} {modifier} {:016x}",
                        f.name(),
                        hash.finish()
                    );
                    if modifier == 0 {
                        let w = I16WeightTable::new(n, out, &d);
                        let mut hash = DefaultHasher::new();
                        w.left.hash(&mut hash);
                        w.max_taps.hash(&mut hash);
                        for i in 0..out as usize {
                            w.weights(i).hash(&mut hash);
                            w.weights_expanded_4ch(i).hash(&mut hash);
                        }
                        println!("i16 {} {n} {out} {:016x}", f.name(), hash.finish());
                    }
                }
            }
        }
    }
}
