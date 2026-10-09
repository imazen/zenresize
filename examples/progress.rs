//! `cargo run --example progress`: the caller owns observation and completion.
use how_far::{Outcome, ResultExt, Total};
use how_far_along::{Phase, PulseTree, Status};
use zenresize::{PixelDescriptor, ResizeConfig, Resizer};

fn main() {
    let config = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let tree = PulseTree::new(Phase::new("resize", Total::Unknown), enough::Unstoppable);
    let observer = tree.observer();
    let result = Resizer::new(&config)
        .try_resize_with_pulse(&vec![128; 20 * 20 * 4], &tree)
        .finish_phase(tree);
    assert_eq!(result.unwrap().len(), 10 * 10 * 4);
    let snapshot = observer.snapshot();
    assert_eq!(snapshot.children[0].completed, 10);
    assert_eq!(snapshot.status, Status::Finished(Outcome::Succeeded));
    println!(
        "resized {} rows: {:?}",
        snapshot.children[0].completed, snapshot.status
    );
}
