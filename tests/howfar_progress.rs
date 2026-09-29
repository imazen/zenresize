use core::sync::atomic::{AtomicU64, Ordering};

use howfar::Report;
use howfar_along::{Outcome, Phase, Progress, Total, poll::ControlHandle};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

struct CancelAfter {
    progress: Progress,
    control: ControlHandle,
    seen: AtomicU64,
    threshold: u64,
}

impl Report for CancelAfter {
    fn advance(&self, completed: u64) {
        self.progress.advance(completed);
        if self.seen.fetch_add(completed, Ordering::Relaxed) + completed >= self.threshold {
            self.control.cancel();
        }
    }
}

#[test]
fn output_rows_reach_the_plan_total_and_match_the_existing_resize() {
    let config = ResizeConfig::builder(128, 128, 64, 64)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let input = vec![128_u8; 128 * 128 * 4];
    let mut output = vec![0_u8; 64 * 64 * 4];
    let expected = Resizer::new(&config).resize(&input);
    let mut phase = Phase::new("resize", Total::Exact(64));
    let observer = phase.observer();

    Resizer::new(&config)
        .try_resize_into_with_progress(&input, &mut output, &enough::Unstoppable, &phase.progress())
        .unwrap();
    phase.finish().unwrap();

    assert_eq!(output, expected);
    assert_eq!(observer.snapshot().completed, 64);
    assert_eq!(observer.snapshot().fraction(), Some(1.0));
}

#[test]
fn cancellation_keeps_only_completed_rows_in_the_snapshot() {
    let config = ResizeConfig::builder(128, 128, 64, 64)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let input = vec![128_u8; 128 * 128 * 4];
    let mut output = vec![0_u8; 64 * 64 * 4];
    let mut phase = Phase::new("resize", Total::Exact(64));
    let observer = phase.observer();
    let control = ControlHandle::new();
    let reporter = CancelAfter {
        progress: phase.progress(),
        control: control.clone(),
        seen: AtomicU64::new(0),
        threshold: 12,
    };

    assert_eq!(
        Resizer::new(&config).try_resize_into_with_progress(
            &input,
            &mut output,
            &control,
            &reporter
        ),
        Err(enough::StopReason::Cancelled),
    );
    phase.finish_with(Outcome::Cancelled).unwrap();
    let snapshot = observer.snapshot();
    assert!(snapshot.completed >= 12 && snapshot.completed < 64);
    assert_eq!(snapshot.completed, reporter.seen.load(Ordering::Relaxed));
}
