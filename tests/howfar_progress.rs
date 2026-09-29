use enough::{Stop, StopReason, Unstoppable};
use howfar::{Execution, NoPulse, Outcome, PhaseSpec, Pulse, Report, Total};
use howfar_along::{Observer, Phase, PulseTree, Status};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

fn config(post_processing: bool) -> ResizeConfig {
    let builder = ResizeConfig::builder(128, 128, 64, 64)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB);
    if post_processing {
        builder.post_sharpen(0.6).post_blur(0.4).build()
    } else {
        builder.build()
    }
}

#[test]
fn library_plans_nested_stages_through_core_pulse_only() {
    for post_processing in [false, true] {
        let config = config(post_processing);
        let input = vec![128_u8; 128 * 128 * 4];
        let expected = Resizer::new(&config).resize(&input);
        let pulse = PulseTree::new(Phase::new("resize", Total::Unknown), &Unstoppable);
        let observer = pulse.observer();

        let output = Resizer::new(&config)
            .try_resize_with_pulse(&input, &pulse)
            .unwrap();

        assert_eq!(output, expected);
        let snapshot = observer.snapshot();
        assert_eq!(snapshot.status, Status::Finished(Outcome::Succeeded));
        assert_eq!(snapshot.fraction(), Some(1.0));
        assert_eq!(snapshot.children[0].name, "resample");
        assert_eq!(snapshot.children[0].completed, 64);
        assert_eq!(snapshot.children[0].units, "rows");
        assert_eq!(snapshot.children.len(), if post_processing { 3 } else { 1 });
        if post_processing {
            assert_eq!(snapshot.children[1].name, "sharpen");
            assert_eq!(snapshot.children[2].name, "blur");
            assert!(
                snapshot.children[1..]
                    .iter()
                    .all(|child| child.completed == 1)
            );
        }

        assert_eq!(
            Resizer::new(&config)
                .try_resize_with_pulse(&input, &NoPulse)
                .unwrap(),
            expected
        );
    }
}

struct CancelAfterRows {
    observer: Observer,
    threshold: u64,
}
impl Stop for CancelAfterRows {
    fn check(&self) -> Result<(), StopReason> {
        if self.observer.snapshot().children[0].completed >= self.threshold {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_keeps_completed_rows_and_skips_later_stages() {
    let config = config(true);
    let input = vec![128_u8; 128 * 128 * 4];
    let mut output = vec![0_u8; 64 * 64 * 4];
    let phase = Phase::new("resize", Total::Unknown);
    let observer = phase.observer();
    let stop = CancelAfterRows {
        observer: observer.clone(),
        threshold: 12,
    };
    let pulse = PulseTree::new(phase, &stop);

    assert!(matches!(
        Resizer::new(&config).try_resize_into_with_pulse(&input, &mut output, &pulse),
        Err(zenresize::ResizePulseError::Work(StopReason::Cancelled))
    ));
    let snapshot = observer.snapshot();
    assert_eq!(snapshot.status, Status::Finished(Outcome::Cancelled));
    assert!(snapshot.children[0].completed >= 12 && snapshot.children[0].completed < 64);
    assert_eq!(
        snapshot.children[0].status,
        Status::Finished(Outcome::Cancelled)
    );
    assert!(
        snapshot.children[1..]
            .iter()
            .all(|child| child.status == Status::Finished(Outcome::Skipped))
    );
}

#[test]
fn resize_nests_under_a_caller_owned_pipeline() {
    let config = config(false);
    let input = vec![128_u8; 128 * 128 * 4];
    let root = PulseTree::new(Phase::new("pipeline", Total::Unknown), &Unstoppable);
    let observer = root.observer();
    let parts = root
        .split(
            Execution::Sequence,
            &[
                PhaseSpec::new("prepare", 1, Total::Exact(1)),
                PhaseSpec::new("resize", 8, Total::Unknown),
                PhaseSpec::new("save", 1, Total::Exact(1)),
            ],
        )
        .unwrap();
    parts[0].advance(1);
    parts[0].finish(Outcome::Succeeded).unwrap();
    Resizer::new(&config)
        .try_resize_with_pulse(&input, parts[1].as_ref())
        .unwrap();
    parts[2].advance(1);
    parts[2].finish(Outcome::Succeeded).unwrap();
    root.finish(Outcome::Succeeded).unwrap();

    let snapshot = observer.snapshot();
    assert_eq!(snapshot.fraction(), Some(1.0));
    assert_eq!(snapshot.children[1].children[0].completed, 64);
}

struct CancelInSharpen(Observer);
impl Stop for CancelInSharpen {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.snapshot().children[0].status == Status::Finished(Outcome::Succeeded) {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_in_post_processing_keeps_resample_complete() {
    let config = config(true);
    let input = vec![128_u8; 128 * 128 * 4];
    let phase = Phase::new("resize", Total::Unknown);
    let observer = phase.observer();
    let stop = CancelInSharpen(observer.clone());
    let pulse = PulseTree::new(phase, &stop);

    assert!(matches!(
        Resizer::new(&config).try_resize_with_pulse(&input, &pulse),
        Err(zenresize::ResizePulseError::Work(StopReason::Cancelled))
    ));
    let snapshot = observer.snapshot();
    assert_eq!(snapshot.children[0].completed, 64);
    assert_eq!(
        snapshot.children[0].status,
        Status::Finished(Outcome::Succeeded)
    );
    assert_eq!(
        snapshot.children[1].status,
        Status::Finished(Outcome::Cancelled)
    );
    assert_eq!(
        snapshot.children[2].status,
        Status::Finished(Outcome::Skipped)
    );
}

#[test]
fn invalid_parent_phase_returns_a_plan_error_before_resizing() {
    let config = config(false);
    let input = vec![128_u8; 128 * 128 * 4];
    let mut output = vec![0_u8; 64 * 64 * 4];
    let pulse = PulseTree::new(Phase::new("already used", Total::Exact(1)), &Unstoppable);
    pulse.advance(1);

    assert!(matches!(
        Resizer::new(&config).try_resize_into_with_pulse(&input, &mut output, &pulse),
        Err(zenresize::ResizePulseError::Plan(
            howfar::PlanError::AlreadyInUse
        ))
    ));
    assert!(output.iter().all(|byte| *byte == 0));
}
