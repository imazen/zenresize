use enough::{Stop, StopReason, Unstoppable};
use how_far::{
    Complete, NoPulse, Outcome, PhaseSpec, ProgressExt, Report, ResultExt, Stages, Total,
};
use how_far_along::{Observer, Phase, PulseTree, Status};
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
        let pulse = PulseTree::new(Phase::new("resize", Total::Unknown), Unstoppable);
        let observer = pulse.observer();

        let result = Resizer::new(&config).try_resize_with_pulse(&input, &pulse);
        assert_eq!(observer.snapshot().status, Status::Running);
        let output = result.finish_phase(pulse).unwrap();

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
fn cancellation_keeps_completed_rows_and_marks_later_stages_not_run() {
    let config = config(true);
    let input = vec![128_u8; 128 * 128 * 4];
    let mut output = vec![0_u8; 64 * 64 * 4];
    let phase = Phase::new("resize", Total::Unknown);
    let observer = phase.observer();
    let stop = CancelAfterRows {
        observer: observer.clone(),
        threshold: 12,
    };
    let pulse = PulseTree::new(phase, stop);

    assert!(matches!(
        Resizer::new(&config)
            .try_resize_into_with_pulse(&input, &mut output, &pulse)
            .finish_phase(pulse),
        Err(StopReason::Cancelled)
    ));
    let snapshot = observer.snapshot();
    assert_eq!(snapshot.status, Status::Finished(Outcome::Cancelled));
    assert_eq!(snapshot.children[0].completed, 12);
    assert_eq!(
        snapshot.children[0].status,
        Status::Finished(Outcome::Cancelled)
    );
    assert!(
        snapshot.children[1..]
            .iter()
            .all(|child| child.status == Status::Finished(Outcome::NotRun))
    );
}

#[test]
fn resize_nests_under_a_caller_owned_pipeline() {
    let config = config(false);
    let input = vec![128_u8; 128 * 128 * 4];
    let root = PulseTree::new(Phase::new("pipeline", Total::Unknown), Unstoppable);
    let observer = root.observer();
    let result = Stages::new(
        &root,
        &[
            PhaseSpec::new("prepare", 1, Total::Exact(1)),
            PhaseSpec::new("resize", 8, Total::Unknown),
            PhaseSpec::new("save", 1, Total::Exact(1)),
        ],
    )
    .complete_with(|stages| {
        stages.run(|stage| stage.step(1))?;
        let output =
            stages.run(|stage| Resizer::new(&config).try_resize_with_pulse(&input, stage))?;
        stages.run(|stage| stage.step(1))?;
        Ok::<_, StopReason>(output)
    });
    assert_eq!(observer.snapshot().status, Status::Running);
    assert_eq!(
        result.finish_phase(root).unwrap(),
        Resizer::new(&config).resize(&input)
    );

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
    let pulse = PulseTree::new(phase, stop);

    assert!(matches!(
        Resizer::new(&config)
            .try_resize_with_pulse(&input, &pulse)
            .finish_phase(pulse),
        Err(StopReason::Cancelled)
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
        Status::Finished(Outcome::NotRun)
    );
}

#[test]
fn rejected_plan_preserves_pixels_and_records_the_protocol_error() {
    use how_far_really::{
        diagnostics::{DiagnosticPulse, Kind, Options},
        profile::{Profiler, StdClock},
    };
    let config = config(false);
    let input = vec![128_u8; 128 * 128 * 4];
    let expected = Resizer::new(&config).resize(&input);
    let profiler = Profiler::new(StdClock::new(), 32);
    let pulse = DiagnosticPulse::new(
        PulseTree::new(Phase::new("already used", Total::Exact(1)), Unstoppable),
        &profiler,
    );
    let observer = pulse.observer();
    pulse.advance(1);
    let output = Resizer::new(&config)
        .try_resize_with_pulse(&input, &pulse)
        .finish_phase(pulse)
        .unwrap();
    assert_eq!(output, expected);
    let trace = profiler.snapshot().with_progress(observer.snapshot());
    assert!(
        trace
            .diagnose(&Options::default())
            .iter()
            .any(|finding| finding.kind == Kind::ProtocolMisuse)
    );
}

#[test]
fn rejected_plan_keeps_cancellation() {
    let pulse = how_far_along::FnPulse::new("already used", |_| Err(StopReason::Cancelled));
    let observer = pulse.observer();
    pulse.advance(1);
    let result = Resizer::new(&config(true))
        .try_resize_with_pulse(&vec![128; 128 * 128 * 4], &pulse)
        .finish_phase(pulse);
    assert_eq!(result, Err(StopReason::Cancelled));
    assert_eq!(
        observer.snapshot().status,
        Status::Finished(Outcome::Cancelled)
    );
}

#[test]
fn strided_patterned_pixels_match_the_existing_resize_with_each_post_pass() {
    let width = 35usize;
    let height = 29usize;
    let stride = width * 4 + 12;
    let mut input = vec![0xA5; stride * height];
    let mut packed = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let pixel = [(x * 7) as u8, (y * 11) as u8, ((x ^ y) * 13) as u8, 255];
            input[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&pixel);
            packed.extend_from_slice(&pixel);
        }
    }
    for (sharpen, blur) in [(0.0, 0.0), (0.6, 0.0), (0.0, 0.4), (0.6, 0.4)] {
        let builder = || {
            ResizeConfig::builder(width as u32, height as u32, 17, 13)
                .format(PixelDescriptor::RGBA8_SRGB)
                .filter(Filter::Lanczos)
                .post_sharpen(sharpen)
                .post_blur(blur)
        };
        let expected = Resizer::new(&builder().build()).resize(&packed);
        let config = builder().in_stride(stride).build();
        assert_eq!(Resizer::new(&config).resize(&input), expected);
        let pulse = PulseTree::new(Phase::new("strided resize", Total::Unknown), Unstoppable);
        let observer = pulse.observer();
        assert_eq!(
            Resizer::new(&config)
                .try_resize_with_pulse(&input, &pulse)
                .finish_phase(pulse)
                .unwrap(),
            expected
        );
        assert_eq!(observer.snapshot().children[0].completed, 13);
        assert_eq!(
            observer.snapshot().status,
            Status::Finished(Outcome::Succeeded)
        );
    }
}
