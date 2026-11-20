package gogpu

import (
	"math"
	"testing"
	"time"
)

// fakeClock is a scripted monotonic time source for deterministic frame-clock
// tests. Advance moves the clock and tick() feeds the new reading.
type fakeClock struct {
	t time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Unix(1_000_000, 0)}
}

func (f *fakeClock) advance(d time.Duration) time.Time {
	f.t = f.t.Add(d)
	return f.t
}

func fixedTestClock(step time.Duration, maxSub int) (*frameClock, *fakeClock) {
	fc := newFakeClock()
	c := newFrameClock(step.Seconds(), maxSub)
	c.rebase(fc.t)
	return c, fc
}

// TestFrameClock_VariableModeClampsDelta checks the legacy path: one callback
// frame per tick, long idles clamped to maxVariableFrameDelta, alpha pinned to 1.
func TestFrameClock_VariableModeClampsDelta(t *testing.T) {
	fc := newFakeClock()
	c := newFrameClock(0, 0) // variable mode, default max sub steps
	c.rebase(fc.t)

	r := c.tick(fc.advance(16 * time.Millisecond))
	if r.fixed {
		t.Fatal("zero/negative step must select variable mode")
	}
	if r.steps != 0 {
		t.Errorf("variable mode steps = %d, want 0", r.steps)
	}
	if r.variableDelta < 0.015 || r.variableDelta > 0.017 {
		t.Errorf("variable delta = %v, want ~0.016", r.variableDelta)
	}
	if r.alpha != 1.0 {
		t.Errorf("variable alpha = %v, want 1.0", r.alpha)
	}

	// Long idle (e.g. WaitEvents blocked for 30s) is clamped.
	r = c.tick(fc.advance(30 * time.Second))
	if r.variableDelta != maxVariableFrameDelta {
		t.Errorf("idle delta = %v, want clamp %v", r.variableDelta, maxVariableFrameDelta)
	}
	if got := c.snapshot().VariableFrames; got != 2 {
		t.Errorf("VariableFrames = %d, want 2", got)
	}
}

// TestFrameClock_FixedStepScheduling drives a 60Hz simulation with a ~16.7ms
// display and checks the classic accumulator cadence including the 0/1/2 step
// alternation that absorbs the 60Hz-vs-wall-clock drift.
func TestFrameClock_FixedStepScheduling(t *testing.T) {
	step := time.Second / 60
	c, fc := fixedTestClock(step, 5)

	var totalSteps int
	frameDt := 16 * time.Millisecond // slightly faster than the 16.667ms step
	alphas := make([]float64, 0, 12)
	for range 12 {
		r := c.tick(fc.advance(frameDt))
		if !r.fixed {
			t.Fatal("expected fixed mode")
		}
		if r.steps < 0 || r.steps > 1 {
			t.Errorf("steps per frame = %d, want 0 or 1 at this cadence", r.steps)
		}
		if r.alpha < 0 || r.alpha > 1 {
			t.Errorf("alpha out of range: %v", r.alpha)
		}
		totalSteps += r.steps
		alphas = append(alphas, r.alpha)
	}
	// 12 * 16ms = 192ms wall time → 11 or 12 fixed steps (192/16.667 ≈ 11.52).
	if totalSteps < 11 || totalSteps > 12 {
		t.Errorf("total steps = %d, want 11..12 over 192ms", totalSteps)
	}
	if got := c.snapshot().FixedSteps; uint64(totalSteps) != got {
		t.Errorf("stats FixedSteps = %d, want %d", got, totalSteps)
	}
	// Alpha must vary — a constant alpha means interpolation never engages.
	allSame := true
	for _, a := range alphas[1:] {
		if a != alphas[0] {
			allSame = false
		}
	}
	if allSame {
		t.Error("alpha was constant across frames; interpolation is not tracking the accumulator")
	}
}

// TestFrameClock_MultipleStepsPerFrame verifies that a frame covering several
// step boundaries replays every fixed moment with the exact step duration.
func TestFrameClock_MultipleStepsPerFrame(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)

	// A 35ms frame against a 10ms step → 3 steps, 5ms remainder → alpha 0.5.
	r := c.tick(fc.advance(35 * time.Millisecond))
	if r.steps != 3 {
		t.Fatalf("steps = %d, want 3", r.steps)
	}
	if math.Abs(r.alpha-0.5) > 1e-9 {
		t.Errorf("alpha = %v, want 0.5", r.alpha)
	}
}

// TestFrameClock_CatchUpCapAndDrop is the headline stall scenario: a 500ms
// hitch against a 10ms step must not schedule 50 updates. It is capped at
// MaxSubSteps, the remainder is discarded, and the next frame starts clean.
func TestFrameClock_CatchUpCapAndDrop(t *testing.T) {
	step := 10 * time.Millisecond
	maxSub := 5
	c, fc := fixedTestClock(step, maxSub)

	r := c.tick(fc.advance(500 * time.Millisecond)) // 50 steps of backlog
	if r.steps != maxSub {
		t.Fatalf("stall produced %d steps, want cap %d", r.steps, maxSub)
	}
	stats := c.snapshot()
	if stats.DroppedSteps != uint64(50-maxSub) {
		t.Errorf("DroppedSteps = %d, want %d", stats.DroppedSteps, 50-maxSub)
	}

	// The next normal frame must start from a clean (fraction-only) accumulator:
	// a 10ms frame yields exactly one step, no inherited backlog.
	r = c.tick(fc.advance(10 * time.Millisecond))
	if r.steps != 1 {
		t.Errorf("post-stall steps = %d, want 1 (backlog must not carry over)", r.steps)
	}
	if got := c.snapshot().DroppedSteps; got != stats.DroppedSteps {
		t.Errorf("DroppedSteps grew to %d on a normal frame", got)
	}
}

// TestFrameClock_PauseResumeNoBurst ensures frozen time is never simulated.
func TestFrameClock_PauseResumeNoBurst(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)

	// Prime with one normal frame.
	if r := c.tick(fc.advance(10 * time.Millisecond)); r.steps != 1 {
		t.Fatalf("prime steps = %d, want 1", r.steps)
	}

	c.Pause()
	if !c.isPaused() {
		t.Fatal("IsPaused false after Pause")
	}
	// Ticks while paused: no steps, alpha preserved.
	r := c.tick(fc.advance(10 * time.Millisecond))
	if !r.paused || r.steps != 0 {
		t.Errorf("paused tick: paused=%v steps=%d, want true/0", r.paused, r.steps)
	}
	r = c.tick(fc.advance(5 * time.Second))
	if r.steps != 0 {
		t.Errorf("5s paused tick produced %d steps", r.steps)
	}

	// Resume re-anchors: the 5s gap cannot become 500 catch-up steps.
	c.Resume(fc.advance(0))
	r = c.tick(fc.advance(10 * time.Millisecond))
	if r.paused || r.steps != 1 {
		t.Errorf("post-resume: paused=%v steps=%d, want false/1", r.paused, r.steps)
	}
	stats := c.snapshot()
	if stats.FixedSteps != 2 {
		t.Errorf("FixedSteps after pause/resume = %d, want 2", stats.FixedSteps)
	}
	if stats.Rebases < 2 {
		t.Errorf("Rebases = %d, want >= 2 (initial + resume)", stats.Rebases)
	}
}

// TestFrameClock_RebaseDiscardsBacklog covers focus-loss / visibility changes.
func TestFrameClock_RebaseDiscardsBacklog(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)

	// Build a little backlog then re-anchor (window lost focus for minutes).
	_ = c.tick(fc.advance(25 * time.Millisecond))
	c.rebase(fc.advance(10 * time.Minute))

	r := c.tick(fc.advance(10 * time.Millisecond))
	if r.steps != 1 {
		t.Errorf("post-rebase steps = %d, want 1 — unfocused time must be dropped", r.steps)
	}
}

// TestFrameClock_NonFiniteAndBackwardsTime checks defensive handling of
// readings that must never reach physics code.
func TestFrameClock_NonFiniteAndBackwardsTime(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)
	_ = c.tick(fc.advance(10 * time.Millisecond))

	for _, dt := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		r := c.advance(dt)
		if r.steps != 0 {
			t.Errorf("advance(%v) produced %d steps, want 0", dt, r.steps)
		}
	}
	// Backwards wall time: no negative steps, clock re-anchored.
	r := c.advance(-0.05)
	if r.steps != 0 {
		t.Errorf("negative dt produced %d steps, want 0", r.steps)
	}
	// Following normal dt behaves cleanly.
	r = c.advance(0.01)
	if r.steps != 1 {
		t.Errorf("post-recovery steps = %d, want 1", r.steps)
	}
	if got := c.snapshot().Rebases; got < 5 { // initial + NaN + 2*Inf + negative
		t.Errorf("Rebases = %d, want >= 5", got)
	}
}

// TestFrameClock_ZeroDeltaNoStep confirms an immediate re-tick does nothing.
func TestFrameClock_ZeroDeltaNoStep(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)

	r := c.tick(fc.advance(0))
	if r.steps != 0 || r.alpha != 0 {
		t.Errorf("zero-dt frame: steps=%d alpha=%v, want 0/0", r.steps, r.alpha)
	}
}

// TestFrameClock_DefaultMaxSubSteps checks constructor fallback.
func TestFrameClock_DefaultMaxSubSteps(t *testing.T) {
	c := newFrameClock(0.01, 0)
	if c.maxSubStep != DefaultMaxSubSteps {
		t.Errorf("maxSubStep = %d, want default %d", c.maxSubStep, DefaultMaxSubSteps)
	}
	c2 := newFrameClock(0.01, -3)
	if c2.maxSubStep != DefaultMaxSubSteps {
		t.Errorf("negative maxSubStep = %d, want default", c2.maxSubStep)
	}
}
