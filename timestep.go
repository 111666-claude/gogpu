package gogpu

import (
	"math"
	"sync"
	"time"
)

// DefaultMaxSubSteps caps how many fixed simulation steps may run inside one
// rendered frame when the loop falls behind real time. At a 1/60s step this
// covers roughly an 83ms stall without ever scheduling an unbounded catch-up.
const DefaultMaxSubSteps = 5

// maxVariableFrameDelta clamps the single-frame delta handed to OnUpdate in
// the legacy variable-step mode. 66ms ≈ 15 FPS minimum — WaitEvents can block
// for seconds on an idle window, and without the clamp physics would teleport
// on the first frame after the wake-up.
const maxVariableFrameDelta = 0.066

// TimestepStats reports fixed-step scheduler activity since Run() started.
// All counters are cumulative and safe to read from any goroutine.
type TimestepStats struct {
	// FixedSteps is the total number of fixed OnUpdate callbacks executed.
	FixedSteps uint64
	// VariableFrames is the total number of OnUpdate callbacks executed in
	// the legacy variable-step mode (one per rendered frame).
	VariableFrames uint64
	// DroppedSteps counts fixed steps that were never simulated because a
	// frame accumulated more than MaxSubSteps of backlog (spiral-of-death
	// protection). The backlog remainder is discarded rather than carried.
	DroppedSteps uint64
	// Rebases is how often the simulation clock was re-anchored to wall
	// time: pause/resume, focus loss, non-finite/backwards time readings.
	Rebases uint64
}

// frameClock drives the application's simulation time. Two modes share one
// implementation:
//
//   - Variable step (FixedStep <= 0, the default): OnUpdate receives the
//     clamped wall-clock delta once per frame — the historical GoGPU behavior.
//   - Fixed step (FixedStep > 0): OnUpdate always receives exactly FixedStep
//     seconds and runs zero or more times per frame; rendering reads
//     InterpolationAlpha to blend between the previous and current state.
//
// All methods are safe for concurrent use. In practice tick/advance run on the
// main thread while Pause/Resume may be called from any goroutine.
type frameClock struct {
	mu sync.Mutex

	step       float64 // fixed simulation step in seconds; <= 0 = variable mode
	maxSubStep int     // maximum fixed steps consumed per frame
	paused     bool

	// initialized is false until the first tick or an explicit rebase.
	initialized bool
	last        time.Time
	accum       float64 // unconsumed simulated time (fixed mode)
	alpha       float64 // interpolation ratio in [0,1] for the next render

	stats TimestepStats
}

// newFrameClock builds the clock for the configured step. A non-positive
// fixedStep selects the legacy variable-step mode. A non-positive maxSubSteps
// falls back to DefaultMaxSubSteps.
func newFrameClock(fixedStep float64, maxSubSteps int) *frameClock {
	c := &frameClock{step: fixedStep, maxSubStep: maxSubSteps, alpha: 1.0}
	if c.maxSubStep <= 0 {
		c.maxSubStep = DefaultMaxSubSteps
	}
	return c
}

// fixed reports whether the clock runs in fixed-step mode.
func (c *frameClock) fixed() bool { return c.step > 0 }

// rebase re-anchors the clock to now and discards any accumulated backlog.
// Call when simulation time should stop tracking wall time: pause, resume,
// window focus loss, tab visibility changes. Guarantees the next tick cannot
// produce a large catch-up burst regardless of how much wall time elapsed.
func (c *frameClock) rebase(now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.initialized = true
	c.last = now
	c.accum = 0
	c.stats.Rebases++
}

// Pause freezes simulation time. Frames keep rendering (with the current
// interpolation alpha) but no OnUpdate callback runs until Resume.
func (c *frameClock) Pause() {
	c.mu.Lock()
	c.paused = true
	c.mu.Unlock()
}

// Resume unfreezes simulation time and re-anchors the clock, so the time spent
// paused is never replayed as a burst of catch-up steps.
func (c *frameClock) Resume(now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.paused = false
	c.initialized = true
	c.last = now
	c.accum = 0
	c.stats.Rebases++
}

func (c *frameClock) isPaused() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.paused
}

// frameResult is the pure output of one frame-clock advance.
type frameResult struct {
	// fixed is true when the clock runs in fixed-step mode.
	fixed bool
	// paused is true while simulation is frozen; steps is always 0 then.
	paused bool
	// variableDelta is the clamped frame delta for variable-step mode.
	// It is 0 in fixed mode.
	variableDelta float64
	// steps is the number of fixed OnUpdate callbacks this frame requires.
	steps int
	// alpha is the interpolation ratio in [0,1] for rendering.
	alpha float64
}

// tick advances the clock against wall time and returns the work the frame
// loop must perform. now is normally time.Now(); tests inject a scripted clock.
func (c *frameClock) tick(now time.Time) frameResult {
	dt := c.measure(now)
	return c.advance(dt)
}

// measure returns the wall-time delta since the previous reading and records
// now as the latest reading, without advancing the simulation. An inactive
// (demand-driven, idle) frame uses this together with advance(0) so the clock
// stays anchored to wall time without simulating across the idle gap.
func (c *frameClock) measure(now time.Time) float64 {
	if !c.initialized {
		c.rebase(now)
		return 0
	}
	dt := now.Sub(c.last).Seconds()
	c.mu.Lock()
	c.last = now
	c.mu.Unlock()
	return dt
}

// advance is the deterministic, time-free core of tick. It takes the frame's
// wall-clock delta in seconds and never reads a clock itself, which lets the
// full fixed-step policy (including NaN/Inf and backwards-time inputs) be
// exercised with a fake clock in unit tests.
func (c *frameClock) advance(dt float64) frameResult {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Non-finite readings are un-advanceable: re-anchor and report no work.
	if math.IsNaN(dt) || math.IsInf(dt, 0) {
		c.accum = 0
		c.stats.Rebases++
		return frameResult{fixed: c.fixed(), paused: c.paused, alpha: c.alpha}
	}
	// Backwards wall time (mocked clocks, NTP jumps on platforms without a
	// monotonic clock): treat as a re-anchor rather than feeding a negative
	// delta into physics.
	if dt < 0 {
		c.accum = 0
		c.stats.Rebases++
		return frameResult{fixed: c.fixed(), paused: c.paused, alpha: c.alpha}
	}

	if c.paused {
		// Frozen: render with the last alpha, never accumulate paused time.
		return frameResult{fixed: c.fixed(), paused: true, alpha: c.alpha}
	}

	if !c.fixed() {
		if dt > maxVariableFrameDelta {
			dt = maxVariableFrameDelta
		}
		c.stats.VariableFrames++
		// Variable mode has no previous/current simulation boundary: every
		// rendered frame represents the latest state exactly.
		c.alpha = 1.0
		return frameResult{fixed: false, variableDelta: dt, alpha: 1.0}
	}

	c.accum += dt

	n := int(c.accum / c.step)
	if n > c.maxSubStep {
		c.stats.DroppedSteps += uint64(n - c.maxSubStep)
		n = c.maxSubStep
		// Discard every whole step the cap cannot simulate and keep only the
		// fractional remainder. Carrying the backlog would make the next
		// frames over-simulate too — a sustained spiral that keeps the picture
		// jittering long after the stall is over.
		c.accum = math.Mod(c.accum, c.step)
	} else {
		c.accum -= float64(n) * c.step
	}

	c.stats.FixedSteps += uint64(n)
	c.alpha = c.accum / c.step
	// Defensive clamp: floating-point remainder must stay within [0,1].
	if c.alpha < 0 {
		c.alpha = 0
	} else if c.alpha > 1 {
		c.alpha = 1
	}
	return frameResult{fixed: true, steps: n, alpha: c.alpha}
}

// interpolationAlpha returns the render blend ratio between the previous and
// the latest simulated state: 0 = exactly on the previous step, 1 = exactly on
// the latest step. Always 1.0 in variable-step mode.
func (c *frameClock) interpolationAlpha() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.alpha
}

// snapshot returns a copy of the cumulative scheduler counters.
func (c *frameClock) snapshot() TimestepStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}
