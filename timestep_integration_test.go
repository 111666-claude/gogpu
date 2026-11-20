package gogpu

import (
	"testing"
	"time"

	"github.com/gogpu/gogpu/input"
	"github.com/gogpu/gogpu/internal/platform"
	"github.com/gogpu/gpucontext"
)

// scriptedManager replays queued platform events. PollEvents drains the head
// batch and returns EventNone once it is empty; tests enqueue one batch per
// runFrame call.
type scriptedManager struct {
	mockManager
	queued [][]platform.Event
}

func (m *scriptedManager) enqueue(ev ...platform.Event) {
	m.queued = append(m.queued, ev)
}

func (m *scriptedManager) PollEvents() platform.Event {
	for len(m.queued) > 0 {
		batch := m.queued[0]
		if len(batch) == 0 {
			m.queued = m.queued[1:]
			continue
		}
		ev := batch[0]
		m.queued[0] = batch[1:]
		return ev
	}
	return platform.Event{}
}

// newFixedStepApp builds an App whose frame machinery runs without a GPU:
// no platform window means renderFrameMultiThread exits early (zero visible
// frames), so the OnUpdate/input/clock paths are exercised deterministically.
func newFixedStepApp(t *testing.T, step time.Duration, maxSub int) *App {
	t.Helper()
	app := &App{
		manager:       &scriptedManager{},
		renderLoop:    &mockRenderLoop{},
		windowManager: newWindowManager(),
		animations:    &AnimationController{},
		inputState:    input.New(),
		config: Config{
			ContinuousRender: true,
			FixedStep:        step,
			MaxSubSteps:      maxSub,
		},
	}
	app.running.Store(true)
	app.invalidator = newInvalidator(func() {})
	app.ensureClock()
	return app
}

func approx(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

// TestRunFrame_FixedStepUpdateCount drives frames of scripted wall durations
// through the clock's deterministic advance plus the real dispatchSimulation
// path (exactly what runFrame calls) and checks invocation count and dt.
func TestRunFrame_FixedStepUpdateCount(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	var dts []float64
	app.onUpdate = func(dt float64) { dts = append(dts, dt) }

	durations := []float64{0.01, 0.01, 0.025, 0.005}
	wantPerFrame := []int{1, 1, 2, 1}
	totalWant := 0
	for i, d := range durations {
		result := app.frameClock.advance(d)
		app.dispatchSimulation(result)
		if result.steps != wantPerFrame[i] {
			t.Errorf("frame %d steps = %d, want %d", i, result.steps, wantPerFrame[i])
		}
		totalWant += wantPerFrame[i]
	}
	if len(dts) != totalWant {
		t.Fatalf("OnUpdate called %d times, want %d", len(dts), totalWant)
	}
	for i, dt := range dts {
		if dt != step.Seconds() {
			t.Errorf("update %d dt = %v, want exact %v", i, dt, step.Seconds())
		}
	}
	if got := app.TimestepStats().FixedSteps; got != uint64(totalWant) {
		t.Errorf("stats FixedSteps = %d, want %d", got, totalWant)
	}
}

// TestRunFrame_FixedStepWithInjectedClock goes through the real tick() entry
// point (the one runFrame uses) with a scripted clock, proving the
// tick→advance→dispatch wiring honors a fixed frame duration end to end.
func TestRunFrame_FixedStepWithInjectedClock(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	var updates int
	app.onUpdate = func(dt float64) {
		updates++
		if dt != step.Seconds() {
			t.Errorf("dt = %v, want %v", dt, step.Seconds())
		}
	}

	fc := newFakeClock()
	app.clockRebase(fc.t)

	// A 35ms frame: tick measures now-last internally.
	r := app.frameClock.tick(fc.advance(35 * time.Millisecond))
	app.dispatchSimulation(r)
	if updates != 3 {
		t.Errorf("updates after 35ms frame = %d, want 3", updates)
	}
	if !approx(app.InterpolationAlpha(), 0.5) {
		t.Errorf("alpha = %v, want 0.5", app.InterpolationAlpha())
	}
}

// TestRunFrame_FullLoopFixedStep drives the actual runFrame with a pinned
// clock reading to prove end-to-end (event → clock → dispatch) behavior in
// continuous fixed-step mode.
func TestRunFrame_FullLoopFixedStep(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	var updates int
	app.onUpdate = func(dt float64) {
		updates++
		if dt != step.Seconds() {
			t.Errorf("dt = %v, want %v", dt, step.Seconds())
		}
	}

	// Anchor at a fixed point, then make measure() see a 35ms gap by pinning
	// last 35ms behind the next time.Now() reading. runFrame calls
	// clock.measure(time.Now()), so set last = now-35ms immediately beforehand.
	app.frameClock.mu.Lock()
	app.frameClock.last = time.Now().Add(-35 * time.Millisecond)
	app.frameClock.initialized = true
	app.frameClock.mu.Unlock()

	app.runFrame()
	if updates < 2 || updates > 3 {
		t.Errorf("full-loop fixed updates = %d, want 2 or 3 for a ~35ms gap", updates)
	}
	if app.InterpolationAlpha() < 0 || app.InterpolationAlpha() > 1 {
		t.Errorf("full-loop alpha = %v out of [0,1]", app.InterpolationAlpha())
	}
}

// TestRunFrame_FixedStepInterpolationAlpha verifies the render-side contract:
// App.InterpolationAlpha and Context.InterpolationAlpha expose the accumulator
// fraction in [0,1], and variable mode always reports 1.0.
func TestRunFrame_FixedStepInterpolationAlpha(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	// 25ms → 2 steps, 5ms remainder → alpha 0.5.
	r := app.frameClock.advance(0.025)
	app.dispatchSimulation(r)
	if got := app.InterpolationAlpha(); !approx(got, 0.5) {
		t.Errorf("App.InterpolationAlpha = %v, want 0.5", got)
	}
	ctx := newContext(app.renderer, 1.0)
	ctx.setInterpolationAlpha(app.InterpolationAlpha())
	if got := ctx.InterpolationAlpha(); !approx(got, 0.5) {
		t.Errorf("Context.InterpolationAlpha = %v, want 0.5", got)
	}

	// Variable mode: alpha pinned to 1.0.
	vapp := newFixedStepApp(t, 0, 0)
	vr := vapp.frameClock.advance(0.016)
	vapp.dispatchSimulation(vr)
	if got := vapp.InterpolationAlpha(); got != 1.0 {
		t.Errorf("variable-mode alpha = %v, want 1.0", got)
	}
}

// TestRunFrame_InputConsumedAtDeterministicFrame ensures input edges are
// visible during every fixed step of a frame and retired exactly once at the
// end, regardless of how many steps run.
func TestRunFrame_InputConsumedAtDeterministicFrame(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	// Queue one KeyDown platform event, processed by the real event path.
	mgr := app.manager.(*scriptedManager)
	mgr.enqueue(platform.Event{
		Type:     platform.EventKeyDown,
		WindowID: 0,
		Key:      gpucontext.KeySpace,
	})

	var edgeSeen []bool
	app.onUpdate = func(float64) {
		edgeSeen = append(edgeSeen, app.inputState.Keyboard().JustPressed(input.KeySpace))
	}

	app.processEventsMultiThread()
	// A 35ms frame runs 3 fixed steps — each must observe the same edge.
	r := app.frameClock.advance(0.035)
	app.dispatchSimulation(r)

	if len(edgeSeen) != 3 {
		t.Fatalf("expected 3 fixed updates, got %d", len(edgeSeen))
	}
	for i, seen := range edgeSeen {
		if !seen {
			t.Errorf("step %d did not see the JustPressed edge", i)
		}
	}
	// After the frame's inputState.Update(), the edge is retired.
	if app.inputState.Keyboard().JustPressed(input.KeySpace) {
		t.Error("JustPressed edge survived past the frame boundary")
	}
	if !app.inputState.Keyboard().Pressed(input.KeySpace) {
		t.Error("held key state was lost when the edge retired")
	}
}

// TestRunFrame_FixedStepReplayDeterminism replays the same scripted frame
// durations on two independent apps and requires identical trajectories:
// update counts, dts, alpha sequence and scheduler stats.
func TestRunFrame_FixedStepReplayDeterminism(t *testing.T) {
	step := 10 * time.Millisecond
	durations := []float64{0.016, 0.016, 0.017, 0.05, 0.016, 0.016, 0.2}

	run := func() (steps int, dts []float64, alphas []float64, stats TimestepStats) {
		app := newFixedStepApp(t, step, 5)
		app.onUpdate = func(dt float64) {
			steps++
			dts = append(dts, dt)
		}
		for _, d := range durations {
			r := app.frameClock.advance(d)
			app.dispatchSimulation(r)
			alphas = append(alphas, r.alpha)
		}
		return steps, dts, alphas, app.TimestepStats()
	}

	s1, dts1, a1, st1 := run()
	s2, dts2, a2, st2 := run()

	if s1 != s2 {
		t.Fatalf("replay step count differs: %d vs %d", s1, s2)
	}
	if len(dts1) != len(dts2) {
		t.Fatalf("replay dt length differs: %d vs %d", len(dts1), len(dts2))
	}
	for i := range dts1 {
		if dts1[i] != dts2[i] {
			t.Errorf("replay dt %d differs: %v vs %v", i, dts1[i], dts2[i])
		}
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Errorf("replay alpha %d differs: %v vs %v", i, a1[i], a2[i])
		}
	}
	if st1 != st2 {
		t.Errorf("replay stats differ: %+v vs %+v", st1, st2)
	}
	// The 0.2s frame (20 steps) must have been capped and backlog dropped.
	if st1.DroppedSteps == 0 {
		t.Error("expected dropped steps from the oversized frame")
	}
	if st1.FixedSteps != uint64(s1) {
		t.Errorf("stats FixedSteps %d != observed %d", st1.FixedSteps, s1)
	}
}

// TestRunFrame_PauseResumeSuppressesUpdates exercises the public App API.
func TestRunFrame_PauseResumeSuppressesUpdates(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)

	var updates int
	app.onUpdate = func(float64) { updates++ }

	app.PauseSimulation()
	if !app.IsSimulationPaused() {
		t.Fatal("IsSimulationPaused false after PauseSimulation")
	}
	r := app.frameClock.advance(0.05)
	app.dispatchSimulation(r)
	if updates != 0 {
		t.Errorf("paused app ran %d updates, want 0", updates)
	}

	app.ResumeSimulation()
	r = app.frameClock.advance(0.01)
	app.dispatchSimulation(r)
	if updates != 1 {
		t.Errorf("resumed app ran %d updates, want 1", updates)
	}
}

// TestRunFrame_PauseAlsoSuppressesVariableMode proves pause is mode-agnostic:
// the legacy variable-step loop must not invoke OnUpdate while frozen either.
func TestRunFrame_PauseAlsoSuppressesVariableMode(t *testing.T) {
	app := &App{
		manager:       &scriptedManager{},
		renderLoop:    &mockRenderLoop{},
		windowManager: newWindowManager(),
		animations:    &AnimationController{},
		config:        Config{ContinuousRender: true},
	}
	app.running.Store(true)
	app.invalidator = newInvalidator(func() {})
	app.ensureClock()

	var updates int
	app.onUpdate = func(float64) { updates++ }

	app.PauseSimulation()
	r := app.frameClock.advance(0.016)
	if !r.paused {
		t.Fatal("variable-mode result should report paused")
	}
	app.dispatchSimulation(r)
	if updates != 0 {
		t.Errorf("paused variable-mode app ran %d updates, want 0", updates)
	}

	app.ResumeSimulation()
	r = app.frameClock.advance(0.016)
	app.dispatchSimulation(r)
	if updates != 1 {
		t.Errorf("resumed variable-mode app ran %d updates, want 1", updates)
	}
}

// TestRunFrame_FocusLossRebasesClock checks the real event path re-anchors the
// clock so an unfocused interval cannot be replayed as catch-up.
func TestRunFrame_FocusLossRebasesClock(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)
	mgr := app.manager.(*scriptedManager)
	mgr.enqueue(platform.Event{Type: platform.EventFocus, WindowID: 0, Focused: false})

	before := app.TimestepStats()
	app.processEventsMultiThread()
	after := app.TimestepStats()
	if after.Rebases <= before.Rebases {
		t.Error("focus event did not rebase the simulation clock")
	}
	if app.HasFocus() {
		t.Error("HasFocus should be false after focus-loss event")
	}

	// Even a huge subsequent frame starts from the rebase: the cap applies,
	// with no inherited backlog from the time spent unfocused.
	r := app.frameClock.advance(60.0)
	if r.steps > app.config.MaxSubSteps {
		t.Errorf("post-focus-loss frame ran %d steps, cap is %d", r.steps, app.config.MaxSubSteps)
	}
}

// TestRunFrame_ShutdownDeliversNoCallbacks ensures that once Quit is requested,
// a queued runFrame exits before any user callback.
func TestRunFrame_ShutdownDeliversNoCallbacks(t *testing.T) {
	app := newFixedStepApp(t, 10*time.Millisecond, 5)
	var updates, draws int
	app.onUpdate = func(float64) { updates++ }
	app.onDraw = func(*Context) { draws++ }

	app.Quit()
	app.runFrame()

	if updates != 0 || draws != 0 {
		t.Errorf("after Quit: updates=%d draws=%d, want both 0", updates, draws)
	}
}

// TestRunFrame_IdleFrameDoesNotSimulate verifies the loopActive gating used by
// runFrame: an inactive (demand-driven) frame feeds dt=0 to the scheduler even
// after a long wall-clock gap, so no fixed steps accumulate while the UI is
// idle, while an active frame with the same measured gap simulates normally.
func TestRunFrame_IdleFrameDoesNotSimulate(t *testing.T) {
	step := 10 * time.Millisecond
	app := newFixedStepApp(t, step, 5)
	fc := newFakeClock()
	app.clockRebase(fc.t)

	var updates int
	app.onUpdate = func(float64) { updates++ }

	// Simulate what runFrame does on an idle frame: measure the gap, then
	// zero it before advance because the loop is not active.
	dt := app.frameClock.measure(fc.advance(10 * time.Second))
	if dt != 10.0 {
		t.Fatalf("measured dt = %v, want 10s", dt)
	}
	app.dispatchSimulation(app.frameClock.advance(0))
	if updates != 0 {
		t.Errorf("idle frame ran %d updates across a 10s gap", updates)
	}

	// Active frame (ContinuousRender/animation): the measured dt is honored.
	dt = app.frameClock.measure(fc.advance(35 * time.Millisecond))
	app.dispatchSimulation(app.frameClock.advance(dt))
	if updates != 3 {
		t.Errorf("active frame ran %d updates, want 3", updates)
	}
}

// TestFrameClock_MeasureDoesNotAdvanceSimulation checks measure only updates
// the anchor: successive measures followed by advance(0) keep alpha/steps at
// rest.
func TestFrameClock_MeasureDoesNotAdvanceSimulation(t *testing.T) {
	step := 10 * time.Millisecond
	c, fc := fixedTestClock(step, 5)

	for range 3 {
		_ = c.measure(fc.advance(time.Hour))
		r := c.advance(0)
		if r.steps != 0 || r.alpha != 0 {
			t.Fatalf("idle measurement produced steps=%d alpha=%v, want 0/0", r.steps, r.alpha)
		}
	}
	if c.snapshot().FixedSteps != 0 {
		t.Error("FixedSteps advanced while only measuring")
	}
}

// TestRunFrame_VariableModeUnchanged proves the legacy path still runs exactly
// one OnUpdate per frame with the same clamp semantics as before.
func TestRunFrame_VariableModeUnchanged(t *testing.T) {
	app := &App{
		manager:       &scriptedManager{},
		renderLoop:    &mockRenderLoop{},
		windowManager: newWindowManager(),
		animations:    &AnimationController{},
		inputState:    input.New(),
		config:        Config{ContinuousRender: true},
	}
	app.running.Store(true)
	app.invalidator = newInvalidator(func() {})
	app.ensureClock()

	var dts []float64
	app.onUpdate = func(dt float64) { dts = append(dts, dt) }

	r := app.frameClock.advance(0.016)
	app.dispatchSimulation(r)
	r = app.frameClock.advance(0) // immediate re-tick: zero delta
	app.dispatchSimulation(r)
	if len(dts) != 2 {
		t.Fatalf("variable mode updates = %d, want 2 (one per frame)", len(dts))
	}
	if dts[0] < 0.015 || dts[0] > 0.017 {
		t.Errorf("variable dt = %v, want ~0.016", dts[0])
	}
	if dts[1] != 0 {
		t.Errorf("zero-delta frame dt = %v, want 0", dts[1])
	}

	// Clamp still applies through the pure advance path.
	r = app.frameClock.advance(30.0)
	if r.variableDelta != maxVariableFrameDelta {
		t.Errorf("clamp = %v, want %v", r.variableDelta, maxVariableFrameDelta)
	}
}

// TestRunFrame_VariableModeDemandDrivenKeepsDelta mirrors runFrame's gating:
// a variable-step clock is NOT fixed, so an inactive frame keeps the measured
// wall delta (only fixed-step idle frames are zeroed).
func TestRunFrame_VariableModeDemandDrivenKeepsDelta(t *testing.T) {
	app := &App{
		manager:       &scriptedManager{},
		renderLoop:    &mockRenderLoop{},
		windowManager: newWindowManager(),
		animations:    &AnimationController{},
		config:        Config{ContinuousRender: false}, // demand-driven
	}
	app.running.Store(true)
	app.invalidator = newInvalidator(func() {})
	app.ensureClock()

	var got float64
	app.onUpdate = func(dt float64) { got = dt }

	fc := newFakeClock()
	app.clockRebase(fc.t)
	loopActive := false // neither continuous nor animating
	dt := app.frameClock.measure(fc.advance(100 * time.Millisecond))
	if !loopActive && app.frameClock.fixed() {
		dt = 0
	}
	app.dispatchSimulation(app.frameClock.advance(dt))

	if got != maxVariableFrameDelta {
		t.Errorf("demand-driven variable dt = %v, want clamped %v", got, maxVariableFrameDelta)
	}
}
