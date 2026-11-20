// Example: Deterministic fixed-timestep game loop with render interpolation.
//
// The simulation advances in fixed 1/60s steps regardless of display refresh
// rate or frame hitches; rendering interpolates between the previous and
// current physics state using Context.InterpolationAlpha(). A frame that
// stalls for half a second never teleports the ball or schedules dozens of
// catch-up updates — the loop caps catch-up at MaxSubSteps (5) and drops the
// rest.
//
// Usage: CGO_ENABLED=0 go run .
package main

import (
	"log"
	"time"

	"github.com/gogpu/gogpu"
	"github.com/gogpu/gogpu/gmath"
)

const (
	windowW = 800
	windowH = 600
	radius  = 24.0
	speed   = 240.0 // logical pixels per simulated second
)

// ball is a trivial deterministic physics body. prev/cur are kept so the
// renderer can interpolate between two fixed simulation moments.
type ball struct {
	prevX, curX float64
	prevY, curY float64
	vx, vy      float64
}

func (b *ball) step(dt float64) {
	b.prevX, b.prevY = b.curX, b.curY
	b.curX += b.vx * dt
	b.curY += b.vy * dt

	// Bounce off the logical window edges.
	if b.curX < radius {
		b.curX = radius
		b.vx = -b.vx
	}
	if b.curX > windowW-radius {
		b.curX = windowW - radius
		b.vx = -b.vx
	}
	if b.curY < radius {
		b.curY = radius
		b.vy = -b.vy
	}
	if b.curY > windowH-radius {
		b.curY = windowH - radius
		b.vy = -b.vy
	}
}

func main() {
	app := gogpu.NewApp(gogpu.DefaultConfig().
		WithTitle("GoGPU - Fixed Timestep").
		WithSize(windowW, windowH).
		WithContinuousRender(true).
		// 60Hz deterministic simulation, at most 5 catch-up steps per frame.
		WithFixedTimestep(time.Second/60, 5))

	b := &ball{
		curX: windowW / 2, curY: windowH / 2,
		prevX: windowW / 2, prevY: windowH / 2,
		vx: speed, vy: speed * 0.7,
	}

	// OnUpdate always receives exactly 1/60s, called zero or more times per
	// rendered frame depending on accumulated wall time.
	app.OnUpdate(func(dt float64) {
		b.step(dt)
	})

	// OnDraw renders the interpolated state. alpha is 0 at the previous
	// simulation step and 1 at the latest one, so motion stays smooth even
	// when display frames do not line up with simulation ticks.
	app.OnDraw(func(dc *gogpu.Context) {
		alpha := dc.InterpolationAlpha()
		x := float32(b.prevX + (b.curX-b.prevX)*alpha)
		y := float32(b.prevY + (b.curY-b.prevY)*alpha)
		_ = x
		_ = y

		dc.Clear(0.05, 0.06, 0.09, 1.0)
		// DrawTriangleColor keeps the example dependency-free; a real app
		// would issue a circle/fill call at the interpolated (x, y).
		if err := dc.DrawTriangleColor(gmath.CornflowerBlue); err != nil {
			log.Printf("draw failed: %v", err)
		}
	})

	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}
