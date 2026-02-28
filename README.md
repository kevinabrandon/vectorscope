# Vectorscope

Draw on your oscilloscope with Python.

Connect your audio L/R to scope CH1/CH2, set to XY mode, and go.

```
Left channel  → X axis
Right channel → Y axis
```

---

## Install

```bash
pip install vectorscope
```

From source:

```bash
git clone https://github.com/kevin/vectorscope
cd vectorscope
pip install -e .
```

### Requirements

- Python 3.8+
- numpy
- sounddevice
- soundfile
- matplotlib
- HersheyFonts

---

## Quick Start

```bash
vectorscope circle
vectorscope ngon --sides 5 --rot-freq 0.5
vectorscope text "Hello"
vectorscope hershey "Hello"
vectorscope spiral
vectorscope clock
vectorscope fractal dragon
vectorscope platonic icosahedron
vectorscope spirograph --R 5 --r 3 --rot-freq 0.2
vectorscope interactive
```

Each command has `--help`:

```bash
vectorscope circle --help
vectorscope text --help
```

---

## Commands

### interactive

A live REPL that lets you switch between any command and tweak parameters while audio keeps playing. No need to restart between commands.

```bash
vectorscope interactive
vectorscope interactive --rate 44100
vectorscope interactive --device "USB Audio"
```

Inside the REPL:

```
> circle
> freq=200
> amp=0.3
> platonic icosahedron --rot-freq 0.3
> perspective=2
> ngon --sides 6 --rot-freq 0.5
> sides=8
> spiral
> help
> fractal dragon
> hershey "Hello" --font futural
```

Type a command name (with optional arguments) to switch. Change parameters with `key=value`. Type `help` to see available commands and current parameter values. Ctrl+C to exit.

| Option | Default | Description |
|--------|---------|-------------|
| `--rate` | 48000 | Sample rate (Hz) |
| `--device` | - | Audio output device |

---

### circle

A circle. Optionally sweeps frequency - useful for debugging oscilloscope ghosting (if the ghost offset doesn't change with frequency, it's a constant delay).

```bash
vectorscope circle
vectorscope circle --freq 200
vectorscope circle --freq -100              # counter-clockwise
vectorscope circle --freq-min 10 --freq-max 800 --sweep-rate 0.1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--freq` | 100 | Frequency in Hz (negative=CCW) |
| `--freq-min` | - | Min freq for sweep mode |
| `--freq-max` | - | Max freq for sweep mode |
| `--sweep-rate` | 0.1 | Sweep speed in Hz |

---

### ngon

Regular polygon. Triangle, square, pentagon, hexagon, and beyond (up to 1024 sides). Optional rotation.

```bash
vectorscope ngon                             # square (default)
vectorscope ngon --sides 3                   # triangle
vectorscope ngon --sides 6 --rot-freq 0.5    # spinning hexagon
vectorscope ngon --sides 5 --rot-freq -1     # pentagon, counter-clockwise
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sides` | 4 | Number of sides (3-1024) |
| `--freq` | 100 | Trace frequency in Hz |
| `--rot-freq` | 0 | Rotation frequency in Hz (0=static, negative=CCW) |

---

### text

Text display using matplotlib font rendering (filled/outline fonts).

```bash
vectorscope text "Hello World"
vectorscope text "Hi" --freq 200
vectorscope text "Hi" -o greeting.wav   # output to WAV file
```

| Option | Default | Description |
|--------|---------|-------------|
| `text` | Hello | Text to display |
| `-o FILE` | - | Output WAV instead of streaming |
| `--font` | DejaVu Sans | Font family |
| `--freq` | 100 | Trace frequency in Hz |
| `--curve-pts` | 30 | Points per curve |
| `--penlift` | 0 | Silence samples between strokes |

---

### hershey

Text display using single-stroke Hershey vector fonts. These fonts are purpose-built for plotters and oscilloscopes -- each letter is a single continuous stroke rather than a filled outline, producing clean, flicker-free results.

```bash
vectorscope hershey "Hello"
vectorscope hershey "Hello" --font futural
vectorscope hershey "Hello" --font scripts
vectorscope hershey "Test" --freq 80
```

| Option | Default | Description |
|--------|---------|-------------|
| `text` | Hershey | Text to display |
| `--font` | futural | Hershey font style |
| `--freq` | 100 | Trace frequency in Hz |
| `--penlift` | 10 | Silence samples between strokes |

---

### spiral

Hypnotic rotating spiral. You are getting sleepy...

```bash
vectorscope spiral
vectorscope spiral --arms 4 --rot-freq 1
vectorscope spiral --arms 6 --turns 8 --rot-freq -0.5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--arms` | 3 | Number of arms |
| `--turns` | 5 | Turns per arm |
| `--freq` | 100 | Trace frequency in Hz |
| `--rot-freq` | 0.5 | Rotation frequency in Hz (negative=CCW) |

---

### clock

Real-time digital clock. Updates each minute.

```bash
vectorscope clock
vectorscope clock --24h
vectorscope clock --freq 200
```

| Option | Default | Description |
|--------|---------|-------------|
| `--24h` | - | 24-hour format |
| `--freq` | 100 | Trace frequency in Hz |

---

### fractal

L-system fractals: `koch`, `dragon`, `sierpinski`, `hilbert`, `levy`

```bash
vectorscope fractal
vectorscope fractal dragon
vectorscope fractal hilbert -i 6
vectorscope fractal sierpinski --freq 30
```

| Option | Default | Description |
|--------|---------|-------------|
| `type` | koch | Fractal type |
| `-i` | varies | Iteration depth |
| `--freq` | 100 | Trace frequency in Hz |

---

### platonic

3D wireframe platonic solids with smooth tumbling rotation and perspective projection. All 5 solids: tetrahedron, cube, octahedron, dodecahedron, icosahedron.

```bash
vectorscope platonic                              # cube (default)
vectorscope platonic tetrahedron
vectorscope platonic icosahedron --rot-freq 0.3
vectorscope platonic dodecahedron --rx 0.1 --ry 0.2 --rz 0
vectorscope platonic cube --perspective 2         # strong perspective
vectorscope platonic cube --perspective 10        # nearly orthographic
```

| Option | Default | Description |
|--------|---------|-------------|
| `type` | cube | Solid type |
| `--rot-freq` | 0.15 | Base rotation frequency in Hz (negative=reverse) |
| `--rx` | - | X-axis rotation override |
| `--ry` | - | Y-axis rotation override |
| `--rz` | - | Z-axis rotation override |
| `--freq` | 100 | Trace frequency in Hz |
| `--perspective` | 3.0 | Camera distance (higher = flatter) |
| `--smooth` | 6 | Trace smoothing at vertices (0=sharp) |
| `--penlift` | 4 | Silence samples between edges |

---

### spirograph

Spirograph (hypotrochoid) patterns. A point on a small circle rolling inside a larger circle traces intricate looping curves.

```bash
vectorscope spirograph
vectorscope spirograph --R 7 --r 3
vectorscope spirograph --R 5 --r 3 --d 0.5
vectorscope spirograph --rot-freq 0.2
vectorscope spirograph --rot-freq -0.2
vectorscope spirograph --animate-d 0.2 1.5 --fade-period 8
```

| Option | Default | Description |
|--------|---------|-------------|
| `--R` | 5 | Fixed circle radius |
| `--r` | 3 | Moving circle radius |
| `--d` | 0.8 | Drawing point distance from moving circle center |
| `--freq` | 100 | Trace frequency in Hz |
| `--rot-freq` | 0 | Rotation frequency in Hz (0=static, negative=CCW) |
| `--animate-d` | - | Animate d between D_MIN and D_MAX over fade period |

---

## Common Options

All commands (except `interactive`) share these:

| Option | Default | Description |
|--------|---------|-------------|
| `--rate` | 48000 | Sample rate (Hz) |
| `--freq` | varies | Trace frequency in Hz (higher = brighter, lower = sharper) |
| `--amp` | 0.7 | Output amplitude (0-1) |
| `--device` | - | Audio output device |
| `--animate-freq` | - | Animate trace freq between FREQ_MIN and FREQ_MAX over fade-period |
| `--noise` | 0 | Noise level (0-1), default for types without explicit level |
| `--fade-period` | 10 | Fade cycle in seconds (noise and animate-freq) |
| `--noise-type` | white | Noise algorithm (see below) |
| `--noise-mode` | sum | How to combine multiple types: `sum` or `cycle` |

`--freq` controls how many times per second the shape is traced. Higher values give a brighter display but lower resolution; lower values give sharper detail but may flicker. Use `--animate-freq FREQ_MIN FREQ_MAX` to smoothly ramp the trace frequency between two values over `--fade-period` seconds.

### Noise

Add `--noise` for a fading static effect - like picking up a signal from deep space:

```bash
vectorscope clock --noise 0.5 --fade-period 8
vectorscope fractal sierpinski --noise 0.3
```

#### Noise Types

| Type | Effect |
|------|--------|
| `white` | Uniform random scatter - classic TV static |
| `perlin` | Smooth organic deformation - shapes breathe and morph |
| `brownian` | Slow drifting deformation - like the shape is made of jelly |
| `correlated` | Circular XY scatter - round halo instead of boxy fuzz |
| `pink` | 1/f noise - natural, flowing feel with emphasis on lower frequencies |
| `normal` | Fuzzy/glowing edges - displaces points perpendicular to the path |
| `sample-hold` | Stepped glitch - holds random offsets then jumps |
| `burst` | Intermittent static - clean signal with brief explosions of noise |
| `ring` | Ring modulation - creates intensity gaps and inversions along the trace |
| `harmonic` | Shimmer - irrational-ratio sine waves that drift endlessly |

```bash
vectorscope circle --noise 0.5 --noise-type perlin
vectorscope fractal koch --noise 0.3 --noise-type normal
vectorscope text "Hello" --noise 0.4 --noise-type harmonic
```

#### Combining Noise Types

Comma-separate types to layer them. Each type can have its own level with `:level`, otherwise it uses `--noise` as the default:

```bash
# Organic deformation + subtle fuzzy edges
vectorscope circle --noise-type perlin:0.4,normal:0.05

# Three types, each at its own intensity
vectorscope fractal koch --noise-type perlin:0.3,ring:0.2,normal:0.05

# Mix explicit levels with a default
vectorscope circle --noise 0.3 --noise-type perlin:0.4,normal
```

Use `--noise-mode cycle` to rotate through types instead of layering, switching each fade period:

```bash
# Cycle through all types
vectorscope circle --noise 0.5 --noise-type all --noise-mode cycle

# Cycle through a specific set
vectorscope circle --noise-type perlin:0.4,normal:0.05,ring:0.3 --noise-mode cycle
```

The current noise type is printed to the terminal when cycling.

---

## As a Library

```python
from vectorscope import CirclePlayer, NgonPlayer, TextPlayer, SpiralPlayer, PlatonicPlayer

player = CirclePlayer()
player.run()

player = NgonPlayer(sides=5, rot_freq=0.5)
player.run()

player = TextPlayer(text="Hi!")
player.run()

player = SpiralPlayer(arms=4, rot_freq=0.3)
player.run()

player = PlatonicPlayer(solid='icosahedron', rot_freq=0.2)
player.run()
```

Roll your own by subclassing `VectorScopePlayer`:

```python
from vectorscope import VectorScopePlayer
import numpy as np

class LissajousPlayer(VectorScopePlayer):
    def __init__(self, freq_x=3, freq_y=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_x = freq_x
        self.freq_y = freq_y

    def audio_callback(self, outdata, frames, time, status):
        t = (self.global_sample + np.arange(frames)) / self.sample_rate
        outdata[:, 0] = np.sin(2 * np.pi * self.freq_x * t) * self.amp
        outdata[:, 1] = np.sin(2 * np.pi * self.freq_y * t) * self.amp
        self.global_sample += frames

LissajousPlayer(freq_x=3, freq_y=2).run()
```

---

## Oscilloscope Tips

- **DC coupling** on both channels
- Same volts/div on CH1 and CH2
- `--freq` controls trace speed: higher = brighter but lower resolution; lower = sharper detail but may flicker
- Analog scopes look smoother but digital works fine

---

## License

MIT
