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

---

## Quick Start

```bash
vectorscope circle
vectorscope text "Hello"
vectorscope spiral
vectorscope clock
vectorscope fractal dragon
```

Each command has `--help`:

```bash
vectorscope circle --help
vectorscope text --help
```

---

## Commands

### circle

A circle. Optionally sweeps frequency - useful for debugging oscilloscope ghosting (if the ghost offset doesn't change with frequency, it's a constant delay).

```bash
vectorscope circle
vectorscope circle --freq 200
vectorscope circle --ccw
vectorscope circle --freq-min 10 --freq-max 800 --sweep-rate 0.1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--freq` | 100 | Frequency in Hz |
| `--freq-min` | - | Min freq for sweep mode |
| `--freq-max` | - | Max freq for sweep mode |
| `--sweep-rate` | 0.1 | Sweep speed in Hz |
| `--ccw` | - | Counter-clockwise |

---

### text

Text display. Interactive mode lets you type new text on the fly.

```bash
vectorscope text "Hello World"
vectorscope text -i                     # interactive - type to change
vectorscope text "Hi" -o greeting.wav   # output to WAV file
```

| Option | Default | Description |
|--------|---------|-------------|
| `text` | Hello | Text to display |
| `-i` | - | Interactive mode |
| `-o FILE` | - | Output WAV instead of streaming |
| `--font` | DejaVu Sans | Font family |
| `--curve-pts` | 30 | Points per curve |
| `--penlift` | 0 | Silence samples between strokes |

---

### spiral

Hypnotic rotating spiral. You are getting sleepy...

```bash
vectorscope spiral
vectorscope spiral --arms 4 --speed 1
vectorscope spiral --arms 6 --turns 8 --ccw
```

| Option | Default | Description |
|--------|---------|-------------|
| `--arms` | 3 | Number of arms |
| `--turns` | 5 | Turns per arm |
| `--speed` | 0.5 | Rotation speed (Hz) |
| `--ccw` | - | Counter-clockwise |

---

### clock

Real-time digital clock. Updates each minute.

```bash
vectorscope clock
vectorscope clock --24h
```

| Option | Default | Description |
|--------|---------|-------------|
| `--24h` | - | 24-hour format |

---

### fractal

L-system fractals: `koch`, `dragon`, `sierpinski`, `hilbert`, `levy`

```bash
vectorscope fractal
vectorscope fractal dragon
vectorscope fractal hilbert -i 6
```

| Option | Default | Description |
|--------|---------|-------------|
| `type` | koch | Fractal type |
| `-i` | varies | Iteration depth |

---

## Common Options

All commands share these:

| Option | Default | Description |
|--------|---------|-------------|
| `--rate` | 48000 | Sample rate (Hz) |
| `--secs` | varies | Trace cycle duration |
| `--amp` | 0.7 | Output amplitude (0-1) |
| `--device` | - | Audio output device |
| `--noise` | 0 | Noise level (0-1), default for types without explicit level |
| `--fade-period` | 10 | Noise fade cycle (seconds) |
| `--noise-type` | white | Noise algorithm (see below) |
| `--noise-mode` | sum | How to combine multiple types: `sum` or `cycle` |

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
from vectorscope import CirclePlayer, TextPlayer, SpiralPlayer

player = CirclePlayer(freq=100)
player.run()

player = TextPlayer(text="Hi!", interactive=True)
player.run()

player = SpiralPlayer(arms=4, speed=0.3)
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
- Digital scopes: use short `--secs` values (0.002-0.01) for stable display
- Analog scopes look smoother but digital works fine

---

## License

MIT
