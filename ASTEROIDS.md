# Asteroids

A vector-rendered Asteroids game designed for oscilloscope display via audio output.

## Credits

The game engine is based on Nick Redshaw's open-source Python Asteroids
(Copyright 2008, GPL-3.0+), found via
[Dave Plummer's vector display project](https://github.com/davepl/vector).
The original code provided the core sprite system, collision detection, rock
splitting, saucer AI, and ship physics.

### Modifications

- Stripped the Pygame renderer and rewrote output as audio-rate XY polylines for oscilloscope display
- Added dynamic refresh and draw-order optimization for flicker reduction
- Arrow key support (ANSI escape sequence handling in raw terminal mode)
- Difficulty system (Easy / Medium / Hard) with per-preset lives, rock count, rock speed, saucer aggression, and friendly fire
- Configurable bullet parameters for both ship and saucers via CLI args
- Safe spawning -- ship respawns at the point farthest from all threats; rocks avoid spawning on top of the player on level-up
- Hyperspace teleports to safest position instead of random
- Saucer fire-rate cooldown (prevents bullet spam with multiple bullets)
- Orphaned bullet tracking -- saucer bullets stay lethal after saucer death and can still break rocks
- Post-death bullet collisions -- in-flight bullets continue scoring after ship destruction
- Continue system -- insert another "coin" at game over to keep your score
- In-game help screen and attract mode UI

## Quick Start

```bash
vectorscope asteroids                    # demo mode
vectorscope asteroids --difficulty easy   # start directly in easy
```

## Controls

| Key | Action |
|-----|--------|
| W / Up Arrow | Thrust |
| A / Left Arrow | Rotate left |
| D / Right Arrow | Rotate right |
| S / Down Arrow | Fire |
| Space | Fire |
| H | Hyperspace |
| 1 / 2 / 3 | Start Easy / Medium / Hard |
| 0 | Demo mode |
| C | Continue (game over screen) |
| ? | Help screen |
| Ctrl+C | Quit |

## Difficulty Presets

| | Easy | Medium | Hard |
|---|---|---|---|
| Lives | 5 | 4 | 3 |
| Starting Rocks | 1 | 3 | 4 |
| Rock Speed | 0.8x | 1.0x | 1.2x |
| Friendly Fire | Off | Off | On |
| Saucer Bullet Speed | 5 | 10 | 15 |
| Saucer Max Bullets | 1 | 2 | 3 |

## Scoring

| Target | Points |
|--------|--------|
| Large Asteroid | 50 |
| Medium Asteroid | 100 |
| Small Asteroid | 200 |
| Large Saucer | 500 |
| Small Saucer | 1000 |
| Extra Life | Every 10,000 pts |

## Gameplay Features

- **Safe Spawning** -- Ship respawns at the position farthest from all threats. New-level rocks avoid spawning near the ship.
- **Hyperspace** -- Teleports the ship to the safest open area on the screen.
- **Friendly Fire** (Hard mode) -- Your own bullets can loop around the screen and destroy you.
- **Orphaned Bullets** -- Saucer bullets remain active after the saucer is destroyed and can still hit you or break rocks.
- **Post-Death Scoring** -- Bullets in flight when you die can still destroy rocks and saucers for points.
- **Continue** -- Press C on the game over screen to keep your score and continue with a fresh set of lives.

## CLI Options

```
--difficulty {easy,medium,hard}   Game difficulty preset
--rocks N                         Initial number of large asteroids
--friendly-fire                   Your own bullets can kill you
--max-vectors N                   Max line segments per frame (0=unlimited)
--aspect RATIO                    Aspect ratio / X scale (default: 1.0)
--penlift N                       Blanked samples between vectors
--dynamic / --no-dynamic          Dynamic refresh rate (default: on)
--optimize / --no-optimize        Optimize draw order (default: on)
```

### Bullet Tuning

```
--ship-bullet-speed SPEED         Ship bullet speed (default: 26)
--ship-bullet-ttl FRAMES          Ship bullet lifetime (default: 105)
--ship-max-bullets N              Max ship bullets (default: 4)
--saucer-bullet-speed SPEED       Saucer bullet speed (default: 5)
--saucer-bullet-ttl FRAMES        Saucer bullet lifetime (default: 60/90)
--saucer-max-bullets N            Max saucer bullets (default: 1)
```

Explicit args override difficulty preset values.
