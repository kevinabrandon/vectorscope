# Asteroids (Vector Renderer)

This directory contains a headless adaptation of an Asteroids game engine for
vector display rendering.

## Attribution
- Original Asteroids code: Copyright (C) 2008 Nick Redshaw
- License: GNU General Public License v3.0 or later (GPL-3.0+)
- The source files in this directory include the original GPL headers.

## Modifications
The game engine is based on Nick Redshaw's open-source Python Asteroids (Copyright 2008, GPL-3.0+).
Dave Plummer's vector display project](https://github.com/davepl/vector) adapted the original code for headless/vector rendering and
integration with the HP 1345A vector display pipeline.  From there the following modifications were
made:
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

## License
This code remains under GPL-3.0+. A copy of the license is provided at:

```
LICENSES/GPL-3.0.txt
```
