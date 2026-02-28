"""
Vectorscope - Draw on your oscilloscope!

A collection of tools for displaying graphics on an oscilloscope in XY mode.
Connect your audio L/R to scope CH1/CH2, set to XY mode, and have fun!

Basic usage:
    $ vectorscope circle
    $ vectorscope text "Hello"
    $ vectorscope spiral --arms 4
    $ vectorscope clock --24h
    $ vectorscope fractal dragon

Or use as a library:
    from vectorscope import CirclePlayer
    player = CirclePlayer(freq=100)
    player.run()
"""

__version__ = "0.1.0"
__author__ = "Kevin"

from .base import VectorScopePlayer, add_common_args, common_args_from_parsed
from .circle import CirclePlayer
from .text import TextPlayer, build_xy_from_text
from .spiral import SpiralPlayer
from .clock import ClockPlayer
from .ngon import NgonPlayer
from .fractal import FractalPlayer
from .platonic import PlatonicPlayer

__all__ = [
    # Base
    "VectorScopePlayer",
    "add_common_args",
    "common_args_from_parsed",
    # Players
    "CirclePlayer",
    "TextPlayer",
    "SpiralPlayer",
    "ClockPlayer",
    "NgonPlayer",
    "FractalPlayer",
    "PlatonicPlayer",
    # Utilities
    "build_xy_from_text",
    # Meta
    "__version__",
]
