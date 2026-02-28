#!/usr/bin/env python3
"""Unified CLI for vectorscope - Draw on your oscilloscope!"""

import argparse

from .base import add_common_args, common_args_from_parsed


def _build_parser():
    """Build the argument parser with all subcommands.

    Returns (parser, subparsers_dict) where subparsers_dict maps command
    names to their subparser objects.
    """
    parser = argparse.ArgumentParser(
        prog='vectorscope',
        description='Draw on your oscilloscope! Connect audio L/R to scope CH1/CH2, set to XY mode.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available displays')
    subs = {}

    # Circle subcommand
    circle_parser = subparsers.add_parser(
        'circle',
        help='Circle with optional frequency sweep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    circle_parser.add_argument("--freq-min", type=float, default=None,
                               help="Min frequency for sweep mode")
    circle_parser.add_argument("--freq-max", type=float, default=None,
                               help="Max frequency for sweep mode")
    circle_parser.add_argument("--sweep-rate", type=float, default=0.1,
                               help="Sweep rate in Hz")
    add_common_args(circle_parser, freq_default=100)
    subs['circle'] = circle_parser

    # Text subcommand
    text_parser = subparsers.add_parser(
        'text',
        help='Text display',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    text_parser.add_argument("text", nargs="?", default="Hello",
                             help="Text to display")
    text_parser.add_argument("--font", default="DejaVu Sans",
                             help="Font family")
    text_parser.add_argument("--curve-pts", type=int, default=30,
                             help="Points per curve segment")
    text_parser.add_argument("--penlift", type=int, default=0,
                             help="Samples of (0,0) between contours")
    text_parser.add_argument("-o", "--out", default=None,
                             help="Output WAV file (generates file instead of streaming)")
    add_common_args(text_parser, freq_default=100)
    subs['text'] = text_parser

    # Spiral subcommand
    spiral_parser = subparsers.add_parser(
        'spiral',
        help='Hypnotic rotating spiral',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    spiral_parser.add_argument("--arms", type=int, default=3,
                               help="Number of spiral arms")
    spiral_parser.add_argument("--turns", type=float, default=5,
                               help="Number of spiral turns")
    spiral_parser.add_argument("--rot-freq", type=float, default=0.5,
                               help="Rotation frequency in Hz (negative=CCW)")
    add_common_args(spiral_parser, freq_default=100)
    subs['spiral'] = spiral_parser

    # Clock subcommand
    clock_parser = subparsers.add_parser(
        'clock',
        help='Digital clock display',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    clock_parser.add_argument("--24h", dest="use_24h", action="store_true",
                              help="Use 24-hour format")
    add_common_args(clock_parser, freq_default=100)
    subs['clock'] = clock_parser

    # N-gon subcommand
    ngon_parser = subparsers.add_parser(
        'ngon',
        help='Regular polygon (triangle, square, pentagon, ...)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ngon_parser.add_argument("--sides", type=int, default=4,
                             help="Number of sides (3=triangle, 4=square, ...)")
    ngon_parser.add_argument("--rot-freq", type=float, default=0.0,
                             help="Rotation frequency in Hz (0=static, negative=CCW)")
    add_common_args(ngon_parser, freq_default=100)
    subs['ngon'] = ngon_parser

    # Fractal subcommand
    fractal_parser = subparsers.add_parser(
        'fractal',
        help='Fractal patterns (koch, dragon, sierpinski, hilbert, levy)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    fractal_parser.add_argument("type", nargs="?", default="koch",
                                choices=['koch', 'dragon', 'sierpinski', 'hilbert', 'levy'],
                                help="Fractal type")
    fractal_parser.add_argument("-i", "--iterations", type=int, default=None,
                                help="Iteration depth (default varies by fractal)")
    add_common_args(fractal_parser, freq_default=100)
    subs['fractal'] = fractal_parser

    # Platonic solids subcommand
    from .platonic import SOLID_NAMES
    platonic_parser = subparsers.add_parser(
        'platonic',
        help='3D wireframe platonic solids',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    platonic_parser.add_argument("type", nargs="?", default="cube",
                                 choices=SOLID_NAMES,
                                 help="Solid type")
    platonic_parser.add_argument("--rot-freq", type=float, default=0.15,
                                 help="Base rotation frequency in Hz (negative=reverse)")
    platonic_parser.add_argument("--rx", type=float, default=None,
                                 help="X-axis rotation speed override")
    platonic_parser.add_argument("--ry", type=float, default=None,
                                 help="Y-axis rotation speed override")
    platonic_parser.add_argument("--rz", type=float, default=None,
                                 help="Z-axis rotation speed override")
    platonic_parser.add_argument("--perspective", type=float, default=3.0,
                                 help="Camera distance (higher = flatter)")
    platonic_parser.add_argument("--smooth", type=int, default=6,
                                 help="Trace smoothing at vertices (0=sharp)")
    platonic_parser.add_argument("--penlift", type=int, default=4,
                                 help="Silence samples between disconnected edges")
    add_common_args(platonic_parser, freq_default=100)
    subs['platonic'] = platonic_parser

    # Spirograph subcommand
    spirograph_parser = subparsers.add_parser(
        'spirograph',
        help='Spirograph patterns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    spirograph_parser.add_argument("--R", type=int, default=5,
                               help="Radius of the fixed circle")
    spirograph_parser.add_argument("--r", type=int, default=3,
                                 help="Radius of the moving circle")
    spirograph_parser.add_argument("--d", type=float, default=0.8,
                                   help="Distance of the drawing point from the moving circle's center")
    spirograph_parser.add_argument("--rot-freq", type=float, default=0.0,
                                 help="Rotation frequency in Hz (0=static, negative=CCW)")
    spirograph_parser.add_argument("--animate-d", type=float, nargs=2, metavar=('D_MIN', 'D_MAX'),
                               help="Animate 'd' between D_MIN and D_MAX over fade_period")
    add_common_args(spirograph_parser, freq_default=100)
    subs['spirograph'] = spirograph_parser

    # Hershey subcommand
    from HersheyFonts import HersheyFonts # Import here for local use
    hf_instance = HersheyFonts() # Instantiate to get default font names
    HERSHEY_FONT_CHOICES = list(hf_instance.default_font_names)

    hershey_parser = subparsers.add_parser(
        'hershey',
        help='Single-stroke Hershey font text rendering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    hershey_parser.add_argument("text", nargs="?", default="Hershey",
                             help="Text to display")
    hershey_parser.add_argument("--font", default="futural", choices=HERSHEY_FONT_CHOICES,
                             help="Hershey font style to use")
    hershey_parser.add_argument("--penlift", type=int, default=10,
                             help="Samples of (0,0) between strokes")
    add_common_args(hershey_parser, freq_default=100)
    subs['hershey'] = hershey_parser

    # Interactive subcommand
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Interactive REPL: switch commands and tweak params live',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    interactive_parser.add_argument("--rate", type=int, default=48000,
                                    help="Sample rate in Hz")
    interactive_parser.add_argument("--device", type=str, default=None,
                                    help="Audio output device")

    return parser, subs


def _create_player(args):
    """Create and return a player instance from parsed args.

    Returns the player, or None if the command produced output
    without needing a player (e.g. text --out).
    """
    if args.command == 'circle':
        from .circle import CirclePlayer
        return CirclePlayer(
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            sweep_rate=args.sweep_rate,
            **common_args_from_parsed(args)
        )

    elif args.command == 'text':
        if args.out:
            from .text import generate_wav
            generate_wav(
                text=args.text,
                output=args.out,
                rate=args.rate,
                freq=args.freq,
                amp=args.amp,
                font=args.font,
                curve_pts=args.curve_pts,
                penlift=args.penlift
            )
            return None
        from .text import TextPlayer
        return TextPlayer(
            text=args.text,
            font=args.font,
            curve_pts=args.curve_pts,
            pen_lift_samples=args.penlift,
            **common_args_from_parsed(args)
        )

    elif args.command == 'spiral':
        from .spiral import SpiralPlayer
        return SpiralPlayer(
            arms=args.arms,
            turns=args.turns,
            rot_freq=args.rot_freq,
            **common_args_from_parsed(args)
        )

    elif args.command == 'clock':
        from .clock import ClockPlayer
        return ClockPlayer(
            use_24h=args.use_24h,
            **common_args_from_parsed(args)
        )

    elif args.command == 'ngon':
        from .ngon import NgonPlayer
        return NgonPlayer(
            sides=args.sides,
            rot_freq=args.rot_freq,
            **common_args_from_parsed(args)
        )

    elif args.command == 'fractal':
        from .fractal import FractalPlayer
        return FractalPlayer(
            fractal_type=args.type,
            iterations=args.iterations,
            **common_args_from_parsed(args)
        )

    elif args.command == 'platonic':
        from .platonic import PlatonicPlayer
        return PlatonicPlayer(
            solid=args.type,
            rot_freq=args.rot_freq,
            rx=args.rx,
            ry=args.ry,
            rz=args.rz,
            perspective=args.perspective,
            corner=args.smooth,
            pen_lift=args.penlift,
            **common_args_from_parsed(args)
        )

    elif args.command == 'spirograph':
        from .spirograph import SpirographPlayer
        return SpirographPlayer(
            R=args.R,
            r=args.r,
            d=args.d,
            rot_freq=args.rot_freq,
            animate_d_range=args.animate_d,
            **common_args_from_parsed(args)
        )

    elif args.command == 'hershey':
        from .hershey_player import HersheyPlayer
        return HersheyPlayer(
            text=args.text,
            font=args.font,
            penlift=args.penlift,
            **common_args_from_parsed(args)
        )

    return None


def main():
    """Main CLI entry point with subcommands."""
    parser, subparsers = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'interactive':
        from .interactive import InteractiveSession
        session = InteractiveSession(parser, subparsers, args)
        session.run()
        return

    player = _create_player(args)
    if player:
        player.run()


if __name__ == "__main__":
    main()
