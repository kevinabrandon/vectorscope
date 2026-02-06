#!/usr/bin/env python3
"""Unified CLI for vectorscope - Draw on your oscilloscope!"""

import argparse

from .base import add_common_args, common_args_from_parsed


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog='vectorscope',
        description='Draw on your oscilloscope! Connect audio L/R to scope CH1/CH2, set to XY mode.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available displays')

    # Circle subcommand
    circle_parser = subparsers.add_parser(
        'circle',
        help='Circle with optional frequency sweep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    circle_parser.add_argument("--freq", type=float, default=100,
                               help="Circle frequency in Hz (constant mode)")
    circle_parser.add_argument("--freq-min", type=float, default=None,
                               help="Min frequency for sweep mode")
    circle_parser.add_argument("--freq-max", type=float, default=None,
                               help="Max frequency for sweep mode")
    circle_parser.add_argument("--sweep-rate", type=float, default=0.1,
                               help="Sweep rate in Hz")
    circle_parser.add_argument("--ccw", action="store_true",
                               help="Rotate counter-clockwise")
    add_common_args(circle_parser)

    # Text subcommand
    text_parser = subparsers.add_parser(
        'text',
        help='Text display (interactive: type to change)',
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
    text_parser.add_argument("-i", "--interactive", action="store_true",
                             help="Interactive mode: type to change text")
    text_parser.add_argument("-o", "--out", default=None,
                             help="Output WAV file (generates file instead of streaming)")
    add_common_args(text_parser, secs_default=0.01)

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
    spiral_parser.add_argument("--speed", type=float, default=0.5,
                               help="Rotation speed in Hz")
    spiral_parser.add_argument("--ccw", action="store_true",
                               help="Rotate counter-clockwise")
    add_common_args(spiral_parser)

    # Clock subcommand
    clock_parser = subparsers.add_parser(
        'clock',
        help='Digital clock display',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    clock_parser.add_argument("--24h", dest="use_24h", action="store_true",
                              help="Use 24-hour format")
    add_common_args(clock_parser, secs_default=0.01)

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
    add_common_args(fractal_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'circle':
        from .circle import CirclePlayer
        player = CirclePlayer(
            freq=args.freq,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            sweep_rate=args.sweep_rate,
            direction=-1 if args.ccw else 1,
            **common_args_from_parsed(args)
        )

    elif args.command == 'text':
        if args.out:
            from .text import generate_wav
            generate_wav(
                text=args.text,
                output=args.out,
                rate=args.rate,
                secs=args.secs,
                amp=args.amp,
                font=args.font,
                curve_pts=args.curve_pts,
                penlift=args.penlift
            )
            return
        from .text import TextPlayer
        player = TextPlayer(
            text=args.text,
            font=args.font,
            curve_pts=args.curve_pts,
            pen_lift_samples=args.penlift,
            interactive=args.interactive,
            **common_args_from_parsed(args)
        )

    elif args.command == 'spiral':
        from .spiral import SpiralPlayer
        player = SpiralPlayer(
            arms=args.arms,
            turns=args.turns,
            speed=args.speed,
            direction=-1 if args.ccw else 1,
            **common_args_from_parsed(args)
        )

    elif args.command == 'clock':
        from .clock import ClockPlayer
        player = ClockPlayer(
            use_24h=args.use_24h,
            **common_args_from_parsed(args)
        )

    elif args.command == 'fractal':
        from .fractal import FractalPlayer
        player = FractalPlayer(
            fractal_type=args.type,
            iterations=args.iterations,
            **common_args_from_parsed(args)
        )

    player.run()


if __name__ == "__main__":
    main()
