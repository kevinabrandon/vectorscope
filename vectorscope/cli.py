#!/usr/bin/env python3
"""Unified CLI for vectorscope - Draw on your oscilloscope!"""

import argparse

from .base import add_common_args, common_args_from_parsed
from .config import load_config, save_config, apply_config_defaults, CONFIG_FILE, SAVEABLE_KEYS


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

    from .hershey_player import HERSHEY_FONT_NAMES
    hershey_names = ', '.join(sorted(HERSHEY_FONT_NAMES))

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
    text_parser.add_argument("--font", default="futural",
                             help=f"Font name (Hershey: {hershey_names}; or any matplotlib font family)")
    text_parser.add_argument("--curve-pts", type=int, default=30,
                             help="Points per curve segment")
    text_parser.add_argument("--penlift", type=int, default=20,
                             help="Blanked samples between contours (0=no pen lifts)")
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
    clock_parser.add_argument("--font", default="futural",
                              help=f"Font name (Hershey: {hershey_names}; or any matplotlib font family)")
    clock_parser.add_argument("--penlift", type=int, default=20,
                              help="Blanked samples between strokes (0=no pen lifts)")
    clock_parser.add_argument("--curve-pts", type=int, default=30,
                              help="Points per curve segment (outline fonts only)")
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

    # Z calibration subcommand
    zcal_parser = subparsers.add_parser(
        'zcal',
        help='Z-channel calibration patterns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    zcal_parser.add_argument("--mode", type=str, default="delay",
                              choices=["delay", "intensity", "blanking"],
                              help="Calibration mode")
    add_common_args(zcal_parser, freq_default=100)
    subs['zcal'] = zcal_parser

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
    interactive_parser.add_argument("--channels", type=int, default=2,
                                    help="Output channels (2=XY, 4=XY+Z+spare)")
    subs['interactive'] = interactive_parser

    # Config subcommand
    config_parser = subparsers.add_parser(
        'config',
        help='View or modify persistent configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config_subs = config_parser.add_subparsers(dest='config_action')
    config_subs.add_parser('path', help='Print config file path')
    config_subs.add_parser('reset', help='Delete config file')
    save_parser = config_subs.add_parser(
        'save', help='Save values to config file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    save_parser.add_argument("--device", type=str, default=None,
                             help="Audio output device")
    save_parser.add_argument("--rate", type=int, default=None,
                             help="Sample rate in Hz")
    save_parser.add_argument("--channels", type=int, default=None,
                             help="Output channels")
    save_parser.add_argument("--z-delay", type=float, default=None,
                             help="Z delay compensation in microseconds")
    save_parser.add_argument("--z-amp", type=float, default=None,
                             help="Z output amplitude (0-1)")
    save_parser.add_argument("--z-gamma", type=float, default=None,
                             help="Z intensity gamma correction")

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
            font=args.font,
            penlift=args.penlift,
            curve_pts=args.curve_pts,
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

    elif args.command == 'zcal':
        if args.channels < 3:
            print("Error: zcal requires --channels >= 3 (e.g. --channels 4)")
            return None
        from .zcal import ZCalPlayer
        return ZCalPlayer(
            mode=args.mode,
            **common_args_from_parsed(args)
        )

    return None


def _handle_config(args):
    """Handle the 'config' subcommand."""
    import json
    import os

    if args.config_action == 'path':
        print(CONFIG_FILE)
    elif args.config_action == 'reset':
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
            print(f"Removed {CONFIG_FILE}")
        else:
            print("No config file found.")
    elif args.config_action == 'save':
        updates = {}
        for key in SAVEABLE_KEYS:
            val = getattr(args, key, None)
            if val is not None:
                updates[key] = val
        if not updates:
            print("Nothing to save. Specify at least one option, e.g. --channels 4")
            return
        save_config(updates)
        print(f"Saved to {CONFIG_FILE}:")
        for k, v in sorted(updates.items()):
            print(f"  {k}: {v}")
    else:
        # No subcommand: print current config
        cfg = load_config()
        if not cfg:
            if os.path.exists(CONFIG_FILE):
                print("Config file is empty.")
            else:
                print("No config file found.")
            return
        print(f"{CONFIG_FILE}:")
        print(json.dumps(cfg, indent=2))


def main():
    """Main CLI entry point with subcommands."""
    parser, subparsers = _build_parser()

    config = load_config()
    apply_config_defaults(parser, config, subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'config':
        _handle_config(args)
        return

    # Print loaded config values so the user can see what's active
    if config:
        parts = [f"{k}={v}" for k, v in sorted(config.items())]
        print(f"config: {', '.join(parts)}")

    if args.command == 'interactive':
        from .interactive import InteractiveSession
        session = InteractiveSession(parser, subparsers, args)
        session.run()
        return

    if args.command == 'zcal':
        from .zcal import ZCalSession
        session = ZCalSession(
            sample_rate=args.rate,
            freq=args.freq,
            amp=args.amp,
            device=args.device,
            channels=args.channels,
            z_amp=args.z_amp,
            z_delay=args.z_delay,
            z_blank=args.z_blank,
            z_gamma=args.z_gamma,
        )
        session.run()
        return

    player = _create_player(args)
    if player:
        player.run()


if __name__ == "__main__":
    main()
