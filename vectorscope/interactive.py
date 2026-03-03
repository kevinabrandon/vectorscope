"""Interactive REPL for vectorscope — switch commands and tweak params live."""

import readline  # noqa: F401 — enables up-arrow history for input()
import shlex
import numpy as np
import sounddevice as sd

from .cli import _create_player
from .base import NullOutputStream

# Params that are stream-level or internal — not changeable mid-session.
_HIDDEN_PARAMS = frozenset({
    'rate', 'device', 'command', 'out', 'interactive',
    'channels',
})


class InteractiveSession:
    """Owns the audio stream and delegates to the current player."""

    def __init__(self, parser, subparsers, args, web_port=None, web_scale_factor=1.0):
        self._parser = parser
        self._subparsers = subparsers  # dict: command name -> subparser
        self._sample_rate = args.rate
        self._device = args.device
        self._channels = args.channels
        self._z_amp = args.z_amp
        self._command_names = sorted(self._subparsers.keys())
        self.current_player = None
        self.current_args = None
        self._current_command = None
        self._web_port = web_port
        self._web_scale_factor = web_scale_factor
        self._web_server = None

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def audio_callback(self, outdata, frames, time, status):
        player = self.current_player
        if player is not None:
            player._z_applied = False
            player.audio_callback(outdata, frames, time, status)
            # Auto-apply Z if the player's callback didn't
            if player.z_enabled and not player._z_applied:
                player._apply_z_channel(outdata, frames)
        else:
            outdata.fill(0)
        if self._web_server is not None:
            if player is not None:
                pre_xy = getattr(player, '_pre_delay_xy', None)
                xy = pre_xy if pre_xy is not None else outdata[:, :2]
                if player._z_applied:
                    pre_z = getattr(player, '_pre_delay_z', None)
                    if pre_z is not None:
                        web_data = np.column_stack([xy, pre_z])
                    else:
                        web_data = np.column_stack([xy, outdata[:, 2]])
                else:
                    web_data = xy.copy()
            else:
                web_data = outdata[:, :2].copy()
            self._web_server.push_frame(web_data)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        # Start web server if requested
        if self._web_port is not None:
            from .web import VectorscopeWebServer
            self._web_server = VectorscopeWebServer(self._web_port)
            self._web_server.set_z_amp(self._z_amp)
            self._web_server.set_web_scale_factor(self._web_scale_factor)
            self._web_server.start()

        if self._device == 'demo':
            stream = NullOutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype='float32',
                callback=self.audio_callback,
            )
        else:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype='float32',
                callback=self.audio_callback,
                device=self._device,
                latency='high',
            )
        stream.start()

        print("Vectorscope Interactive Mode")
        print(f"Commands: {', '.join(self._command_names)}")
        print("Type a command to start, 'help' for info, 'q' to exit.")

        try:
            while True:
                try:
                    line = input("\n> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                self._handle_input(line)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            if self._web_server:
                self._web_server.stop()
            print("\nStopped.")

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def _handle_input(self, line):
        tokens = shlex.split(line)
        first = tokens[0]

        if first in ('exit', 'quit', 'q'):
            raise KeyboardInterrupt
        elif first == 'help':
            self._show_help()
        elif first in self._command_names:
            self._switch_command(tokens)
        elif '=' in line:
            self._update_params(line)
        else:
            print(f"Unknown command: {first}")
            print(f"Available: {', '.join(self._command_names)}, help")

    # ------------------------------------------------------------------
    # Command switching
    # ------------------------------------------------------------------

    def _switch_command(self, tokens):
        cmd = tokens[0]
        # Build a full argv for argparse: command + remaining tokens
        try:
            args = self._parser.parse_args(tokens)
        except SystemExit:
            # argparse prints its own error; don't crash the REPL
            return

        # Force stream-level params to match our stream
        args.rate = self._sample_rate
        args.device = self._device
        args.channels = self._channels
        if not hasattr(args, 'z_blank'):
            args.z_blank = True

        # Drop to silence while building the new player — player creation
        # can be CPU-heavy (text rendering, fractals) and would starve
        # the audio callback thread via GIL contention.
        self.current_player = None

        try:
            player = _create_player(args)
        except Exception as e:
            print(f"Error: {e}")
            return
        if player is None:
            # e.g. text --out produces a file, not a player
            return

        self.current_player = player
        self.current_args = args
        self._current_command = cmd
        player._on_start()
        self._push_web_metadata()
        print()
        self._print_params()

    # ------------------------------------------------------------------
    # Param updates
    # ------------------------------------------------------------------

    def _update_params(self, line):
        if self.current_args is None:
            print("No active command. Type a command first.")
            return

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")
            return

        # Build action type, choices, and nargs lookups from the current subparser
        action_types = self._get_action_types()
        action_choices = self._get_action_choices()
        action_nargs = self._get_action_nargs()

        changed = []
        for token in tokens:
            if '=' not in token:
                print(f"Invalid param (expected key=value): {token}")
                continue
            key, value_str = token.split('=', 1)

            if key in _HIDDEN_PARAMS:
                print(f"Cannot change '{key}' in interactive mode.")
                continue

            if not hasattr(self.current_args, key):
                print(f"Unknown param: {key}")
                continue

            current_value = getattr(self.current_args, key)
            declared_type = action_types.get(key)
            nargs = action_nargs.get(key)
            try:
                new_value = self._coerce(current_value, value_str,
                                         declared_type, nargs=nargs)
            except ValueError as e:
                print(f"Invalid value for {key}: {e}")
                continue

            choices = action_choices.get(key)
            if choices is not None and new_value not in choices:
                print(f"Invalid value for {key}: choose from {', '.join(str(c) for c in choices)}")
                continue

            setattr(self.current_args, key, new_value)
            changed.append(key)

        if not changed:
            return

        # Force stream-level params
        self.current_args.rate = self._sample_rate
        self.current_args.device = self._device
        self.current_args.channels = self._channels

        # Silence during rebuild (see _switch_command comment)
        self.current_player = None

        try:
            player = _create_player(self.current_args)
        except Exception as e:
            print(f"Error: {e}")
            return
        if player is None:
            return
        self.current_player = player
        self._push_web_metadata()
        self._print_params(only=set(changed))

    # ------------------------------------------------------------------
    # Type coercion
    # ------------------------------------------------------------------

    def _get_action_types(self):
        """Return {dest: type_func} from the current subparser's actions."""
        sub = self._subparsers.get(self._current_command)
        if sub is None:
            return {}
        return {
            action.dest: action.type
            for action in sub._actions
            if action.type is not None
        }

    def _get_action_choices(self):
        """Return {dest: choices} from the current subparser's actions."""
        sub = self._subparsers.get(self._current_command)
        if sub is None:
            return {}
        return {
            action.dest: action.choices
            for action in sub._actions
            if action.choices is not None
        }

    def _get_action_nargs(self):
        """Return {dest: nargs} for actions with nargs > 1."""
        sub = self._subparsers.get(self._current_command)
        if sub is None:
            return {}
        return {
            action.dest: action.nargs
            for action in sub._actions
            if isinstance(action.nargs, int) and action.nargs > 1
        }

    @staticmethod
    def _coerce(current_value, value_str, declared_type=None, nargs=None):
        """Convert value_str to the appropriate type.

        Uses declared_type (from argparse action) when available,
        falling back to inference from current_value's type.
        For nargs > 1, splits on commas and coerces each element.
        """
        if value_str.lower() == 'none':
            return None

        # Multi-value params: split on commas, coerce each element
        if isinstance(nargs, int) and nargs > 1:
            parts = [s.strip() for s in value_str.split(',')]
            if len(parts) != nargs:
                raise ValueError(f"expected {nargs} comma-separated values, got {len(parts)}")
            if declared_type is not None:
                return [declared_type(p) for p in parts]
            return parts

        # If argparse declared a type, use it directly
        if declared_type is not None:
            return declared_type(value_str)

        if isinstance(current_value, bool):
            if value_str.lower() in ('true', '1', 'yes'):
                return True
            if value_str.lower() in ('false', '0', 'no'):
                return False
            raise ValueError(f"expected true/false, got '{value_str}'")

        if isinstance(current_value, int):
            return int(value_str)

        if isinstance(current_value, float):
            return float(value_str)

        if current_value is None:
            # Try numeric, fall back to string
            try:
                return int(value_str)
            except ValueError:
                pass
            try:
                return float(value_str)
            except ValueError:
                pass
            return value_str

        # Default: string
        return value_str

    # ------------------------------------------------------------------
    # Web metadata
    # ------------------------------------------------------------------

    def _push_web_metadata(self):
        """Send current command/params to web clients."""
        if self._web_server is None or self.current_args is None:
            return
        params = {}
        for key in vars(self.current_args):
            if key in _HIDDEN_PARAMS:
                continue
            val = getattr(self.current_args, key)
            if val is not None and not key.startswith('_'):
                # JSON-safe: convert lists and basic types
                if isinstance(val, (int, float, str, bool)):
                    params[key] = val
                elif isinstance(val, list):
                    params[key] = val
        self._web_server.push_metadata({
            'command': self._current_command,
            'channels': self._channels,
            'params': params,
        })

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_params(self, only=None):
        """Print parameter table for the current command."""
        if self.current_args is None or self._current_command is None:
            return

        sub = self._subparsers.get(self._current_command)
        if sub is None:
            return

        # Build help text, choices, and nargs lookups from the subparser's actions
        help_map = {}
        choices_map = {}
        nargs_map = {}
        for action in sub._actions:
            if action.dest in _HIDDEN_PARAMS:
                continue
            if isinstance(action, (type(sub._subparsers_action)
                                   if hasattr(sub, '_subparsers_action')
                                   else type(None))):
                continue
            help_map[action.dest] = action.help or ''
            if action.choices is not None:
                choices_map[action.dest] = action.choices
            if isinstance(action.nargs, int) and action.nargs > 1:
                nargs_map[action.dest] = action.metavar or (action.dest.upper(),) * action.nargs

        for dest, help_text in help_map.items():
            if not hasattr(self.current_args, dest):
                continue
            if only is not None and dest not in only:
                continue
            value = getattr(self.current_args, dest)
            # Format list values as comma-separated
            if isinstance(value, list):
                value_str = ','.join(str(v) for v in value)
            else:
                value_str = str(value)
            suffix = ''
            if dest in choices_map:
                suffix = f" ({', '.join(str(c) for c in choices_map[dest])})"
            elif dest in nargs_map:
                labels = nargs_map[dest]
                suffix = f" (set as {dest}={','.join(labels)})"
            print(f"  {dest:<20s}{value_str:<12s}{help_text}{suffix}")

    def _show_help(self):
        if self.current_args is None:
            print("Available commands:")
            for name in self._command_names:
                sub = self._subparsers[name]
                desc = sub.description or sub.format_usage().strip()
                # Use the help kwarg from add_parser if available
                help_text = ''
                for action in self._parser._subparsers._actions:
                    if hasattr(action, '_parser_class'):
                        choices = getattr(action, 'choices', {}) or {}
                        if name in choices:
                            help_text = getattr(action, '_choices_actions', [])
                            for ca in help_text:
                                if ca.dest == name:
                                    help_text = ca.help or ''
                                    break
                            else:
                                help_text = ''
                            break
                print(f"  {name:<16s}{help_text}")
            print("\nType a command name to start, e.g.: circle")
            print("Change params with key=value, e.g.: freq=200")
        else:
            print(f"Current command: {self._current_command}")
            print()
            self._print_params()
            print(f"\nAvailable commands: {', '.join(self._command_names)}")
            print("Change params with key=value, e.g.: freq=200")
