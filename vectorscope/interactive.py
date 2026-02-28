"""Interactive REPL for vectorscope — switch commands and tweak params live."""

import readline  # noqa: F401 — enables up-arrow history for input()
import shlex
import numpy as np
import sounddevice as sd

from .cli import _create_player

# Params that are stream-level or internal — not changeable mid-session.
_HIDDEN_PARAMS = frozenset({
    'rate', 'device', 'command', 'out', 'interactive',
})


class InteractiveSession:
    """Owns the audio stream and delegates to the current player."""

    def __init__(self, parser, subparsers, args):
        self._parser = parser
        self._subparsers = subparsers  # dict: command name -> subparser
        self._sample_rate = args.rate
        self._device = args.device
        self._command_names = sorted(self._subparsers.keys())
        self.current_player = None
        self.current_args = None
        self._current_command = None

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def audio_callback(self, outdata, frames, time, status):
        player = self.current_player
        if player is not None:
            player.audio_callback(outdata, frames, time, status)
        else:
            outdata.fill(0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=2,
            dtype='float32',
            callback=self.audio_callback,
            device=self._device,
            latency='high',
        )
        stream.start()

        print("Vectorscope Interactive Mode")
        print(f"Commands: {', '.join(self._command_names)}")
        print("Type a command to start, 'help' for info, Ctrl+C to exit.")

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
            print("\nStopped.")

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def _handle_input(self, line):
        tokens = shlex.split(line)
        first = tokens[0]

        if first == 'help':
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

        # Drop to silence while building the new player — player creation
        # can be CPU-heavy (text rendering, fractals) and would starve
        # the audio callback thread via GIL contention.
        self.current_player = None

        player = _create_player(args)
        if player is None:
            # e.g. text --out produces a file, not a player
            return

        self.current_player = player
        self.current_args = args
        self._current_command = cmd
        player._on_start()
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

        # Build action type lookup from the current subparser
        action_types = self._get_action_types()

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
            try:
                new_value = self._coerce(current_value, value_str,
                                         declared_type)
            except ValueError as e:
                print(f"Invalid value for {key}: {e}")
                continue

            setattr(self.current_args, key, new_value)
            changed.append(key)

        if not changed:
            return

        # Force stream-level params
        self.current_args.rate = self._sample_rate
        self.current_args.device = self._device

        # Silence during rebuild (see _switch_command comment)
        self.current_player = None

        player = _create_player(self.current_args)
        if player is None:
            return
        self.current_player = player
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

    @staticmethod
    def _coerce(current_value, value_str, declared_type=None):
        """Convert value_str to the appropriate type.

        Uses declared_type (from argparse action) when available,
        falling back to inference from current_value's type.
        """
        if value_str.lower() == 'none':
            return None

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
    # Display
    # ------------------------------------------------------------------

    def _print_params(self, only=None):
        """Print parameter table for the current command."""
        if self.current_args is None or self._current_command is None:
            return

        sub = self._subparsers.get(self._current_command)
        if sub is None:
            return

        # Build help text lookup from the subparser's actions
        help_map = {}
        for action in sub._actions:
            if action.dest in _HIDDEN_PARAMS:
                continue
            if isinstance(action, (type(sub._subparsers_action)
                                   if hasattr(sub, '_subparsers_action')
                                   else type(None))):
                continue
            help_map[action.dest] = action.help or ''

        for dest, help_text in help_map.items():
            if not hasattr(self.current_args, dest):
                continue
            if only is not None and dest not in only:
                continue
            value = getattr(self.current_args, dest)
            print(f"  {dest:<20s}{str(value):<12s}{help_text}")

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
