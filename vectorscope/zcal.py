"""Z-channel calibration patterns for oscilloscope intensity modulation."""

import readline  # noqa: F401 — enables up-arrow history for input()

import numpy as np
import sounddevice as sd

from .base import VectorScopePlayer


class ZCalPlayer(VectorScopePlayer):
    """Calibration patterns for Z-channel (intensity) alignment.

    Modes:
        delay:     Circle with half bright / half dark. Adjust --z-delay
                   until the bright/dark boundary sits at 12 o'clock.
        intensity: Circle with Z ramping 0->1 around the circumference.
                   Verifies polarity and range.
        blanking:  Square with large pen-lift gaps. Z blanks the gaps.
                   Verifies blanking works.
    """

    def __init__(self, mode='delay', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        if mode == 'blanking':
            self._build_blanking_pattern()

    def _build_blanking_pattern(self):
        """Square (4-gon) with large pen-lift gaps between edges."""
        # 4 edges with substantial gaps between them
        corners = np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1]
        ], dtype=np.float32) * 0.7

        edge_samples = self.samples // 8  # half the cycle is edges
        gap_samples = self.samples // 8   # half the cycle is gaps

        parts = []
        blanking_parts = []
        for i in range(4):
            # Edge: interpolate between corners
            a = corners[i]
            b = corners[(i + 1) % 4]
            t = np.linspace(0, 1, edge_samples, dtype=np.float32)
            edge = a * (1 - t[:, np.newaxis]) + b * t[:, np.newaxis]
            parts.append(edge)
            blanking_parts.append(np.zeros(edge_samples, dtype=bool))

            # Gap: hold at endpoint (blanked)
            gap = np.tile(b, (gap_samples, 1))
            parts.append(gap)
            blanking_parts.append(np.ones(gap_samples, dtype=bool))

        xy = np.vstack(parts).astype(np.float32)
        blanking = np.concatenate(blanking_parts)

        # Pad or trim to exact samples
        if len(xy) < self.samples:
            pad_n = self.samples - len(xy)
            xy = np.vstack([xy, np.tile(xy[-1], (pad_n, 1))])
            blanking = np.concatenate([blanking, np.zeros(pad_n, dtype=bool)])
        else:
            xy = xy[:self.samples]
            blanking = blanking[:self.samples]

        self.xy_data = xy * self.amp
        self.xy_blanking = blanking

    def audio_callback(self, outdata, frames, time, status):
        self._check_status(status)

        if self.mode == 'delay':
            self._delay_callback(outdata, frames)
        elif self.mode == 'intensity':
            self._intensity_callback(outdata, frames)
        else:
            # blanking mode uses xy_data + xy_blanking via _fill_buffer
            self._fill_buffer(outdata, frames)
            self._apply_z_channel(outdata, frames)
            self.global_sample += frames
            return

        self.global_sample += frames

    def _delay_callback(self, outdata, frames):
        """Circle with half bright / half dark for delay calibration."""
        phase = self._compute_trace_phase(frames)
        theta = 2 * np.pi * (phase % 1.0)
        outdata[:, 0] = (np.cos(theta) * self.amp).astype(np.float32)
        outdata[:, 1] = (np.sin(theta) * self.amp).astype(np.float32)

        if self.z_enabled:
            # Top half (0 to pi) = bright, bottom half (pi to 2pi) = dark
            # Boundary at 12 o'clock when delay is correct
            intensity = np.where(theta <= np.pi, 1.0, 0.0).astype(np.float32)
            self.z_intensity = intensity
            self._z_blanking = None
            self._apply_z_channel(outdata, frames)

    def _intensity_callback(self, outdata, frames):
        """Circle with Z ramping 0->1 around circumference."""
        phase = self._compute_trace_phase(frames)
        theta = 2 * np.pi * (phase % 1.0)
        outdata[:, 0] = (np.cos(theta) * self.amp).astype(np.float32)
        outdata[:, 1] = (np.sin(theta) * self.amp).astype(np.float32)

        if self.z_enabled:
            # Linear ramp 0->1 around the circle
            intensity = (theta / (2 * np.pi)).astype(np.float32)
            self.z_intensity = intensity
            self._z_blanking = None
            self._apply_z_channel(outdata, frames)

    def _on_start(self):
        print(f"Z calibration: {self.mode} mode")
        print(f"  z-delay: {self.z_delay} µs ({self._z_delay_samples} samples at {self.sample_rate} Hz)")
        print()
        if self.mode == 'delay':
            print("  HOW TO USE")
            print("  ----------")
            print("  A circle is drawn on XY. The Z channel makes the top half")
            print("  bright and the bottom half dark. If your audio interface")
            print("  has a delay on channels 3/4 relative to 1/2, the bright/dark")
            print("  boundary will appear rotated away from 12 o'clock.")
            print()
            print("  1. Run:  vectorscope zcal --channels 4")
            print("  2. Look at where the bright/dark boundary falls on the circle.")
            print("     If it's at 12 o'clock, delay is zero -- you're done.")
            print("  3. If the boundary is rotated clockwise, try positive --z-delay.")
            print("     If counter-clockwise, try negative --z-delay.")
            print("     Example:  vectorscope zcal --channels 4 --z-delay 100")
            print("          or:  vectorscope zcal --channels 4 --z-delay -100")
            print("     (value is in microseconds; one sample at 48kHz ~ 20.8 µs)")
            print("  4. Keep adjusting until the boundary sits at 12 o'clock.")
            print("  5. Use that --z-delay value for all other commands.")
        elif self.mode == 'intensity':
            print("  HOW TO USE")
            print("  ----------")
            print("  A circle is drawn on XY. The Z channel ramps intensity")
            print("  linearly from 0 (blanked) to 1 (full bright) around the")
            print("  circumference, starting and ending at 12 o'clock.")
            print()
            print("  1. Run:  vectorscope zcal --channels 4 --mode intensity")
            print("  2. You should see the circle fade from invisible at 12 o'clock")
            print("     to full brightness just before 12 o'clock (clockwise).")
            print("  3. If it's inverted (bright at top, dark at bottom), your scope's")
            print("     Z polarity is opposite -- you may need an inverting amp.")
            print("  4. Adjust --z-amp to control the intensity range:")
            print("     vectorscope zcal --channels 4 --mode intensity --z-amp 0.5")
        elif self.mode == 'blanking':
            print("  HOW TO USE")
            print("  ----------")
            print("  A square is drawn with large gaps between each edge.")
            print("  The Z channel blanks (turns off the beam) during the gaps.")
            print()
            print("  1. Run:  vectorscope zcal --channels 4 --mode blanking")
            print("  2. You should see four separate line segments (square edges)")
            print("     with no visible trace between them.")
            print("  3. If you see faint lines connecting the edges, increase --z-amp:")
            print("     vectorscope zcal --channels 4 --mode blanking --z-amp 1.0")
            print("     or use an external amplifier to boost the Z voltage.")
            print("  4. If the edges themselves are invisible, your Z polarity may")
            print("     be inverted -- try the intensity mode to diagnose.")
        print()
        print("  Press Ctrl+C to stop.")

    def _on_stop(self):
        """Silent stop — ZCalSession handles save prompts."""
        pass


class ZCalSession:
    """Guided interactive Z-channel calibration wizard.

    Walks through delay, intensity, and blanking calibration stages
    in a single session with a persistent audio stream.
    """

    def __init__(self, sample_rate=48000, freq=100, amp=0.7, device=None,
                 channels=4, z_amp=1.0, z_delay=0.0, z_blank=True,
                 z_gamma=1.0):
        self._sample_rate = sample_rate
        self._freq = freq
        self._amp = amp
        self._device = device
        self._channels = channels
        self._z_amp = z_amp
        self._z_delay = z_delay
        self._z_blank = z_blank
        self._z_gamma = z_gamma
        self._player = None
        self._saved_state = self._snapshot()

    def _snapshot(self):
        """Capture current calibration state for change tracking."""
        return {
            'z_delay': self._z_delay,
            'z_amp': self._z_amp,
            'z_gamma': self._z_gamma,
            'channels': self._channels,
            'rate': self._sample_rate,
        }

    def _has_changes(self):
        """True if current state differs from last saved state."""
        return self._snapshot() != self._saved_state

    def _make_player(self, mode):
        """Create a ZCalPlayer for the given mode with current settings."""
        return ZCalPlayer(
            mode=mode,
            sample_rate=self._sample_rate,
            freq=self._freq,
            amp=self._amp,
            device=self._device,
            channels=self._channels,
            z_amp=self._z_amp,
            z_delay=self._z_delay,
            z_blank=self._z_blank,
            z_gamma=self._z_gamma,
        )

    def _swap_player(self, mode):
        """Atomically swap to a new player for the given mode."""
        self._player = None  # silence while building
        player = self._make_player(mode)
        self._player = player
        return player

    def audio_callback(self, outdata, frames, time, status):
        player = self._player
        if player is not None:
            player.audio_callback(outdata, frames, time, status)
        else:
            outdata.fill(0)

    def _offer_save(self):
        """Prompt to save if settings changed since last save."""
        if not self._has_changes():
            return
        from .config import save_config, CONFIG_FILE
        print()
        print("Current settings:")
        print(f"  z-delay:  {self._z_delay} µs")
        print(f"  z-amp:    {self._z_amp}")
        print(f"  z-gamma:  {self._z_gamma}")
        print(f"  channels: {self._channels}")
        print(f"  rate:     {self._sample_rate}")
        try:
            answer = input(f"Save to {CONFIG_FILE}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
            print()
        if answer == 'y':
            save_config(self._snapshot())
            self._saved_state = self._snapshot()
            print(f"Saved to {CONFIG_FILE}")

    def _print_stage_header(self, number, title):
        print()
        print(f"{'=' * 50}")
        print(f"  STAGE {number}: {title}")
        print(f"{'=' * 50}")

    def _print_delay_instructions(self):
        print()
        print("  A circle is drawn on XY. The Z channel makes the top half")
        print("  bright and the bottom half dark. Adjust z_delay until the")
        print("  bright/dark boundary sits at 12 o'clock.")
        print()
        print("  If the boundary is rotated clockwise, try positive values.")
        print("  If counter-clockwise, try negative values.")
        print(f"  (one sample at {self._sample_rate} Hz ~ "
              f"{1e6 / self._sample_rate:.1f} µs)")
        print()
        print(f"  Current z_delay: {self._z_delay} µs")
        print()
        print("  Type a number to set z_delay, Enter/done to advance, Ctrl+C to exit.")

    def _print_intensity_instructions(self):
        print()
        print("  A circle is drawn on XY. The Z channel ramps intensity")
        print("  from 0 (blanked) to 1 (full bright) around the circumference,")
        print("  starting at 12 o'clock (clockwise).")
        print()
        print("  Adjust z_amp to control the intensity range.")
        print("  Adjust z_gamma so the brightness ramp looks perceptually even.")
        print("  (gamma=1.0 is linear, >1 darkens midtones, <1 brightens midtones)")
        print()
        print(f"  Current z_amp:   {self._z_amp}")
        print(f"  Current z_gamma: {self._z_gamma}")
        print()
        print("  Type a number to set z_amp, or z_gamma=N.")
        print("  Enter/done to advance, Ctrl+C to exit.")

    def _print_blanking_instructions(self):
        print()
        print("  A square is drawn with large gaps between each edge.")
        print("  The Z channel blanks (turns off the beam) during the gaps.")
        print()
        print("  You should see four separate line segments with no visible")
        print("  trace between them. If you see faint connecting lines,")
        print("  increase z_amp. You can also fine-tune z_delay here.")
        print()
        print(f"  Current z_amp:   {self._z_amp}")
        print(f"  Current z_delay: {self._z_delay} µs")
        print()
        print("  Type a number to set z_amp, or z_delay=N / z_amp=N.")
        print("  Enter/done to finish, Ctrl+C to exit.")

    def _run_delay_stage(self):
        """Stage 1: Delay calibration. Returns False if user wants to exit."""
        self._print_stage_header(1, "DELAY")
        self._swap_player('delay')
        self._print_delay_instructions()

        while True:
            try:
                line = input("\n  z_delay> ").strip()
            except (EOFError, KeyboardInterrupt):
                return False

            if not line or line.lower() == 'done':
                return True

            try:
                self._z_delay = float(line)
                self._swap_player('delay')
                z_samples = round(self._z_delay * self._sample_rate / 1e6)
                print(f"  z_delay = {self._z_delay} µs ({z_samples} samples)")
            except ValueError:
                print(f"  Invalid number: {line}")

    def _run_intensity_stage(self):
        """Stage 2: Intensity + gamma calibration. Returns False if user wants to exit."""
        self._print_stage_header(2, "INTENSITY")
        self._swap_player('intensity')
        self._print_intensity_instructions()

        while True:
            try:
                line = input("\n  z_amp> ").strip()
            except (EOFError, KeyboardInterrupt):
                return False

            if not line or line.lower() == 'done':
                return True

            # key=value form
            if '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                if key == 'z_gamma':
                    try:
                        self._z_gamma = float(val)
                        self._swap_player('intensity')
                        print(f"  z_gamma = {self._z_gamma}")
                    except ValueError:
                        print(f"  Invalid number: {val}")
                elif key == 'z_amp':
                    try:
                        self._z_amp = float(val)
                        self._swap_player('intensity')
                        print(f"  z_amp = {self._z_amp}")
                    except ValueError:
                        print(f"  Invalid number: {val}")
                else:
                    print(f"  Unknown param: {key} (use z_amp or z_gamma)")
                continue

            # Bare number → z_amp
            try:
                self._z_amp = float(line)
                self._swap_player('intensity')
                print(f"  z_amp = {self._z_amp}")
            except ValueError:
                print(f"  Unknown input: {line}")

    def _run_blanking_stage(self):
        """Stage 3: Blanking calibration. Returns False if user wants to exit."""
        self._print_stage_header(3, "BLANKING")
        self._swap_player('blanking')
        self._print_blanking_instructions()

        while True:
            try:
                line = input("\n  zcal> ").strip()
            except (EOFError, KeyboardInterrupt):
                return False

            if not line or line.lower() == 'done':
                return True

            # key=value form
            if '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                if key == 'z_delay':
                    try:
                        self._z_delay = float(val)
                        self._swap_player('blanking')
                        z_samples = round(self._z_delay * self._sample_rate / 1e6)
                        print(f"  z_delay = {self._z_delay} µs ({z_samples} samples)")
                    except ValueError:
                        print(f"  Invalid number: {val}")
                elif key == 'z_amp':
                    try:
                        self._z_amp = float(val)
                        self._swap_player('blanking')
                        print(f"  z_amp = {self._z_amp}")
                    except ValueError:
                        print(f"  Invalid number: {val}")
                elif key == 'z_gamma':
                    try:
                        self._z_gamma = float(val)
                        self._swap_player('blanking')
                        print(f"  z_gamma = {self._z_gamma}")
                    except ValueError:
                        print(f"  Invalid number: {val}")
                else:
                    print(f"  Unknown param: {key} (use z_delay, z_amp, or z_gamma)")
                continue

            # Bare number → z_amp
            try:
                self._z_amp = float(line)
                self._swap_player('blanking')
                print(f"  z_amp = {self._z_amp}")
            except ValueError:
                print(f"  Unknown input: {line}")

    def run(self):
        """Run the full guided calibration session."""
        from .config import load_config, CONFIG_FILE

        # Show current config
        config = load_config()
        if config:
            print(f"\nLoaded from {CONFIG_FILE}:")
            for k, v in sorted(config.items()):
                print(f"  {k}: {v}")
        print(f"\nSession settings:")
        print(f"  channels:  {self._channels}")
        print(f"  rate:      {self._sample_rate}")
        print(f"  z_delay:   {self._z_delay} µs")
        print(f"  z_amp:     {self._z_amp}")
        print(f"  z_gamma:   {self._z_gamma}")

        if self._channels < 3:
            print("\nError: zcal requires channels >= 3 (e.g. --channels 4)")
            return

        # Open persistent stream
        stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype='float32',
            callback=self.audio_callback,
            device=self._device,
            latency='high',
        )
        stream.start()

        try:
            if not self._run_delay_stage():
                return
            self._offer_save()

            if not self._run_intensity_stage():
                return
            self._offer_save()

            if not self._run_blanking_stage():
                return
            self._offer_save()

            print("\nCalibration complete!")
        except KeyboardInterrupt:
            pass
        finally:
            self._player = None
            stream.stop()
            stream.close()
            # Final save offer if anything changed
            self._offer_save()
            print()
            print("Final settings:")
            print(f"  z_delay:  {self._z_delay} µs")
            print(f"  z_amp:    {self._z_amp}")
            print(f"  z_gamma:  {self._z_gamma}")
            print(f"  channels: {self._channels}")
            print(f"  rate:     {self._sample_rate}")
