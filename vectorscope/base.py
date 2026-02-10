"""Base class for real-time oscilloscope XY display."""

import argparse
import numpy as np
import sounddevice as sd

NOISE_TYPES = [
    'white', 'perlin', 'brownian', 'correlated', 'pink',
    'normal', 'sample-hold', 'burst', 'ring', 'harmonic',
    'all',
]

# Types that 'all' cycles through (everything except 'all' itself)
_CYCLE_TYPES = [t for t in NOISE_TYPES if t != 'all']


class VectorScopePlayer:
    """
    Base class for real-time XY oscilloscope audio streaming.

    Subclass this and implement your own visualization by either:
    - Setting self.xy_data in __init__ (for static patterns)
    - Overriding audio_callback() (for dynamic/animated patterns)
    """

    def __init__(self, sample_rate=48000, secs=0.05, amp=0.7, device=None,
                 noise=0.0, fade_period=10.0, noise_type='white',
                 noise_mode='sum'):
        self.sample_rate = sample_rate
        self.secs = secs
        self.amp = amp
        self.device = device
        self.noise = noise
        self.fade_period = fade_period
        self.samples = int(sample_rate * secs)
        self.xy_data = None
        self.position = 0
        self.global_sample = 0

        # Parse noise type(s): "name", "name:level", comma-separated, or "all"
        raw = noise_type or 'white'
        parsed = []
        for tok in raw.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if ':' in tok:
                name, lvl = tok.split(':', 1)
                parsed.append((name.strip(), float(lvl.strip())))
            else:
                parsed.append((tok, None))
        if not parsed:
            parsed = [('white', None)]

        if len(parsed) == 1 and parsed[0][0] == 'all':
            all_lvl = parsed[0][1]
            self._noise_list = [(t, all_lvl) for t in _CYCLE_TYPES]
            # 'all' defaults to cycle mode unless explicitly set to sum
            if noise_mode == 'sum':
                noise_mode = 'cycle'
        elif len(parsed) > 1 or parsed[0][1] is not None:
            names = [n for n, _ in parsed]
            bad = [n for n in names if n not in _CYCLE_TYPES]
            if bad:
                raise ValueError(f"Unknown noise type(s): {', '.join(bad)}")
            self._noise_list = parsed
        else:
            if parsed[0][0] not in NOISE_TYPES:
                raise ValueError(f"Unknown noise type: {parsed[0][0]}")
            self._noise_list = None
        self.noise_type = parsed[0][0]
        self._noise_mode = noise_mode

        # Cycle tracking
        self._last_noise_cycle = -1
        self._noise_cycle_index = 0

        self._init_noise_state()

    def _init_noise_state(self):
        """Initialize/reset state for stateful noise generators."""
        # Perlin/brownian: random value table for smooth interpolation
        self._noise_table_size = 4096
        self._noise_table = np.random.uniform(
            -1, 1, (self._noise_table_size, 2)
        ).astype(np.float32)

        # Sample-and-hold: current held value and countdown
        self._sample_hold_value = np.zeros(2, dtype=np.float32)
        self._sample_hold_counter = 0

        # Burst: remaining samples in current burst
        self._burst_remaining = 0

    # ------------------------------------------------------------------
    # Noise generators
    # ------------------------------------------------------------------

    def _generate_noise(self, frames, t, noise_type=None):
        """Dispatch to the selected noise generator."""
        nt = noise_type or self.noise_type
        if nt == 'white':
            return self._white_noise(frames)
        elif nt == 'perlin':
            return self._perlin_noise(frames)
        elif nt == 'brownian':
            return self._brownian_noise(frames)
        elif nt == 'correlated':
            return self._correlated_noise(frames)
        elif nt == 'pink':
            return self._pink_noise(frames)
        elif nt == 'sample-hold':
            return self._sample_hold_noise(frames)
        elif nt == 'burst':
            return self._burst_noise(frames)
        elif nt == 'ring':
            return self._ring_noise(frames, t)
        elif nt == 'harmonic':
            return self._harmonic_noise(frames, t)
        return self._white_noise(frames)

    def _white_noise(self, frames):
        """Uniform random noise — classic TV static."""
        return np.random.uniform(-1, 1, (frames, 2)).astype(np.float32)

    def _perlin_noise(self, frames):
        """Smooth, organic noise using value noise with multiple octaves (fBm).

        Samples noise along the trace position so different parts of the
        shape get different displacement, then slowly evolves over time.
        """
        # Position within the trace cycle (0→1, repeating)
        samples = np.arange(self.global_sample, self.global_sample + frames)
        trace_phase = (samples % self.samples).astype(np.float32) / self.samples
        # Slow time drift so the deformation evolves
        time_offset = self.global_sample / self.sample_rate * 0.3

        result = np.zeros((frames, 2), dtype=np.float32)
        num_octaves = 4
        total_amp = 0.0

        for octave in range(num_octaves):
            spatial_freq = 4.0 * (2 ** octave)  # cycles along the trace
            amp = 1.0 / (1.5 ** octave)
            total_amp += amp
            # Each octave drifts at a slightly different rate
            phase = trace_phase * spatial_freq + time_offset * (octave + 1) * 0.37
            idx = phase.astype(np.int64) % self._noise_table_size
            frac = (phase - np.floor(phase)).astype(np.float32)
            # Smoothstep interpolation
            frac = frac * frac * (3 - 2 * frac)
            frac = frac[:, np.newaxis]
            v0 = self._noise_table[idx]
            v1 = self._noise_table[(idx + 1) % self._noise_table_size]
            result += amp * (v0 * (1 - frac) + v1 * frac)

        result /= total_amp
        return result

    def _brownian_noise(self, frames):
        """Slow drifting deformation -- like the shape is made of jelly.

        Low-frequency noise along the trace that evolves very slowly,
        so different parts of the shape wander independently.
        """
        samples = np.arange(self.global_sample, self.global_sample + frames)
        trace_phase = (samples % self.samples).astype(np.float32) / self.samples
        time_offset = self.global_sample / self.sample_rate * 0.08  # very slow

        # Single octave, low spatial frequency → large smooth deformations
        phase = trace_phase * 2.0 + time_offset
        idx = phase.astype(np.int64) % self._noise_table_size
        frac = (phase - np.floor(phase)).astype(np.float32)
        frac = frac * frac * (3 - 2 * frac)
        frac = frac[:, np.newaxis]
        v0 = self._noise_table[idx]
        v1 = self._noise_table[(idx + 1) % self._noise_table_size]
        return (v0 * (1 - frac) + v1 * frac)

    def _correlated_noise(self, frames):
        """XY-correlated noise — scatters points in circles instead of squares.

        Uses polar coordinates so the noise halo is round, not boxy.
        """
        r = np.sqrt(np.random.uniform(0, 1, frames)).astype(np.float32)
        theta = np.random.uniform(0, 2 * np.pi, frames).astype(np.float32)
        noise = np.empty((frames, 2), dtype=np.float32)
        noise[:, 0] = r * np.cos(theta)
        noise[:, 1] = r * np.sin(theta)
        return noise

    def _pink_noise(self, frames):
        """1/f noise — emphasizes lower frequencies for a natural, flowing feel.

        Generated by shaping white noise in the frequency domain.
        """
        if frames < 4:
            return self._white_noise(frames)
        white = np.random.normal(0, 1, (frames, 2)).astype(np.float32)
        freqs = np.fft.rfftfreq(frames, d=1.0 / self.sample_rate)
        freqs[0] = 1.0  # avoid div-by-zero
        pink_filter = (1.0 / np.sqrt(freqs))[:, np.newaxis].astype(np.float32)
        pink_filter[0] = 0.0  # kill DC
        spectrum = np.fft.rfft(white, axis=0)
        result = np.fft.irfft(spectrum * pink_filter, n=frames, axis=0).astype(np.float32)
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result /= max_val
        return result

    def _sample_hold_noise(self, frames):
        """Stepped noise — holds random values then jumps.

        Creates a blocky, digital-glitch aesthetic where the shape snaps
        between distorted positions.
        """
        hold_samples = max(1, int(self.sample_rate * 0.005))  # 5 ms hold
        noise = np.empty((frames, 2), dtype=np.float32)
        pos = 0
        counter = self._sample_hold_counter
        value = self._sample_hold_value.copy()

        while pos < frames:
            if counter <= 0:
                value = np.random.uniform(-1, 1, 2).astype(np.float32)
                counter = hold_samples
            chunk = min(counter, frames - pos)
            noise[pos:pos + chunk] = value
            pos += chunk
            counter -= chunk

        self._sample_hold_counter = counter
        self._sample_hold_value = value.copy()
        return noise

    def _burst_noise(self, frames):
        """Intermittent bursts of heavy static interference.

        Clean most of the time, then briefly explodes into noise — like
        actual radio interference.
        """
        noise = np.zeros((frames, 2), dtype=np.float32)
        burst_samples = int(self.sample_rate * 0.03)  # 30 ms bursts

        # ~2.5 bursts/sec, normalized to frame size so rate is consistent
        burst_prob = 1.0 - (1.0 - 2.5 / self.sample_rate) ** frames

        if self._burst_remaining > 0:
            n = min(self._burst_remaining, frames)
            noise[:n] = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
            self._burst_remaining -= n
        elif np.random.random() < burst_prob:
            start = np.random.randint(0, max(1, frames))
            n = min(burst_samples, frames - start)
            noise[start:start + n] = np.random.uniform(
                -1, 1, (n, 2)
            ).astype(np.float32)
            self._burst_remaining = max(0, burst_samples - n)

        return noise

    def _ring_noise(self, frames, t):
        """Ring modulation — creates intensity variations and gaps along the trace.

        Uses slowly varying FM-synthesized modulators so the pattern
        evolves over time.
        """
        modulator = np.empty((frames, 2), dtype=np.float32)
        modulator[:, 0] = np.sin(
            2 * np.pi * 47.0 * t + np.sin(2 * np.pi * 0.10 * t) * 5
        )
        modulator[:, 1] = np.sin(
            2 * np.pi * 61.0 * t + np.sin(2 * np.pi * 0.13 * t) * 5
        )
        return modulator

    def _harmonic_noise(self, frames, t):
        """Irrational-ratio sine waves — slowly evolving shimmer.

        Adds sine waves at golden-ratio / e / pi multiples of the trace
        frequency so they never quite sync up, creating Lissajous-like
        interference that drifts endlessly.
        """
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        base = 1.0 / self.secs
        freqs = [base * phi, base * np.sqrt(2), base * np.e, base * np.pi]
        amps_list = [0.4, 0.3, 0.2, 0.1]

        noise = np.zeros((frames, 2), dtype=np.float32)
        for i, (f, a) in enumerate(zip(freqs, amps_list)):
            phase_off = i * 0.7
            noise[:, 0] += a * np.sin(2 * np.pi * f * t + phase_off)
            noise[:, 1] += a * np.cos(2 * np.pi * f * 1.07 * t + phase_off)
        return noise

    # ------------------------------------------------------------------
    # Noise mixing
    # ------------------------------------------------------------------

    # Noise types that always use additive mixing (displacement-style)
    _ADDITIVE_TYPES = frozenset(('perlin', 'brownian', 'harmonic', 'burst'))

    def _apply_one_noise(self, outdata, frames, t, noise_level, noise_type,
                         force_additive=False):
        """Apply a single noise type to the output buffer.

        When force_additive is True (used in sum mode), crossfade types
        are applied additively so stacking multiple doesn't attenuate
        the signal.
        """
        if noise_type == 'normal':
            self._apply_normal_noise(outdata, noise_level)
        else:
            noise_signal = self._generate_noise(frames, t, noise_type)

            if noise_type == 'ring':
                # Multiplicative: blend between clean and ring-modulated
                outdata[:] *= (1 - noise_level) + noise_level * noise_signal
            elif noise_type in self._ADDITIVE_TYPES or force_additive:
                # Additive: displace/layer on top of the shape
                outdata[:] += noise_signal * self.amp * noise_level
            else:
                # Standard crossfade between signal and noise
                outdata[:] = (outdata * (1 - noise_level)
                              + noise_signal * self.amp * noise_level)

    def _noise_level_for(self, lfo, level):
        """Compute per-sample noise level array from LFO and a noise amount."""
        return (lfo * level)[:, np.newaxis]

    def _apply_noise(self, outdata, frames):
        """Mix noise into the output buffer with LFO fade modulation."""
        if not self._noise_list and self.noise <= 0:
            return

        t = np.arange(self.global_sample, self.global_sample + frames) / self.sample_rate
        lfo = 0.5 - 0.5 * np.cos(2 * np.pi * t / self.fade_period)

        if self._noise_list and self._noise_mode == 'sum':
            # Layer all noise types together (additive so they don't
            # attenuate the signal when stacking)
            for nt, lvl in self._noise_list:
                effective = lvl if lvl is not None else self.noise
                if effective <= 0:
                    continue
                nl = self._noise_level_for(lfo, effective)
                self._apply_one_noise(outdata, frames, t, nl, nt,
                                      force_additive=True)
        elif self._noise_list and self._noise_mode == 'cycle':
            # Rotate through noise types each fade period
            t_start = self.global_sample / self.sample_rate
            current_cycle = int(t_start / self.fade_period)
            if current_cycle != self._last_noise_cycle:
                self._last_noise_cycle = current_cycle
                self._noise_cycle_index = current_cycle % len(self._noise_list)
                self._init_noise_state()
                name, lvl = self._noise_list[self._noise_cycle_index]
                label = f"{name}:{lvl}" if lvl is not None else name
                print(f"  noise: {label}")
            nt, lvl = self._noise_list[self._noise_cycle_index]
            effective = lvl if lvl is not None else self.noise
            if effective > 0:
                nl = self._noise_level_for(lfo, effective)
                self._apply_one_noise(outdata, frames, t, nl, nt)
        else:
            # Single noise type
            if self.noise <= 0:
                return
            nl = self._noise_level_for(lfo, self.noise)
            self._apply_one_noise(outdata, frames, t, nl, self.noise_type)

        np.clip(outdata, -1.0, 1.0, out=outdata)

    def _apply_normal_noise(self, outdata, noise_level):
        """Displace points along the path normal for fuzzy / glowing edges.

        Computes the tangent of the XY path, then pushes each point
        perpendicular to it by a random amount.
        """
        dx = np.diff(outdata[:, 0], prepend=outdata[0, 0])
        dy = np.diff(outdata[:, 1], prepend=outdata[0, 1])

        length = np.sqrt(dx * dx + dy * dy)
        length = np.maximum(length, 1e-8)
        # Normal is (-dy, dx)
        nx = -dy / length
        ny = dx / length

        displacement = np.random.normal(0, 1, len(outdata)).astype(np.float32)
        displacement *= noise_level.squeeze() * self.amp

        outdata[:, 0] += nx * displacement
        outdata[:, 1] += ny * displacement

    # ------------------------------------------------------------------
    # Buffer / callback / run
    # ------------------------------------------------------------------

    def _fill_buffer(self, outdata, frames):
        """Fill output buffer by looping through xy_data."""
        if self.xy_data is None:
            outdata.fill(0)
            return

        data_len = len(self.xy_data)
        out_idx = 0

        while out_idx < frames:
            chunk_size = min(frames - out_idx, data_len - self.position)
            outdata[out_idx:out_idx + chunk_size] = self.xy_data[self.position:self.position + chunk_size]
            self.position = (self.position + chunk_size) % data_len
            out_idx += chunk_size

    def audio_callback(self, outdata, frames, time, status):
        """Sounddevice callback. Override for custom behavior."""
        if status:
            print(f"Audio status: {status}")

        self._fill_buffer(outdata, frames)
        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        """Called when stream starts. Override for custom message."""
        print("Playing. Press Ctrl+C to stop.")

    def _on_stop(self):
        """Called when stream stops. Override for custom message."""
        print("\nStopped.")

    def run(self):
        """Start the audio stream."""
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype='float32',
            callback=self.audio_callback,
            device=self.device
        ):
            self._on_start()
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                self._on_stop()


def add_common_args(parser, secs_default=0.05):
    """Add common arguments to an argument parser."""
    parser.add_argument("--rate", type=int, default=48000,
                        help="Sample rate in Hz")
    parser.add_argument("--secs", type=float, default=secs_default,
                        help="Duration of one trace cycle")
    parser.add_argument("--amp", type=float, default=0.7,
                        help="Output amplitude (0-1)")
    parser.add_argument("--device", type=str, default=None,
                        help="Audio output device")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level (0-1)")
    parser.add_argument("--fade-period", type=float, default=10.0,
                        help="Noise fade cycle in seconds")
    parser.add_argument("--noise-type", type=str, default="white",
                        help="Noise algorithm (%(default)s). Options: "
                             + ", ".join(NOISE_TYPES)
                             + ". Comma-separate to combine, e.g. "
                               "perlin:0.4,normal:0.05 (level per type, "
                               "defaults to --noise)")
    parser.add_argument("--noise-mode", type=str, default="sum",
                        choices=["sum", "cycle"],
                        help="How to combine multiple noise types: "
                             "sum layers them together, cycle rotates each fade period")


def common_args_from_parsed(args):
    """Extract common arguments as a dict for passing to VectorScopePlayer."""
    return {
        'sample_rate': args.rate,
        'secs': args.secs,
        'amp': args.amp,
        'device': args.device,
        'noise': args.noise,
        'fade_period': args.fade_period,
        'noise_type': args.noise_type,
        'noise_mode': args.noise_mode,
    }
