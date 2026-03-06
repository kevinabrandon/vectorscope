"""Base class for real-time oscilloscope XY display."""

import argparse
import ctypes
import threading
import time as _time

import numpy as np
import sounddevice as sd

# Suppress ALSA's C-level error messages (underrun warnings etc.)
# by installing a no-op error handler via ctypes.
try:
    _alsa_err_t = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_int,
                                   ctypes.c_char_p)
    _alsa_err_handler = _alsa_err_t(lambda *a: None)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(
        _alsa_err_handler)
except (OSError, AttributeError):
    pass  # not Linux / ALSA not available

NOISE_TYPES = [
    'white', 'perlin', 'brownian', 'correlated', 'pink',
    'normal', 'sample-hold', 'burst', 'ring', 'harmonic',
    'all',
]

# Types that 'all' cycles through (everything except 'all' itself)
_CYCLE_TYPES = [t for t in NOISE_TYPES if t != 'all']


class NullOutputStream:
    """A dummy OutputStream that discards data, used for 'demo' mode."""
    def __init__(self, samplerate, channels, dtype, callback, device=None, blocksize=0, latency=None):
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.blocksize = blocksize or 1024
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def close(self):
        self.stop()

    def sleep(self, ms):
        import time
        time.sleep(ms / 1000.0)

    class _NullStatus:
        """Mimics sounddevice CallbackFlags — falsy when no errors."""
        output_underflow = False
        output_overflow = False
        input_underflow = False
        input_overflow = False
        priming_output = False
        def __bool__(self):
            return False

    def _run(self):
        import time
        from collections import namedtuple
        status = self._NullStatus()
        StreamTime = namedtuple('StreamTime', ['currentTime', 'outputBufferDacTime'])
        frame_duration = self.blocksize / self.samplerate

        while self._running:
            start_time = time.monotonic()
            outdata = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            dummy_time = StreamTime(time.monotonic(), time.monotonic() + frame_duration)
            self.callback(outdata, self.blocksize, dummy_time, status)
            elapsed = time.monotonic() - start_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class VectorScopePlayer:
    """
    Base class for real-time XY oscilloscope audio streaming.

    Subclass this and implement your own visualization by either:
    - Setting self.xy_data in __init__ (for static patterns)
    - Overriding audio_callback() (for dynamic/animated patterns)
    """
    def __init__(self, sample_rate=48000, freq=100, amp=0.7, device=None,
                 noise=0.0, fade_period=10.0, noise_type='white',
                 noise_mode='sum', animate_freq_range=None, freq_sign=1,
                 channels=2, z_amp=1.0, z_delay=0.0, z_blank=True,
                 z_gamma=1.0, web_port=None, web_scale_factor=1.0,
                 perf_log_period=1.0):
        self.sample_rate = sample_rate
        self.freq = abs(freq)
        self.secs = 1.0 / self.freq
        self.amp = amp
        self.device = device
        self.noise = noise
        self.fade_period = fade_period
        self.animate_freq_range = animate_freq_range
        self.samples = int(sample_rate * self.secs)
        self.xy_data = None
        self.xy_blanking = None
        self.position = 0
        self.global_sample = 0
        self._stream_start_time = _time.monotonic()
        self._last_status_time = 0.0
        self._last_trace_phase = 0.0
        self._frac_position = 0.0
        self._web_port = web_port
        self._web_scale_factor = web_scale_factor
        self._perf_log_period = perf_log_period
        self._web_server = None
        self._command_name = None
        
        # Threading support
        self._lock = threading.Lock()

        # Performance statistics
        self.perf_log = open("vectorscope_perf.log", "a", encoding="utf-8")
        self.stats = {
            'compute_time': 0.0,
            'compute_count': 0,
            'refresh_count': 0, # New: track every audio cycle
            'callback_time': 0.0,
            'callback_count': 0,
            'wait_time': 0.0,
            'wait_count': 0,
            'vector_count': 0,
            'pen_lifts': 0,
            'samples_count': 0,
            'last_callback_end': None,
            'last_stats_print': _time.monotonic()
        }

        # Z-channel (intensity modulation)
        self.channels = channels
        self.z_enabled = channels >= 3
        self.z_amp = z_amp
        self.z_delay = z_delay  # microseconds
        self.z_blank = z_blank
        self.z_gamma = z_gamma
        self._z_blanking = None
        self.z_intensity = None
        self._z_delay_samples = round(z_delay * sample_rate / 1e6) if z_delay != 0 else 0
        if self._z_delay_samples > 0:
            # Positive: Z arrives late at scope → delay XY to compensate
            self._xy_delay_buf = np.zeros((self._z_delay_samples, 2), dtype=np.float32)
            self._z_delay_buf = np.zeros(0, dtype=np.float32)
        elif self._z_delay_samples < 0:
            # Negative: Z arrives early at scope → delay Z to compensate
            self._xy_delay_buf = np.zeros((0, 2), dtype=np.float32)
            self._z_delay_buf = np.zeros(-self._z_delay_samples, dtype=np.float32)
        else:
            self._xy_delay_buf = np.zeros((0, 2), dtype=np.float32)
            self._z_delay_buf = np.zeros(0, dtype=np.float32)
        self._pre_delay_xy = None
        self._pre_delay_z = None
        self._z_intensity_buf = None
        self._z_applied = False
        
        # Pre-calculated output buffers
        self._z_data = None
        self._web_data = None
        
        # Static metrics caching
        self._last_n_vectors = 0
        self._last_n_penlifts = 0
        self._last_n_samples = 0

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
        base = self.freq
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

        # Noise operates only on XY channels
        xy = outdata[:, :2]

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
                self._apply_one_noise(xy, frames, t, nl, nt,
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
                self._apply_one_noise(xy, frames, t, nl, nt)
        else:
            # Single noise type
            if self.noise <= 0:
                return
            nl = self._noise_level_for(lfo, self.noise)
            self._apply_one_noise(xy, frames, t, nl, self.noise_type)

        np.clip(xy, -1.0, 1.0, out=xy)

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
    # Z-channel (intensity modulation)
    # ------------------------------------------------------------------

    def _prepare_output(self, xy, blanking=None, intensities=None):
        """Pre-calculate Z-channel and Web data for the given XY samples.
        
        This method moves heavy math out of the high-pressure audio callback.
        """
        if not self.z_enabled:
            with self._lock:
                self.xy_data = xy.astype(np.float32)
                self.xy_blanking = blanking
                self._z_data = None
                if self._web_server:
                    self._web_data = xy.astype(np.float32)
            return

        samples = len(xy)
        intensity = np.ones(samples, dtype=np.float32)
        if intensities is not None:
            intensity = intensities.copy()
        
        if self.z_blank and blanking is not None:
            intensity[blanking] = 0.0
            
        # Gamma correction
        if self.z_gamma != 1.0:
            intensity **= self.z_gamma
            
        # Map to voltage (inverted: bright = negative voltage)
        z_signal = (1.0 - 2.0 * intensity) * self.z_amp
        z_signal = z_signal.astype(np.float32)

        # Prepare web data (using un-delayed signals)
        web_data = None
        if self._web_server:
            web_data = np.column_stack([xy, z_signal]).astype(np.float32)

        # Apply delay compensation for physical scope via persistent delay line
        if self._z_delay_samples > 0:
            # Positive: Z arrives late → delay XY (shift XY later)
            d = self._z_delay_samples
            full = np.vstack([self._xy_delay_buf, xy])
            self._xy_delay_buf = full[-d:].copy()
            xy = full[:samples]
        elif self._z_delay_samples < 0:
            # Negative: Z arrives early → delay Z (shift Z later)
            d = -self._z_delay_samples
            full = np.concatenate([self._z_delay_buf, z_signal])
            self._z_delay_buf = full[-d:].copy()
            z_signal = full[:samples]

        with self._lock:
            self.xy_data = xy.astype(np.float32)
            self.xy_blanking = blanking
            self._z_data = z_signal
            self._web_data = web_data
            
            # Cache for static reporting
            self._last_n_vectors = len(xy) // 10 if blanking is None else int((~blanking).sum() / 10)
            self._last_n_samples = len(xy)
            if blanking is not None:
                # Count transitions as a proxy for pen-lifts
                self._last_n_penlifts = int(np.sum(np.diff(blanking.astype(int)) > 0))
            else:
                self._last_n_penlifts = 0

    def _increment_compute_stats(self, duration, n_vectors, n_penlifts=0, n_samples=0):
        """Helper for players to record compute time and complexity."""
        if hasattr(self, 'stats'):
            self.stats['compute_time'] += duration
            self.stats['compute_count'] += 1
            self.stats['vector_count'] += n_vectors
            self.stats['pen_lifts'] += n_penlifts
            self.stats['samples_count'] += n_samples
            
            # Keep the cache updated for the next "idle" log entry
            self._last_n_vectors = n_vectors
            self._last_n_penlifts = n_penlifts
            self._last_n_samples = n_samples

    def _check_perf_log(self):
        """Periodically write performance stats to log file."""
        if not hasattr(self, 'stats'):
            return
            
        now = _time.monotonic()
        if now - self.stats['last_stats_print'] > self._perf_log_period:
            st = self.stats
            dur = now - st['last_stats_print']

            c_avg = (st['compute_time'] / st['compute_count'] * 1000) if st['compute_count'] > 0 else 0
            cb_avg = (st['callback_time'] / st['callback_count'] * 1000) if st['callback_count'] > 0 else 0
            w_avg = (st['wait_time'] / st['wait_count'] * 1000) if st['wait_count'] > 0 else 0
            v_avg = (st['vector_count'] / st['compute_count']) if st['compute_count'] > 0 else self._last_n_vectors
            p_avg = (st['pen_lifts'] / st['compute_count']) if st['compute_count'] > 0 else self._last_n_penlifts
            s_avg = (st['samples_count'] / st['compute_count']) if st['compute_count'] > 0 else self._last_n_samples

            lfps = st['compute_count'] / dur
            bfps = self.sample_rate / s_avg if s_avg > 0 else 0

            cmd = self._command_name or type(self).__name__
            ts = _time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = (f"[{ts}] {cmd:16} | dur: {dur:4.2f}s | lfps: {lfps:5.1f} | "
                        f"bfps: {bfps:5.1f} | vcts: {int(v_avg):4} | plfts: {int(p_avg):3} | "
                        f"smp: {int(s_avg):6} | cmp: {c_avg:5.2f}ms | cbk: {cb_avg:5.2f}ms | "
                        f"wait: {w_avg:5.2f}ms\n")
            self.perf_log.write(log_line)

            if 'status_messages' in st:
                for msg in st['status_messages']:
                    self.perf_log.write(f"{msg}\n")
                st['status_messages'] = []

            self.perf_log.flush()

            # Reset accumulators
            st['compute_time'] = st['callback_time'] = st['wait_time'] = st['vector_count'] = 0.0
            st['compute_count'] = st['refresh_count'] = st['callback_count'] = st['wait_count'] = 0
            st['pen_lifts'] = st['samples_count'] = 0
            st['last_stats_print'] = now
    # ------------------------------------------------------------------
    # Trace frequency animation
    # ------------------------------------------------------------------

    def _get_animated_freq(self, frames):
        """Return per-sample frequency array (animated) or scalar (static)."""
        if self.animate_freq_range is None:
            return self.freq
        t = (self.global_sample + np.arange(frames)) / self.sample_rate
        lfo = 0.5 - 0.5 * np.cos(2 * np.pi * t / self.fade_period)
        f_min, f_max = self.animate_freq_range
        return f_min + lfo * (f_max - f_min)

    def _compute_trace_phase(self, frames):
        """Return cumulative trace phase (cycle count, unwrapped).

        Uses phase integration when animated for smooth frequency changes,
        or direct freq * t when static.
        """
        if self.animate_freq_range is not None:
            freq = self._get_animated_freq(frames)
            phase_inc = freq / self.sample_rate
            phase = self._last_trace_phase + np.cumsum(phase_inc)
            self._last_trace_phase = phase[-1]
            return phase
        t = (self.global_sample + np.arange(frames)) / self.sample_rate
        return self.freq * t

    # ------------------------------------------------------------------
    # Buffer / callback / run
    # ------------------------------------------------------------------

    def _fill_buffer(self, outdata, frames):
        """Fill output buffer by looping through pre-calculated data."""
        if self.xy_data is None:
            outdata.fill(0)
            return

        data_len = len(self.xy_data)
        xy_out = outdata[:, :2]
        z_out = outdata[:, 2] if self.z_enabled else None

        # Determine indices for this block
        out_idx = 0
        self.position %= data_len

        while out_idx < frames:
            chunk_size = min(frames - out_idx, data_len - self.position)
            if chunk_size <= 0:
                self.position = 0
                continue

            xy_out[out_idx:out_idx + chunk_size] = self.xy_data[self.position:self.position + chunk_size]
            if z_out is not None and self._z_data is not None:
                z_out[out_idx:out_idx + chunk_size] = self._z_data[self.position:self.position + chunk_size]
            
            self.position = (self.position + chunk_size) % data_len
            out_idx += chunk_size

    def _check_status(self, status, log_func=None):
        """Log audio status, suppressing startup underruns and rate-limiting."""
        if not status:
            return
        now = _time.monotonic()
        # Suppress during the first 0.5s (ALSA device may still be settling)
        if now - self._stream_start_time < 0.5:
            return
        # Rate-limit to one message per second
        if now - self._last_status_time < 1.0:
            return
        self._last_status_time = now
        msg = f"Audio status: {status}"
        print(msg)
        if log_func:
            log_func(msg)

    def audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback. Minimal work mode."""
        t0 = _time.perf_counter()
        
        has_stats = hasattr(self, 'stats')
        if has_stats and self.stats['last_callback_end'] is not None:
            self.stats['wait_time'] += (t0 - self.stats['last_callback_end'])
            self.stats['wait_count'] += 1
        
        def log_status(msg):
            if has_stats:
                import time as _t
                ts = _t.strftime("%Y-%m-%d %H:%M:%S")
                self.stats.setdefault('status_messages', []).append(f"[{ts}] {msg}")

        self._check_status(status, log_func=log_status)

        with self._lock:
            self._fill_buffer(outdata, frames)
            
            if has_stats:
                self.stats['refresh_count'] += 1
            
        self._apply_noise(outdata, frames)
        
        # Zero spare channel
        if self.channels >= 4:
            outdata[:, 3] = 0.0
            
        self.global_sample += frames

        if has_stats:
            tend = _time.perf_counter()
            self.stats['callback_time'] += (tend - t0)
            self.stats['callback_count'] += 1
            self.stats['last_callback_end'] = tend

    def _on_start(self):
        """Called when stream starts. Override for custom message."""
        print("Playing. Press Ctrl+C to stop.")

    def _on_stop(self):
        """Called when stream stops. Override for custom message."""
        print("\nStopped.")

    def run(self):
        """Start the audio stream."""
        self._stream_start_time = _time.monotonic()
        self._last_status_time = 0.0

        # Start web server if requested
        if self._web_port is not None:
            from .web import VectorscopeWebServer
            self._web_server = VectorscopeWebServer(self._web_port)
            self._web_server.set_z_amp(self.z_amp)
            self._web_server.set_web_scale_factor(self._web_scale_factor)
            self._web_server.start()
            # Send initial metadata
            self._web_server.push_metadata({
                'command': self._command_name or type(self).__name__,
                'channels': self.channels,
            })

        # Use the standard optimized callback
        callback = self.audio_callback

        if self.device == 'demo':
            stream = NullOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
            )
        else:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
                device=self.device,
                latency='high',
            )
        stream.start()
        self._on_start()
        try:
            while True:
                # Push data to web server from main thread
                if self._web_server is not None:
                    web_data_to_push = None
                    with self._lock:
                        if self._web_data is not None:
                            web_data_to_push = self._web_data
                            self._web_data = None # Push once per calculation
                    
                    if web_data_to_push is not None:
                        self._web_server.push_frame(web_data_to_push)

                # Periodically log performance stats
                self._check_perf_log()

                if self.device == 'demo':
                    stream.sleep(10)
                else:
                    sd.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            self.perf_log.close()
            stream.stop()
            stream.close()
            if self._web_server:
                self._web_server.stop()
            self._on_stop()


def add_common_args(parser, freq_default=100, rate_default=48000):
    """Add common arguments to an argument parser.

    freq_default sets the default trace frequency for the command.
    rate_default sets the default sample rate for the command.
    """
    parser.add_argument("--rate", type=int, default=rate_default,
                        help="Sample rate in Hz")
    parser.add_argument("--freq", type=float, default=freq_default,
                        help="Trace frequency in Hz")
    parser.add_argument("--amp", type=float, default=0.7,
                        help="Output amplitude (0-1)")
    parser.add_argument("--device", type=str, default=None,
                        help="Audio output device")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level (0-1)")
    parser.add_argument("--fade-period", type=float, default=10.0,
                        help="Fade cycle in seconds (noise and animate-freq)")
    parser.add_argument("--animate-freq", type=float, nargs=2,
                        metavar=('FREQ_MIN', 'FREQ_MAX'), default=None,
                        help="Animate trace frequency between FREQ_MIN and "
                             "FREQ_MAX over fade-period")
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

    # Z-channel (intensity modulation)
    parser.add_argument("--channels", type=int, default=2,
                        help="Output channels (2=XY, 4=XY+Z+spare)")
    parser.add_argument("--z-amp", type=float, default=1.0,
                        help="Z output amplitude (0-1)")
    parser.add_argument("--z-delay", type=float, default=0.0,
                        help="Z delay compensation in microseconds (positive=Z earlier, negative=Z later)")
    parser.add_argument("--no-z-blank", dest="z_blank", action="store_false",
                        default=True,
                        help="Disable pen-lift blanking on Z (blanking on by default when channels>=3)")
    parser.add_argument("--z-gamma", type=float, default=1.0,
                        help="Z intensity gamma correction (1.0=linear, >1=darker midtones, <1=brighter midtones)")

    # Web viewer
    parser.add_argument("--web", action="store_true", default=False,
                        help="Enable web-based oscilloscope viewer")
    parser.add_argument("--web-port", type=int, default=8080,
                        help="Port for web viewer")
    parser.add_argument("--web-scale-factor", type=float, default=1.0,
                        help="Scale factor for the web viewer")
    parser.add_argument("--perf-log-period", type=float, default=1.0,
                        help="Period in seconds for writing performance stats to log")


def common_args_from_parsed(args):
    """Extract common arguments as a dict for passing to VectorScopePlayer."""
    return {
        'sample_rate': args.rate,
        'freq': args.freq,
        'amp': args.amp,
        'device': args.device,
        'noise': args.noise,
        'fade_period': args.fade_period,
        'noise_type': args.noise_type,
        'noise_mode': args.noise_mode,
        'animate_freq_range': args.animate_freq,
        'freq_sign': -1 if args.freq < 0 else 1,
        'channels': args.channels,
        'z_amp': args.z_amp,
        'z_delay': args.z_delay,
        'z_blank': args.z_blank,
        'z_gamma': args.z_gamma,
        'web_port': args.web_port if getattr(args, 'web', False) else None,
        'web_scale_factor': args.web_scale_factor,
        'perf_log_period': args.perf_log_period,
    }
