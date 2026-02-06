"""Base class for real-time oscilloscope XY display."""

import argparse
import numpy as np
import sounddevice as sd


class VectorScopePlayer:
    """
    Base class for real-time XY oscilloscope audio streaming.

    Subclass this and implement your own visualization by either:
    - Setting self.xy_data in __init__ (for static patterns)
    - Overriding audio_callback() (for dynamic/animated patterns)
    """

    def __init__(self, sample_rate=48000, secs=0.05, amp=0.7, device=None,
                 noise=0.0, fade_period=10.0):
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

    def _apply_noise(self, outdata, frames):
        """Mix in noise with slow fade modulation."""
        if self.noise > 0:
            t = np.arange(self.global_sample, self.global_sample + frames) / self.sample_rate
            lfo = 0.5 - 0.5 * np.cos(2 * np.pi * t / self.fade_period)
            noise_level = (lfo * self.noise)[:, np.newaxis]
            random_noise = np.random.uniform(-self.amp, self.amp, (frames, 2)).astype(np.float32)
            outdata[:] = outdata * (1 - noise_level) + random_noise * noise_level
            np.clip(outdata, -1.0, 1.0, out=outdata)

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


def common_args_from_parsed(args):
    """Extract common arguments as a dict for passing to VectorScopePlayer."""
    return {
        'sample_rate': args.rate,
        'secs': args.secs,
        'amp': args.amp,
        'device': args.device,
        'noise': args.noise,
        'fade_period': args.fade_period,
    }
