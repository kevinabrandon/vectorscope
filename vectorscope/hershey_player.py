
"""
Hershey font rendering for oscilloscope display.
"""
import numpy as np
from .polyline import polylines_to_xy
from HersheyFonts import HersheyFonts

# Build the set of known Hershey font names once at import time
_hf_tmp = HersheyFonts()
HERSHEY_FONT_NAMES = set(_hf_tmp.default_font_names)
del _hf_tmp


def is_hershey_font(name):
    """Return True if *name* matches a known Hershey font."""
    return name in HERSHEY_FONT_NAMES

def build_xy_from_hershey(hf, text, samples, amp, penlift_samples):
    """Render text with HersheyFonts and return (xy_data, blanking, intensity) arrays.

    Parameters:
        hf: An initialized HersheyFonts instance (font loaded, normalize_rendering called).
        text: The string to render.
        samples: Total number of output samples.
        amp: Amplitude scaling factor.
        penlift_samples: Number of blanked samples between strokes.

    Returns:
        (xy_data, blanking, intensity) where xy_data is float32 shape (samples, 2),
        blanking is bool shape (samples,), and intensity is float32 shape (samples,).
    """
    raw_strokes = hf.strokes_for_text(text)

    polys = [np.array(stroke, dtype=np.float64) for stroke in raw_strokes]

    return polylines_to_xy(polys, samples, amp=amp,
                           pen_lift_samples=penlift_samples)
