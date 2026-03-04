"""Text rendering for oscilloscope XY display."""

import numpy as np
import soundfile as sf

from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path

from .base import VectorScopePlayer
from .polyline import polylines_to_xy

# Re-export for backward compatibility
from .polyline import resample_polyline  # noqa: F401


def path_to_polylines(mpl_path: Path, points_per_curve: int = 30):
    """
    Convert a Matplotlib Path (with CURVE3/CURVE4) into a list of polylines.
    Returns list[np.ndarray] where each polyline is Nx2 float array.
    """
    polys = []
    cur = []

    verts = mpl_path.vertices
    codes = mpl_path.codes
    if codes is None:
        return [verts.copy()]

    i = 0
    last = None
    while i < len(verts):
        code = codes[i]
        v = verts[i]

        if code == Path.MOVETO:
            if len(cur) > 1:
                polys.append(np.array(cur, dtype=np.float64))
            cur = [v]
            last = v
            i += 1

        elif code == Path.LINETO:
            cur.append(v)
            last = v
            i += 1

        elif code == Path.CLOSEPOLY:
            if len(cur) > 1:
                cur.append(cur[0])
                polys.append(np.array(cur, dtype=np.float64))
            cur = []
            last = None
            i += 1

        elif code == Path.CURVE3:
            if last is None:
                i += 1
                continue
            p0 = last
            p1 = v
            p2 = verts[i + 1]
            ts = np.linspace(0.0, 1.0, points_per_curve, endpoint=True)
            curve = (1-ts)[:,None]**2 * p0 + 2*(1-ts)[:,None]*ts[:,None]*p1 + ts[:,None]**2 * p2
            cur.extend(curve[1:].tolist())
            last = p2
            i += 2

        elif code == Path.CURVE4:
            if last is None:
                i += 1
                continue
            p0 = last
            p1 = v
            p2 = verts[i + 1]
            p3 = verts[i + 2]
            ts = np.linspace(0.0, 1.0, points_per_curve, endpoint=True)
            curve = ((1-ts)[:,None]**3 * p0 +
                     3*(1-ts)[:,None]**2 * ts[:,None] * p1 +
                     3*(1-ts)[:,None] * ts[:,None]**2 * p2 +
                     ts[:,None]**3 * p3)
            cur.extend(curve[1:].tolist())
            last = p3
            i += 3

        else:
            i += 1

    if len(cur) > 1:
        polys.append(np.array(cur, dtype=np.float64))

    return polys


def build_xy_from_text(text, font_size=1.0, font_family="DejaVu Sans", curve_pts=30,
                       samples=48000*5, pen_lift_samples=300, amp=1.0):
    """Convert text to XY coordinate array for oscilloscope display."""
    # Escape special characters that trigger matplotlib mathtext
    text = text.replace('$', r'\$')

    fp = FontProperties(family=font_family)
    tp = TextPath((0, 0), text, size=font_size, prop=fp)

    polys = path_to_polylines(tp, points_per_curve=curve_pts)

    return polylines_to_xy(polys, samples, amp=amp,
                           pen_lift_samples=pen_lift_samples)


class TextPlayer(VectorScopePlayer):
    """Real-time text display with interactive input.

    Supports both matplotlib outline fonts and single-stroke Hershey fonts.
    The font type is auto-detected from the font name.
    """

    def __init__(self, text="Hello", font="futural", curve_pts=30,
                 pen_lift_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.font = font
        self.curve_pts = curve_pts
        self.pen_lift_samples = pen_lift_samples

        from .hershey_player import is_hershey_font
        self._is_hershey = is_hershey_font(font)
        if self._is_hershey:
            from HersheyFonts import HersheyFonts
            self.hf = HersheyFonts()
            self.hf.load_default_font(font)
            self.hf.normalize_rendering(1.0)

        self._update_text(text)

    def _update_text(self, text):
        """Regenerate XY data for new text."""
        self.text = text
        if self._is_hershey:
            from .hershey_player import build_xy_from_hershey
            self.xy_data, self.xy_blanking, self.z_intensity = build_xy_from_hershey(
                self.hf, text, self.samples, self.amp, self.pen_lift_samples
            )
        else:
            self.xy_data, self.xy_blanking, self.z_intensity = build_xy_from_text(
                text,
                font_family=self.font,
                curve_pts=self.curve_pts,
                samples=self.samples,
                pen_lift_samples=self.pen_lift_samples,
                amp=self.amp
            )
        self.position = 0

    def _on_start(self):
        print(f"Displaying: {self.text}")
        print("Press Ctrl+C to stop.")


def generate_wav(text, output, rate, freq, amp, font, curve_pts, penlift):
    """Generate a WAV file."""
    from .hershey_player import is_hershey_font

    samples = int(rate / abs(freq))

    if is_hershey_font(font):
        from .hershey_player import build_xy_from_hershey
        from HersheyFonts import HersheyFonts
        hf = HersheyFonts()
        hf.load_default_font(font)
        hf.normalize_rendering(1.0)
        xy, _blanking, _intensity = build_xy_from_hershey(hf, text, samples, amp, penlift)
    else:
        xy, _blanking, _intensity = build_xy_from_text(
            text,
            font_size=1.0,
            font_family=font,
            curve_pts=curve_pts,
            samples=samples,
            pen_lift_samples=penlift,
            amp=amp
        )

    xy[-1] = xy[0]

    sf.write(output, xy.astype(np.float32), rate)
    print(f"Wrote {output} ({rate} Hz, {samples} frames). L=X, R=Y")
