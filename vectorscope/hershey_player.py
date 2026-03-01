
"""
Hershey font rendering for oscilloscope display.
"""
import numpy as np
from .base import VectorScopePlayer
from .text import resample_polyline
from HersheyFonts import HersheyFonts

def build_xy_from_hershey(hf, text, samples, amp, penlift_samples):
    """Render text with HersheyFonts and return (xy_data, blanking) arrays.

    Parameters:
        hf: An initialized HersheyFonts instance (font loaded, normalize_rendering called).
        text: The string to render.
        samples: Total number of output samples.
        amp: Amplitude scaling factor.
        penlift_samples: Number of blanked samples between strokes.

    Returns:
        (xy_data, blanking) where xy_data is float32 shape (samples, 2)
        and blanking is bool shape (samples,).
    """
    raw_strokes = hf.strokes_for_text(text)

    all_polys = []
    for stroke in raw_strokes:
        poly_np = np.array(stroke, dtype=np.float32)
        all_polys.append(poly_np)
    if not all_polys:
        return (np.zeros((samples, 2), dtype=np.float32),
                np.zeros(samples, dtype=bool))

    # Global normalization: center the entire block of text and scale to [-1, 1] range
    all_points_flat = np.vstack(all_polys)
    min_x, min_y = np.min(all_points_flat, axis=0)
    max_x, max_y = np.max(all_points_flat, axis=0)

    text_width = max_x - min_x
    text_height = max_y - min_y

    max_dim = max(text_width, text_height)
    if max_dim < 1e-6:
        max_dim = 1.0

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    final_polys = []
    for poly in all_polys:
        centered_poly = poly - [center_x, center_y]
        scaled_poly = (centered_poly / (max_dim / 2.0)) * 0.9
        final_polys.append(scaled_poly)

    all_polys = final_polys

    # Resample polylines to have a constant drawing speed and add pen lifts
    total_samples = samples - (len(all_polys) * penlift_samples)

    total_length = 0.0
    for p in all_polys:
        if len(p) > 1:
            total_length += np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))

    if total_length < 1e-6:
        total_length = 1.0

    output_points = []
    blanking_parts = []
    for i, poly in enumerate(all_polys):
        if len(poly) > 1:
            poly_len = np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))
            num_points = max(2, int(total_samples * (poly_len / total_length)))
            resampled = resample_polyline(poly, num_points)
            output_points.append(resampled)
            blanking_parts.append(np.zeros(len(resampled), dtype=bool))

        if penlift_samples > 0 and i < len(all_polys) - 1:
            start_pt = output_points[-1][-1] if output_points else np.zeros(2, dtype=np.float32)
            end_pt = all_polys[i + 1][0]
            t_lift = np.linspace(0, 1, penlift_samples, dtype=np.float32)
            lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
            output_points.append(lift)
            blanking_parts.append(np.ones(penlift_samples, dtype=bool))

    # Closing pen lift: wrap from end of last stroke back to start of first
    if penlift_samples > 0 and output_points:
        start_pt = output_points[-1][-1]
        end_pt = all_polys[0][0]
        t_lift = np.linspace(0, 1, penlift_samples, dtype=np.float32)
        lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
        output_points.append(lift)
        blanking_parts.append(np.ones(penlift_samples, dtype=bool))

    if not output_points:
        return (np.zeros((samples, 2), dtype=np.float32),
                np.zeros(samples, dtype=bool))

    final_xy = np.vstack(output_points).astype(np.float32)
    blanking = np.concatenate(blanking_parts) if blanking_parts else np.zeros(len(final_xy), dtype=bool)

    # Ensure the data is exactly samples long
    if len(final_xy) > samples:
        final_xy = final_xy[:samples]
        blanking = blanking[:samples]
    elif len(final_xy) < samples:
        pad_n = samples - len(final_xy)
        padding = np.tile(final_xy[-1], (pad_n, 1))
        final_xy = np.vstack([final_xy, padding])
        blanking = np.concatenate([blanking, np.ones(pad_n, dtype=bool)])

    return (np.clip(final_xy * amp, -1.0, 1.0), blanking)


class HersheyPlayer(VectorScopePlayer):
    """
    Renders text using the single-stroke Hershey fonts.
    """

    def __init__(self, text="Hello", font="futural", penlift=20, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.font_name = font
        self.penlift_samples = penlift

        # Instantiate the HersheyFonts object
        self.hf = HersheyFonts()
        # Load the selected font
        self.hf.load_default_font(self.font_name)
        # Apply library's built-in normalization
        self.hf.normalize_rendering(1.0)

        self._build_xy_data()

    def _build_xy_data(self):
        """
        Convert the text into a series of XY coordinates using HersheyFonts library.
        """
        self.xy_data, self.xy_blanking = build_xy_from_hershey(
            self.hf, self.text, self.samples, self.amp, self.penlift_samples
        )
    
    def _on_start(self):
        print(f"Displaying '{self.text}' with Hershey '{self.font_name}' font.")
        print("Press Ctrl+C to stop.")

