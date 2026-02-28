
"""
Hershey font rendering for oscilloscope display.
"""
import numpy as np
from .base import VectorScopePlayer
from .text import resample_polyline
from HersheyFonts import HersheyFonts

class HersheyPlayer(VectorScopePlayer):
    """
    Renders text using the single-stroke Hershey fonts.
    """

    def __init__(self, text="Hello", font="futural", penlift=10, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.font_name = font
        self.penlift_samples = penlift
        
        # Instantiate the HersheyFonts object
        self.hf = HersheyFonts()
        # Load the selected font
        self.hf.load_default_font(self.font_name)
        # Apply library's built-in normalization
        self.hf.normalize_rendering(1.0) # Factor of 1.0 maps to internal units, which should be normalized
        
        self._build_xy_data()

    def _build_xy_data(self):
        """
        Convert the text into a series of XY coordinates using HersheyFonts library.
        """
        # strokes_for_text returns an iterable of continuous strokes (polylines)
        # Each stroke is a list of (x,y) tuples. The library applies its internal
        # scaling and offsets (from normalize_rendering) to this output.
        raw_strokes = self.hf.strokes_for_text(self.text)

        all_polys = []
        for stroke in raw_strokes:
            poly_np = np.array(stroke, dtype=np.float32)
            # The library's normalize_rendering likely handles Y-orientation, so no manual Y-flip here.
            all_polys.append(poly_np)
        if not all_polys:
            self.xy_data = np.zeros((self.samples, 2), dtype=np.float32)
            return
            
        # Global normalization: center the entire block of text and scale to [-1, 1] range
        all_points_flat = np.vstack(all_polys)
        min_x, min_y = np.min(all_points_flat, axis=0)
        max_x, max_y = np.max(all_points_flat, axis=0)
        
        text_width = max_x - min_x
        text_height = max_y - min_y

        # Calculate the maximum dimension and scale to fit into [-1, 1] range
        max_dim = max(text_width, text_height)
        if max_dim < 1e-6: # Avoid division by zero for empty text
            max_dim = 1.0
        
        # Center the text before scaling
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        # Apply scaling and centering
        # Scale to fill 90% of [-1, 1] range to provide some padding
        final_polys = []
        for poly in all_polys:
            centered_poly = poly - [center_x, center_y]
            scaled_poly = (centered_poly / (max_dim / 2.0)) * 0.9 
            final_polys.append(scaled_poly)
        
        all_polys = final_polys # Use the new scaled and centered polylines

        # Resample polylines to have a constant drawing speed and add pen lifts
        total_samples = self.samples - (len(all_polys) * self.penlift_samples)
        
        # Calculate total length of all actual drawing segments
        total_length = 0.0
        for p in all_polys:
            if len(p) > 1:
                total_length += np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        
        if total_length < 1e-6: # Handle case of single point or very short lines
            total_length = 1.0

        output_points = []
        for i, poly in enumerate(all_polys):
            if len(poly) > 1:
                poly_len = np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))
                num_points = max(2, int(total_samples * (poly_len / total_length)))
                output_points.append(resample_polyline(poly, num_points))

            if self.penlift_samples > 0 and i < len(all_polys) - 1:
                output_points.append(np.zeros((self.penlift_samples, 2), dtype=np.float32))
        
        if not output_points:
            self.xy_data = np.zeros((self.samples, 2), dtype=np.float32)
            return

        final_xy = np.vstack(output_points).astype(np.float32)
        
        # Ensure the data is exactly self.samples long
        if len(final_xy) > self.samples:
            final_xy = final_xy[:self.samples]
        elif len(final_xy) < self.samples:
            padding = np.zeros((self.samples - len(final_xy), 2), dtype=np.float32)
            final_xy = np.vstack([final_xy, padding])

        self.xy_data = np.clip(final_xy * self.amp, -1.0, 1.0)
    
    def _on_start(self):
        print(f"Displaying '{self.text}' with Hershey '{self.font_name}' font.")
        print("Press Ctrl+C to stop.")

