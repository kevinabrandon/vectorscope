"""SVG file display for oscilloscope XY rendering."""

import numpy as np

from .base import VectorScopePlayer
from .polyline import polylines_to_xy


def svg_paths_to_polylines(svg_paths, curve_pts=30):
    """Convert svgpathtools Path objects to a list of Nx2 numpy polylines.

    Each continuous subpath becomes one polyline. Bézier curves and arcs
    are tessellated with *curve_pts* samples per segment. Y is flipped
    (SVG is Y-down, oscilloscope is Y-up).
    """
    polys = []
    for path in svg_paths:
        if len(path) == 0:
            continue
        pts = []
        for seg in path:
            ts = np.linspace(0.0, 1.0, curve_pts, endpoint=False)
            for t in ts:
                p = seg.point(t)
                pts.append((p.real, -p.imag))
        # Add the final endpoint of the last segment
        p = path[-1].point(1.0)
        pts.append((p.real, -p.imag))
        if len(pts) >= 2:
            polys.append(np.array(pts, dtype=np.float64))
    return polys


class SVGPlayer(VectorScopePlayer):
    """Display an SVG file on the oscilloscope.

    Loads SVG path data via svgpathtools, converts to polylines, then
    feeds them through the shared polylines_to_xy pipeline.
    """

    def __init__(self, filepath, curve_pts=30, pen_lift_samples=20, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.curve_pts = curve_pts
        self.pen_lift_samples = pen_lift_samples
        self._load_svg()

    def _load_svg(self):
        from svgpathtools import svg2paths

        paths, _attrs = svg2paths(
            self.filepath,
            convert_circles_to_paths=True,
            convert_ellipses_to_paths=True,
            convert_lines_to_paths=True,
            convert_polylines_to_paths=True,
            convert_polygons_to_paths=True,
            convert_rectangles_to_paths=True,
        )
        if not paths:
            print(f"Warning: no paths found in {self.filepath}")
            self.xy_data = np.zeros((self.samples, 2), dtype=np.float32)
            self.xy_blanking = np.zeros(self.samples, dtype=bool)
            self.z_intensity = np.ones(self.samples, dtype=np.float32)
            return

        polys = svg_paths_to_polylines(paths, curve_pts=self.curve_pts)
        self.xy_data, self.xy_blanking, self.z_intensity = polylines_to_xy(
            polys, self.samples, amp=self.amp,
            pen_lift_samples=self.pen_lift_samples,
        )
        self.position = 0

    def _on_start(self):
        print(f"Displaying: {self.filepath}")
        print("Press Ctrl+C to stop.")
