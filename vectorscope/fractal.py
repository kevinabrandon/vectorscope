"""Real-time oscilloscope fractal display."""

import numpy as np

from .base import VectorScopePlayer


def koch_snowflake(iterations=4):
    """Generate Koch snowflake as a list of (x, y) points."""
    def koch_segment(p1, p2, depth):
        if depth == 0:
            return [p1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        a = p1
        b = (p1[0] + dx/3, p1[1] + dy/3)
        angle = np.pi / 3
        cx = b[0] + dx/3 * np.cos(angle) - dy/3 * np.sin(angle)
        cy = b[1] + dx/3 * np.sin(angle) + dy/3 * np.cos(angle)
        c = (cx, cy)
        d = (p1[0] + 2*dx/3, p1[1] + 2*dy/3)
        e = p2

        points = []
        points.extend(koch_segment(a, b, depth - 1))
        points.extend(koch_segment(b, c, depth - 1))
        points.extend(koch_segment(c, d, depth - 1))
        points.extend(koch_segment(d, e, depth - 1))
        return points

    h = np.sqrt(3) / 2
    triangle = [
        (0.0, h * 2/3),
        (-0.5, -h * 1/3),
        (0.5, -h * 1/3),
        (0.0, h * 2/3)
    ]

    points = []
    for i in range(len(triangle) - 1):
        points.extend(koch_segment(triangle[i], triangle[i+1], iterations))
    points.append(triangle[-1])

    return np.array(points, dtype=np.float64)


def dragon_curve(iterations=12):
    """Generate dragon curve using L-system."""
    sequence = "F"
    for _ in range(iterations):
        new_seq = ""
        for c in sequence:
            if c == "F":
                new_seq += "F+G"
            elif c == "G":
                new_seq += "F-G"
            else:
                new_seq += c
        sequence = new_seq

    x, y = 0.0, 0.0
    angle = 0.0
    step = 1.0
    points = [(x, y)]

    for c in sequence:
        if c in "FG":
            x += step * np.cos(angle)
            y += step * np.sin(angle)
            points.append((x, y))
        elif c == "+":
            angle += np.pi / 2
        elif c == "-":
            angle -= np.pi / 2

    return np.array(points, dtype=np.float64)


def sierpinski_arrowhead(iterations=6):
    """Generate Sierpinski arrowhead curve using L-system."""
    if iterations % 2 == 0:
        sequence = "A"
    else:
        sequence = "B"

    for _ in range(iterations):
        new_seq = ""
        for c in sequence:
            if c == "A":
                new_seq += "B-A-B"
            elif c == "B":
                new_seq += "A+B+A"
            else:
                new_seq += c
        sequence = new_seq

    x, y = 0.0, 0.0
    angle = 0.0 if iterations % 2 == 0 else np.pi / 3
    step = 1.0
    points = [(x, y)]

    for c in sequence:
        if c in "AB":
            x += step * np.cos(angle)
            y += step * np.sin(angle)
            points.append((x, y))
        elif c == "+":
            angle += np.pi / 3
        elif c == "-":
            angle -= np.pi / 3

    return np.array(points, dtype=np.float64)


def hilbert_curve(iterations=5):
    """Generate Hilbert curve using L-system."""
    sequence = "A"
    for _ in range(iterations):
        new_seq = ""
        for c in sequence:
            if c == "A":
                new_seq += "+BF-AFA-FB+"
            elif c == "B":
                new_seq += "-AF+BFB+FA-"
            else:
                new_seq += c
        sequence = new_seq

    x, y = 0.0, 0.0
    angle = 0.0
    step = 1.0
    points = [(x, y)]

    for c in sequence:
        if c == "F":
            x += step * np.cos(angle)
            y += step * np.sin(angle)
            points.append((x, y))
        elif c == "+":
            angle += np.pi / 2
        elif c == "-":
            angle -= np.pi / 2

    return np.array(points, dtype=np.float64)


def levy_c_curve(iterations=12):
    """Generate Levy C curve."""
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    for _ in range(iterations):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            mid = (p1 + p2) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            peak = mid + np.array([-dy, dx]) / 2
            new_points.extend([peak, p2])
        points = np.array(new_points, dtype=np.float64)

    return points


FRACTALS = {
    "koch": koch_snowflake,
    "dragon": dragon_curve,
    "sierpinski": sierpinski_arrowhead,
    "hilbert": hilbert_curve,
    "levy": levy_c_curve,
}

DEFAULT_ITERATIONS = {
    "koch": 4,
    "dragon": 12,
    "sierpinski": 6,
    "hilbert": 5,
    "levy": 12,
}


def normalize_points(points):
    """Center and scale points to [-1, 1] range."""
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2
    span = (max_xy - min_xy).max()
    if span <= 0:
        span = 1.0
    return (points - center) / (span / 2) * 0.95


def resample_path(points, n_samples):
    """Resample path to n_samples points at constant speed."""
    if len(points) < 2:
        return np.zeros((n_samples, 2), dtype=np.float64)

    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_length[-1]

    if total_length <= 0:
        return np.repeat(points[:1], n_samples, axis=0)

    t = np.linspace(0, total_length, n_samples)
    x = np.interp(t, arc_length, points[:, 0])
    y = np.interp(t, arc_length, points[:, 1])

    return np.column_stack([x, y])


class FractalPlayer(VectorScopePlayer):
    """Display various fractal patterns."""

    def __init__(self, fractal_type="koch", iterations=None, **kwargs):
        super().__init__(**kwargs)
        self.fractal_type = fractal_type
        self.iterations = iterations or DEFAULT_ITERATIONS[fractal_type]
        self._generate_fractal()

    def _generate_fractal(self):
        """Generate and prepare fractal XY data."""
        fractal_func = FRACTALS[self.fractal_type]
        points = fractal_func(self.iterations)
        points = normalize_points(points)
        xy = resample_path(points, self.samples)

        self.xy_data = np.clip(xy * self.amp, -1.0, 1.0).astype(np.float32)

    def _on_start(self):
        print(f"📐 {self.fractal_type.title()} fractal (iterations={self.iterations})")
        print("  Press Ctrl+C to stop.")
