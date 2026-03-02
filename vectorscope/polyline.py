"""Shared polyline processing pipeline for oscilloscope rendering.

Pipeline: filter → normalize → optimize contour order → resample + pen lifts
→ pad/truncate → amplitude clip → float32 output.
"""

import numpy as np


def normalize_polylines(polys, margin=0.9):
    """Center and scale polylines into [-1, 1] range (preserving aspect).

    Args:
        polys: List of Nx2 arrays.
        margin: Scale factor applied after normalization (0.9 = 10% margin).
    """
    all_pts = np.vstack([p for p in polys if len(p) > 0])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = (max_xy - min_xy).max()
    if span <= 0:
        span = 1.0
    out = []
    for p in polys:
        q = (p - center) / (span / 2.0) * margin
        out.append(q)
    return out


def resample_polyline(p, n):
    """Resample a polyline p (Nx2) to n points at approximately constant speed."""
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0) if len(p) == 1 else np.zeros((n, 2), dtype=np.float64)
    diffs = np.diff(p, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate(([0.0], np.cumsum(seglen)))
    total = s[-1]
    if total <= 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, total, n)
    x = np.interp(t, s, p[:, 0])
    y = np.interp(t, s, p[:, 1])
    return np.column_stack((x, y))


def optimize_contour_order(polys):
    """Reorder contours and rotate closed ones to minimize beam travel.

    For each contour after the first, rotates the polyline so tracing
    begins at the point closest to where the previous contour ended.
    Then reorders contours greedily by nearest start point.
    """
    if len(polys) <= 1:
        return polys

    def _rotate_to_nearest(poly, target):
        """Rotate a closed polyline so it starts nearest to target."""
        if not np.allclose(poly[0], poly[-1], atol=1e-10):
            return poly  # open contour, can't rotate
        # Exclude duplicate closing point for distance calc
        dists = ((poly[:-1] - target) ** 2).sum(axis=1)
        best = int(np.argmin(dists))
        if best == 0:
            return poly
        return np.vstack([poly[best:-1], poly[:best + 1]])

    # Greedy nearest-neighbor ordering
    remaining = list(range(len(polys)))
    ordered = [remaining.pop(0)]
    while remaining:
        prev_end = polys[ordered[-1]][-1]
        best_idx = min(remaining,
                       key=lambda j: ((polys[j][0] - prev_end) ** 2).sum())
        remaining.remove(best_idx)
        ordered.append(best_idx)

    # Rotate each contour to start near previous end
    result = [polys[ordered[0]]]
    for k in range(1, len(ordered)):
        prev_end = result[-1][-1]
        result.append(_rotate_to_nearest(polys[ordered[k]], prev_end))

    return result


def polylines_to_xy(polys, samples, amp=1.0, pen_lift_samples=0,
                    margin=0.9, optimize_order=True,
                    min_points_per_segment=2, normalize=True):
    """Unified polyline-to-XY pipeline.

    Args:
        polys: List of Nx2 arrays (raw polylines from font renderer).
        samples: Total number of output samples.
        amp: Amplitude scaling factor.
        pen_lift_samples: Blanked interpolation samples between contours.
        margin: Scaling margin (0.9 = 10% border).
        optimize_order: If True, reorder contours to minimize beam travel.
        min_points_per_segment: Minimum points per resampled segment.
        normalize: If True, automatically center and scale to [-1, 1].

    Returns:
        (xy_data, blanking) where xy_data is float32 (samples, 2)
        and blanking is bool (samples,).
    """
    # Filter short segments
    polys = [p for p in polys if len(p) >= 2]

    if not polys:
        return (np.zeros((samples, 2), dtype=np.float32),
                np.zeros(samples, dtype=bool))

    # Normalize
    if normalize:
        polys = normalize_polylines(polys, margin=margin)

    # Optimize contour order
    if optimize_order:
        polys = optimize_contour_order(polys)

    # Allocate samples across contours.
    # We must ensure each contour gets at least min_points_per_segment.
    pen_lift_total = pen_lift_samples * len(polys)  # between each + closing wrap
    contour_budget = samples - pen_lift_total
    
    if contour_budget < len(polys) * min_points_per_segment:
        # If we're extremely over-budget, we just give everyone the minimum
        # and accept that the total count will exceed 'samples'. 
        # The final truncation at the end of the function will handle it.
        segment_counts = [min_points_per_segment] * len(polys)
    else:
        # Proportional allocation of the REMAINING budget
        lengths = []
        for p in polys:
            d = np.diff(p, axis=0)
            lengths.append(float(np.sqrt((d**2).sum(axis=1)).sum()))
        total_len = sum(lengths) if lengths else 1.0
        
        remaining_budget = contour_budget - (len(polys) * min_points_per_segment)
        segment_counts = []
        for L in lengths:
            n = min_points_per_segment + int(remaining_budget * (L / total_len))
            segment_counts.append(n)

    # Build XY sequence with pen lifts between contours
    out = []
    blanking_parts = []
    for i, (p, n) in enumerate(zip(polys, segment_counts)):
        pts = resample_polyline(p, n)
        out.append(pts)
        blanking_parts.append(np.zeros(len(pts), dtype=bool))

        if pen_lift_samples > 0 and i < len(polys) - 1:
            start_pt = pts[-1]
            end_pt = polys[i + 1][0]
            t_lift = np.linspace(0, 1, pen_lift_samples, dtype=np.float64)
            lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
            out.append(lift)
            blanking_parts.append(np.ones(pen_lift_samples, dtype=bool))

    # Closing pen lift: wrap from end of last contour back to start of first
    if pen_lift_samples > 0 and out:
        start_pt = out[-1][-1]
        end_pt = polys[0][0]
        t_lift = np.linspace(0, 1, pen_lift_samples, dtype=np.float64)
        lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
        out.append(lift)
        blanking_parts.append(np.ones(pen_lift_samples, dtype=bool))

    xy = np.vstack(out)
    blanking = np.concatenate(blanking_parts)

    # Pad or truncate to exact sample count
    if len(xy) < samples:
        pad_n = samples - len(xy)
        padding = np.tile(xy[-1], (pad_n, 1))
        xy = np.vstack([xy, padding])
        blanking = np.concatenate([blanking, np.ones(pad_n, dtype=bool)])
    else:
        xy = xy[:samples]
        blanking = blanking[:samples]

    # Amplitude clip
    xy = np.clip(xy * amp, -1.0, 1.0).astype(np.float32)

    return xy, blanking
