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
                    min_pen_lift_samples=4,
                    margin=0.9, optimize_order=True,
                    min_points_per_segment=2, normalize=True,
                    intensities=None):
    """Unified polyline-to-XY pipeline.

    Args:
        polys: List of Nx2 arrays (raw polylines from font renderer).
        samples: Total number of output samples.
        amp: Amplitude scaling factor.
        pen_lift_samples: Max blanked interpolation samples between contours.
        min_pen_lift_samples: Minimum pen-lift samples for zero-distance hops.
        margin: Scaling margin (0.9 = 10% border).
        optimize_order: If True, reorder contours to minimize beam travel.
        min_points_per_segment: Minimum points per resampled segment.
        normalize: If True, automatically center and scale to [-1, 1].
        intensities: Optional list of floats (0-1) for each polyline.

    Returns:
        (xy_data, blanking, intensity_data, n_penlifts, n_samples) 
        where xy_data is float32 (samples, 2), blanking is bool (samples,),
        intensity_data is float32 (samples,), n_penlifts is int, and 
        n_samples is int.
    """
    # Filter short segments
    if intensities is not None:
        new_polys, new_intensities = [], []
        for p, inten in zip(polys, intensities):
            if len(p) >= 2:
                new_polys.append(p)
                new_intensities.append(inten)
        polys, intensities = new_polys, new_intensities
    else:
        polys = [p for p in polys if len(p) >= 2]

    if not polys:
        return (np.zeros((samples, 2), dtype=np.float32),
                np.zeros(samples, dtype=bool),
                np.ones(samples, dtype=np.float32))

    # Normalize
    if normalize:
        polys = normalize_polylines(polys, margin=margin)

    # Optimize contour order
    if optimize_order:
        # 1. Reorder contours greedily
        indices = list(range(len(polys)))
        remaining = indices[1:]
        ordered = [indices[0]]
        while remaining:
            prev_end = polys[ordered[-1]][-1]
            best_idx = min(remaining,
                           key=lambda j: ((polys[j][0] - prev_end) ** 2).sum())
            remaining.remove(best_idx)
            ordered.append(best_idx)
        
        # 2. Map original intensities to the new order
        if intensities is not None:
            intensities = [intensities[i] for i in ordered]
        
        # 3. Apply the new order and rotate closed polylines
        polys = [polys[i] for i in ordered]
        new_polys = [polys[0]]
        for k in range(1, len(polys)):
            prev_end = new_polys[-1][-1]
            poly = polys[k]
            if np.allclose(poly[0], poly[-1], atol=1e-10):
                dists = ((poly[:-1] - prev_end) ** 2).sum(axis=1)
                best = int(np.argmin(dists))
                if best != 0:
                    poly = np.vstack([poly[best:-1], poly[:best + 1]])
            new_polys.append(poly)
        polys = new_polys

    # Pre-compute per-transition pen-lift counts (distance-proportional)
    lift_counts = []
    if pen_lift_samples > 0:
        for i in range(len(polys)):
            next_start = polys[i + 1][0] if i < len(polys) - 1 else polys[0][0]
            dist = float(np.sqrt(((next_start - polys[i][-1]) ** 2).sum()))
            t = min(1.0, dist / 2.0)
            n = int(min_pen_lift_samples + (pen_lift_samples - min_pen_lift_samples) * t)
            lift_counts.append(max(min_pen_lift_samples, n))

    # Allocate samples across contours.
    # We must ensure each contour gets at least min_points_per_segment.
    pen_lift_total = sum(lift_counts) if lift_counts else 0
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
    intensity_parts = []
    for i, (p, n) in enumerate(zip(polys, segment_counts)):
        pts = resample_polyline(p, n)
        out.append(pts)
        blanking_parts.append(np.zeros(len(pts), dtype=bool))
        inten = intensities[i] if intensities is not None else 1.0
        intensity_parts.append(np.full(len(pts), inten, dtype=np.float32))

        if lift_counts and i < len(polys) - 1:
            n_lift = lift_counts[i]
            start_pt = pts[-1]
            end_pt = polys[i + 1][0]
            t_lift = np.linspace(0, 1, n_lift, dtype=np.float64)
            lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
            out.append(lift)
            blanking_parts.append(np.ones(n_lift, dtype=bool))
            intensity_parts.append(np.zeros(n_lift, dtype=np.float32))

    # Closing pen lift: wrap from end of last contour back to start of first
    if lift_counts and out:
        n_lift = lift_counts[-1]
        start_pt = out[-1][-1]
        end_pt = polys[0][0]
        t_lift = np.linspace(0, 1, n_lift, dtype=np.float64)
        lift = start_pt * (1 - t_lift[:, np.newaxis]) + end_pt * t_lift[:, np.newaxis]
        out.append(lift)
        blanking_parts.append(np.ones(n_lift, dtype=bool))
        intensity_parts.append(np.zeros(n_lift, dtype=np.float32))

    xy = np.vstack(out)
    blanking = np.concatenate(blanking_parts)
    intensity_data = np.concatenate(intensity_parts)

    # Pad or truncate to exact sample count
    if len(xy) < samples:
        pad_n = samples - len(xy)
        padding = np.tile(xy[-1], (pad_n, 1))
        xy = np.vstack([xy, padding])
        blanking = np.concatenate([blanking, np.ones(pad_n, dtype=bool)])
        intensity_data = np.concatenate([intensity_data, np.zeros(pad_n, dtype=np.float32)])
    else:
        xy = xy[:samples]
        blanking = blanking[:samples]
        intensity_data = intensity_data[:samples]

    # Amplitude clip
    xy = np.clip(xy * amp, -1.0, 1.0).astype(np.float32)

    return xy, blanking, intensity_data, len(polys), len(xy)


def optimize_flat_path(xy, blanking, intensity):
    """Reorder strokes in a flat path to minimize total hop travel distance.

    Uses greedy nearest-neighbor: each stroke starts nearest to where
    the previous stroke ended.
    """
    if len(xy) < 2:
        return xy, blanking, intensity

    hop_starts = np.where(blanking)[0]
    if len(hop_starts) == 0:
        return xy, blanking, intensity   # single stroke, nothing to reorder

    starts = np.concatenate([[0], hop_starts])
    ends   = np.concatenate([hop_starts - 1, [len(xy) - 1]])
    n = len(starts)

    strokes_xy  = [xy[s:e+1]               for s, e in zip(starts, ends)]
    strokes_blk = [blanking[s:e+1].copy()  for s, e in zip(starts, ends)]
    strokes_itn = [intensity[s:e+1]         for s, e in zip(starts, ends)]

    # Greedy nearest-neighbor reorder anchored at stroke 0
    remaining = list(range(1, n))
    order = [0]
    while remaining:
        prev_end = strokes_xy[order[-1]][-1]
        best = min(remaining,
                   key=lambda j: ((strokes_xy[j][0] - prev_end) ** 2).sum())
        remaining.remove(best)
        order.append(best)

    # Reconstruct path in new order
    out_xy, out_blk, out_itn = [], [], []
    for i, k in enumerate(order):
        blk_k = strokes_blk[k]
        blk_k[0] = (i > 0)   # first stroke: no hop arrival; rest: hop arrival
        out_xy.append(strokes_xy[k])
        out_blk.append(blk_k)
        out_itn.append(strokes_itn[k])

    return (np.vstack(out_xy),
            np.concatenate(out_blk),
            np.concatenate(out_itn))


def path_to_xy(xy, blanking, intensity, samples, amp=1.0, min_hop_samples=4, max_hop_speed=0.02):
    """Convert a pre-built flat path with embedded pen-lifts to XY output.

    Unlike polylines_to_xy, this takes a single flat path (built by
    PolylineBuilder) with blanked hop segments already embedded.

    Blanked hops use a modified arc-length parameterization: each hop is
    assigned a fixed weight (min_hop_samples/samples of the total parameter
    range) regardless of its actual travel distance.  This guarantees the
    Z-blank circuit gets exactly min_hop_samples to settle between objects
    while the remaining budget is distributed across visible arc proportionally.
    Long hops still get more samples than short ones (proportional to actual
    distance within that fixed-weight allocation — they just all share the same
    total weight class).

    Blanking convention (trailing-edge): blanking[i] = True means the segment
    FROM point i-1 TO point i was blanked.  blanking[0] is unused.

    Args:
        xy: Nx2 float64 array, pre-normalized to oscilloscope coordinates.
        blanking: N bool array.
        intensity: N float32 array.
        samples: desired output sample count.
        amp: amplitude scaling applied before clipping to [-1, 1].
        min_hop_samples: samples reserved per blanked hop for Z-blank settle.

    Returns:
        (xy_out, blanking_out, intensity_out) as (float32 Sx2, bool S, float32 S).
    """
    if len(xy) < 2:
        out_xy = np.zeros((samples, 2), dtype=np.float32)
        out_blk = np.ones(samples, dtype=bool)
        out_itn = np.zeros(samples, dtype=np.float32)
        return out_xy, out_blk, out_itn

    # Close the loop: blanked return from last point to first.
    # Without this the beam jumps bare across the screen at the frame boundary,
    # causing inter-frame ringing and a faint visible line.
    if not np.allclose(xy[-1], xy[0], atol=1e-9):
        xy = np.vstack([xy, xy[0:1]])
        blanking = np.concatenate([blanking, [True]])
        intensity = np.concatenate([intensity, np.float32([0.0])])

    diffs = np.diff(xy, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))

    # Build effective arc-length parameterization.
    # hop_mask[i] = True means segment i (xy[i]→xy[i+1]) is a blanked hop.
    hop_mask = blanking[1:]  # trailing-edge: segment i blanked iff blanking[i+1]=True
    n_hops = int(hop_mask.sum())

    if n_hops > 0:
        # Per-hop effective arc: distance-proportional with floor, so each hop
        # gets max(min_hop_samples, ceil(dist/max_hop_speed)) samples.
        # Derivation: C_i/(sum(C_j) + L_visible)*samples = alloc_i
        #   → C_i = alloc_i * L_visible / vis_budget
        hop_seglens = seglen[hop_mask]
        hop_alloc = np.maximum(
            float(min_hop_samples),
            np.ceil(hop_seglens / max_hop_speed)
        )
        total_hop_alloc = hop_alloc.sum()
        vis_budget = max(1.0, samples - total_hop_alloc)
        L_visible = float(seglen[~hop_mask].sum()) if n_hops < len(hop_mask) else 0.0

        if L_visible > 0:
            C_per_hop = hop_alloc * L_visible / vis_budget
            effective_seglen = seglen.copy()
            effective_seglen[hop_mask] = C_per_hop
        else:
            # Only hops, no visible content — equal weight per hop
            effective_seglen = np.where(hop_mask, 1.0, seglen)
    else:
        effective_seglen = seglen

    s = np.concatenate(([0.0], np.cumsum(effective_seglen)))
    total = s[-1]

    if total <= 0:
        out_xy = np.tile(xy[:1], (samples, 1)).astype(np.float32)
        out_blk = blanking[:1].repeat(samples)
        out_itn = np.zeros(samples, dtype=np.float32)
        return out_xy, out_blk, out_itn

    t = np.linspace(0.0, total, samples)
    x_out = np.interp(t, s, xy[:, 0])
    y_out = np.interp(t, s, xy[:, 1])

    # For each output sample find which segment it falls in (0 = first segment).
    # Segment i spans xy[i]→xy[i+1]; trailing-edge blanking is blanking[i+1].
    seg_idx = np.clip(np.searchsorted(s[1:], t, side='left'), 0, len(xy) - 2)
    blk_out = blanking[seg_idx + 1]
    itn_out = np.where(blk_out, np.float32(0.0), intensity[seg_idx + 1]).astype(np.float32)

    xy_out = np.clip(np.column_stack((x_out, y_out)) * amp, -1.0, 1.0).astype(np.float32)
    return xy_out, blk_out, itn_out
