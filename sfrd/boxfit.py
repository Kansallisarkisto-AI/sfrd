import numpy as np
from ortools.sat.python import cp_model

def build_cost_image(gray, downscale=1, peak=2, normalization_max_percentile=98):
    """Build an edge-cost image using Sobel filters

    Args:
        scale (int): average-pool the grayscale image down by this block size before
                taking the gradient (bigger = coarser/faster)
        peak (int): peak cost (represents normalization_max_percentile)
                resulting cost array will contain integers 0...peak

    Returns:
        an np.int32 array of shape (ceil(H/scale), ceil(W/scale)).
    """
    import cv2
    h, w = gray.shape
    g = gray.astype(np.float32)

    if downscale > 1:
        new_w = (w + downscale - 1) // downscale
        new_h = (h + downscale - 1) // downscale
        g = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # normalize against a high percentile rather than the max, so one
    # unusually sharp pixel doesn't wash out the whole scale
    # adjusting this percentile makes sense, used to be 99
    ref = np.percentile(grad, normalization_max_percentile) or 1.0
    normalized = np.clip(grad / ref, 0, 1)

    cost = peak * (normalized)
    return np.clip(cost, 0, peak).astype(np.int32)

def _fit_from_four_corners(pts):
    """
    Special case: exactly 4 points (a quad, e.g. an already-simplified
    corner set with no points sampled along the edges).
 
    With only 4 points there's nothing to average per side directly, so
    instead take the MIDPOINT of each edge (each pair of adjacent corners,
    assuming points are given in polygon order around the perimeter) and
    use those midpoints' coordinates as the box sides. This is the
    equivalent of "the middles" when you don't have enough points per side
    to average out noise.
 
    Edges are classified as horizontal (top/bottom) or vertical
    (left/right) by whichever of dx/dy is larger for that edge, so this
    works regardless of which corner the point order starts at.
    """
    xs, ys = pts[:, 0], pts[:, 1]
    midpoints = []
    is_vertical = []
    for i in range(4):
        p0, p1 = pts[i], pts[(i + 1) % 4]
        mid = (p0 + p1) / 2
        dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
        midpoints.append(mid)
        is_vertical.append(dy > dx)  # steep edge -> left/right side
 
    midpoints = np.array(midpoints)
    vertical_mids = midpoints[np.array(is_vertical)]
    horizontal_mids = midpoints[~np.array(is_vertical)]
 
    # Need exactly 2 of each for a sensible quad; fall back to plain
    # min/max bounding box otherwise (degenerate shape).
    if len(vertical_mids) != 2 or len(horizontal_mids) != 2:
        return xs.min(), ys.min(), xs.max(), ys.max()
 
    left, right = sorted(vertical_mids[:, 0])
    top, bottom = sorted(horizontal_mids[:, 1])
    return left, top, right, bottom

def cells_to_polygons(cells):
    """
    cells: array-like of shape (N, 4), each [x0, y0, x1, y1]

    Returns: np.ndarray of shape (N, 4, 2), each a clockwise polygon
             starting at top-left: (x0,y0) -> (x1,y0) -> (x1,y1) -> (x0,y1)
    """
    cells = np.asarray(cells, dtype=float)
    x0, y0, x1, y1 = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]

    polys = np.stack([
        np.stack([x0, y0], axis=1),   # top-left
        np.stack([x1, y0], axis=1),   # top-right
        np.stack([x1, y1], axis=1),   # bottom-right
        np.stack([x0, y1], axis=1),   # bottom-left
    ], axis=1)

    return polys  # shape (N, 4, 2)
 
 
def fit_middle_axis_aligned_box(points, iterations=3):
    """
    Fit an axis-aligned box through the MIDDLE of a noisy polygon
    (instead of enclosing all points like a normal bounding box).
 
    For polygons with many points (a noisy contour), this works by
    assigning each point to its nearest side (left/right/top/bottom), then
    averaging each side's points to get that side's position, repeating a
    few times to converge.
 
    For exactly 4 points (just the corners, no points along the edges),
    there's nothing to average per side, so it instead uses each edge's
    midpoint (see _fit_from_four_corners) -- the "middle" of each pair of
    adjacent corners.
 
    Returns: left, right, top, bottom
    """
    pts = np.asarray(points, dtype=float)
 
    if len(pts) == 4:
        return _fit_from_four_corners(pts)
 
    xs, ys = pts[:, 0], pts[:, 1]
 
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
 
    for _ in range(iterations):
        dists = np.stack([
            np.abs(xs - left), np.abs(xs - right),
            np.abs(ys - top), np.abs(ys - bottom),
        ], axis=1)
        side = np.argmin(dists, axis=1)
 
        left = xs[side == 0].mean() if np.any(side == 0) else left
        right = xs[side == 1].mean() if np.any(side == 1) else right
        top = ys[side == 2].mean() if np.any(side == 2) else top
        bottom = ys[side == 3].mean() if np.any(side == 3) else bottom
 
    return left, top, right, bottom

def _pair_order(ci, cj):
    """For two rectangles (x0, y0, x1, y1), decide which comes first.

    If the boxes are cleanly separated on exactly one axis, use that axis.
    If they're separated on both (e.g. diagonal grid neighbors) or on
    neither (overlapping/ambiguous original detections), pick whichever
    axis has the larger signed gap.
    """
    xi0, yi0, xi1, yi1 = ci
    xj0, yj0, xj1, yj1 = cj

    if xi1 <= xj0:
        gx, xf = xj0 - xi1, True
    elif xj1 <= xi0:
        gx, xf = xi0 - xj1, False
    else:
        gx, xf = -(min(xi1, xj1) - max(xi0, xj0)), xi0 <= xj0

    if yi1 <= yj0:
        gy, yf = yj0 - yi1, True
    elif yj1 <= yi0:
        gy, yf = yi0 - yj1, False
    else:
        gy, yf = -(min(yi1, yj1) - max(yi0, yj0)), yi0 <= yj0

    return ('x', xf) if gx >= gy else ('y', yf)

def _can_touch(ci, cj, max_expand):
    """Check if rectangles ci and cj can possibly overlap,
    considering a maximum expansion of max_expand on every side.
    (we can disregard shrinking because it can't create any new overlaps)
    Use this to prune unnecessary constraints.
    """
    xi0, yi0, xi1, yi1 = ci
    xj0, yj0, xj1, yj1 = cj
    x_reach = (xi1 + max_expand) > (xj0 - max_expand) and (xj1 + max_expand) > (xi0 - max_expand)
    y_reach = (yi1 + max_expand) > (yj0 - max_expand) and (yj1 + max_expand) > (yi0 - max_expand)
    return x_reach and y_reach

def _rle(vals):
    """Run-length encode a list of ints into (start_offset, end_offset,
    value) triples."""
    segs = []
    start, cur = 0, vals[0]

    for i in range(1, len(vals)):
        if vals[i] != cur:
            segs.append((start, i - 1, cur))
            start, cur = i, vals[i]
    segs.append((start, len(vals) - 1, cur))

    return segs

def solve(cells, cost, cost_downscale=1, solve_downscale=1, max_expand=25, max_push=15,
          min_size=15, gap=1, w_edge=1, w_dev=1, w_dev_grow=None,
          max_time=60.0, workers=8):
    """
    Optimize detected cell boundaries using a CP-SAT model.

    Each cell edge may move inward or outward within the configured limits.
    Cell edges have a cost based on the Canny edges (high-frequency content) of the image.
    The cost is higher when the cell edge lies on a Canny edge in the image.

    The objective balances edge costs against penalties for shrinking
    or growing the original cells, while enforcing minimum cell sizes and
    preventing neighboring cells from overlapping.

    This will tend to push edges away from written content or other lines in the image.

    Args:
        cells: Array-like of shape (N, 4) containing detected cell boxes as
            [x0, y0, x1, y1] in original-image pixel coordinates.
        cost: Integer edge-cost image of shape (H, W), typically produced by
            build_cost_image().
        cost_downscale: Number of original-image pixels represented by one entry
            in `cost`. Must match the downscaling used to build the cost image.
        solve_downscale: Number of original-image pixels represented by one
            coordinate unit in the CP-SAT model. Larger values reduce model
            resolution and may improve solving speed.
        max_expand: Maximum number of pixels an edge may move outward from its
            original position.
        max_push: Maximum number of pixels an edge may move inward from its
            original position.
        min_size: Minimum allowed width and height of a cell, in pixels.
        gap: Minimum gap between neighboring cells, in pixels.
        w_edge: Weight applied to the image edge-cost term.
        w_dev: Penalty per pixel for moving an edge inward, shrinking the cell.
        w_dev_grow: Penalty per pixel for moving an edge outward, growing the
            cell. If None, defaults to 0.5 * w_dev.
        max_time: Maximum CP-SAT solving time in seconds.
        workers: Number of CP-SAT search workers.

    Returns:
        new_cells: NumPy array of shape (N, 4) containing the optimized
            [x0, y0, x1, y1] cell boxes in original-image pixel coordinates.
        info: Dictionary containing solver status, objective value, timing,
            edge/shrink/grow costs, constraint counts, and lookup statistics.

    Raises:
        RuntimeError: If CP-SAT fails to find a feasible solution.
    """
    cells = np.asarray(cells, int)
    Hn, Wn = cost.shape          # native cost-image size
    n = len(cells)

    # Create CP-SAT model
    mdl = cp_model.CpModel()

    if w_dev_grow is None:
        w_dev_grow = 0.5 * w_dev

    def downscaled_solver_coordinate(v):
        return int(round(v / solve_downscale))

    max_expand_s = max(1, round(max_expand / solve_downscale))
    max_push_s = max(1, round(max_push / solve_downscale))
    min_size_s = max(1, round(min_size / solve_downscale))
    gap_s = max(0, round(gap / solve_downscale))

    X0, X1, Y0, Y1, dev_shrink, dev_grow, edge_costs = [], [], [], [], [], [], []
    lookup_stats = {'positions': 0, 'segments': 0, 'constant': 0, 'total': 0}
    bound = max_expand_s + max_push_s

    # Add model variables and size constraints for each cell (rectangle)
    for x0, y0, x1, y1 in cells:
        x0s, y0s, x1s, y1s = downscaled_solver_coordinate(x0), \
                             downscaled_solver_coordinate(y0), \
                             downscaled_solver_coordinate(x1), \
                             downscaled_solver_coordinate(y1)

        # int variables describing expanded or shrunk edge positions
        a = mdl.new_int_var(x0s - max_expand_s, x0s + max_push_s, "")
        b = mdl.new_int_var(x1s - max_push_s, x1s + max_expand_s, "")
        c = mdl.new_int_var(y0s - max_expand_s, y0s + max_push_s, "")
        d = mdl.new_int_var(y1s - max_push_s, y1s + max_expand_s, "")

        # minimum size constraint
        mdl.add(b - a >= min_size_s)
        mdl.add(d - c >= min_size_s)

        # Add variables for amount of shrinkage/growth.
        # Used for the penalty terms "w_dev * sum(dev_shrink)""
        # and "w_dev_grow * sum(dev_grow)".
        # 'a' and 'c' are the "lo" edges (x0, y0): moving them to a larger value
        # means the edge moved inward (shrinking).
        # A smaller value means it moved outward (growing). 
        # 'b' and 'd' are the "hi" edges (x1, y1): the sign is 
        # flipped so larger means outward (growing).
        for var, orig_s, is_lo in ((a, x0s, True), (b, x1s, False),
                                    (c, y0s, True), (d, y1s, False)):
            shrink = mdl.new_int_var(0, bound, "")
            grow = mdl.new_int_var(0, bound, "")
            delta = var - orig_s
            if is_lo:
                mdl.add(shrink - grow == delta)
            else:
                mdl.add(grow - shrink == delta)
            dev_shrink.append(shrink)
            dev_grow.append(grow)

        X0.append(a); X1.append(b); Y0.append(c); Y1.append(d)

        # Precalculate *ceteris paribus* edge costs, i.e. the
        # accumulated cost values along the edge line boundary for each possible edge position.
        # This means we assume that when side edges are moved,
        # the top and bottom edges would stay unmoved and vice versa.
        # The assumption is not strictly true but it speeds up solving
        # significantly and is valid enough when the range of movement is small.
        ny0, ny1 = np.clip([round(y0 / cost_downscale), round(y1 / cost_downscale)], 0, Hn)
        nx0, nx1 = np.clip([round(x0 / cost_downscale), round(x1 / cost_downscale)], 0, Wn)
        vcost = cost[ny0:ny1, :].sum(axis=0) if ny1 > ny0 else np.zeros(Wn, int)
        hcost = cost[:, nx0:nx1].sum(axis=1) if nx1 > nx0 else np.zeros(Hn, int)

        def lookup(var, lo_s, hi_s, table, size_native):
            vals = []
            for q in range(lo_s, hi_s + 1):
                idx = (q * solve_downscale) // cost_downscale
                idx = min(max(idx, 0), size_native - 1)
                vals.append(int(table[idx]))

            # run-length encode the costs to prune the number of constraints
            # this will identify ranges of more than 1px with a constant cost
            segs = _rle(vals)

            # collect some statistics
            lookup_stats['positions'] += len(vals)
            lookup_stats['segments'] += len(segs)
            lookup_stats['total'] += 1

            # output a constant for results with one segment
            # this means the cost is always the same along this axis
            if len(segs) == 1:
                lookup_stats['constant'] += 1
                return segs[0][2]

            # For results with more than one segment, output a variable
            # this means the cost changes depending on which segment the edge (var)
            # lies on (the "cost position"). 
            # Only one "cost position" can be active at a time for each edge.
            cv = mdl.new_int_var(min(vals), max(vals), "")
            edge_cost_positions = []
            for s_off, e_off, val in segs:
                edge_position_active = mdl.new_bool_var("")
                # check when edge position is activated by var (the edge location)
                mdl.add(var - lo_s >= s_off).only_enforce_if(edge_position_active)
                mdl.add(var - lo_s <= e_off).only_enforce_if(edge_position_active)
                # impose cost when edge position is activated
                mdl.add(cv == val).only_enforce_if(edge_position_active)
                edge_cost_positions.append(edge_position_active)
            mdl.add_exactly_one(edge_cost_positions)
            return cv

        # add edge costs for edges, only inside allowed range of movement
        edge_costs.append(lookup(a, x0s - max_expand_s, x0s + max_push_s, vcost, Wn))
        edge_costs.append(lookup(b, x1s - max_push_s, x1s + max_expand_s, vcost, Wn))
        edge_costs.append(lookup(c, y0s - max_expand_s, y0s + max_push_s, hcost, Hn))
        edge_costs.append(lookup(d, y1s - max_push_s, y1s + max_expand_s, hcost, Hn))

    # Add minimum number of constraints based on the original cell order.
    # We prune away any constraints that would be satisfied by the
    # cell starting position + maximum expansion constraint anyway.
    #
    # This should result in around O(n * neighbors-per-cell) constraints
    #
    # Benefits: 
    # * pairwise constraints on the same number line are transitive
    # * much cheaper than a NoOverlap2D constraint
    # * solution will not switch around the cell order, rather
    #   just fine-tune the boundaries
    # Issues:
    # * solution will not allow overlapping cells 
    #   (good for tables, might be problematic for forms,
    #   where the overlaps are empty space that is covered "just in case")
    n_pairs = n_constrained = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if not _can_touch(cells[i], cells[j], max_expand):
                continue
            n_constrained += 1
            axis, i_first = _pair_order(cells[i], cells[j])
            if axis == 'x':
                if i_first:
                    mdl.add(X1[i] + gap_s <= X0[j])
                else:
                    mdl.add(X1[j] + gap_s <= X0[i])
            else:
                if i_first:
                    mdl.add(Y1[i] + gap_s <= Y0[j])
                else:
                    mdl.add(Y1[j] + gap_s <= Y0[i])

    # minimize the sum of edge costs and shrink/grow penalties
    mdl.minimize(w_edge * sum(edge_costs)
                 + w_dev * sum(dev_shrink)
                 + w_dev_grow * sum(dev_grow))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = workers
    solver.parameters.relative_gap_limit = 0.05  # allow 5% gap from optimal
    solver.parameters.log_search_progress = False
    st = solver.solve(mdl)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(solver.status_name(st))

    # scale solve-space coordinates back to pixel space
    new_cells = np.array([[solver.value(X0[k]) * solve_downscale, solver.value(Y0[k]) * solve_downscale,
                           solver.value(X1[k]) * solve_downscale, solver.value(Y1[k]) * solve_downscale]
                          for k in range(n)])
    def _val(v):
        return v if isinstance(v, int) else solver.value(v)

    info = dict(status=solver.status_name(st), objective=solver.objective_value,
                wall=solver.wall_time,
                edge_cost=sum(_val(v) for v in edge_costs),
                shrink=sum(solver.value(v) for v in dev_shrink),
                grow=sum(solver.value(v) for v in dev_grow),
                pairs_total=n_pairs, pairs_constrained=n_constrained,
                lookup_positions=lookup_stats['positions'],
                lookup_segments=lookup_stats['segments'],
                lookup_constant=lookup_stats['constant'],
                lookup_total=lookup_stats['total'])
    return new_cells, info


# REMOVE:
def make_test_data(path='crop.png', seed=7, keep=0.85, cost_scale=1, peak=6):
    import cv2
    gray = cv2.imread(path, 0)
    xs = [512, 639, 765, 891, 1017, 1143, 1269, 1392, 1517, 1644, 1770]
    ys = [151, 204, 255, 308, 362, 415, 469, 522, 574, 627, 693]
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if i < 3 or rng.random() > keep:
                continue
            a, b = rng.integers(-6, 18, 2)
            c, d = rng.integers(-18, 6, 2)
            cells.append([xs[i] + a, ys[j] + b, xs[i + 1] + c, ys[j + 1] + d])
    cells = np.array(cells)
    cost = build_cost_image(gray, downscale=cost_scale, peak=peak)
    return gray, cells, cost


def draw(gray, cells, new_cells, cost, cost_scale, path, scale=1.8):
    import cv2
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis = (vis * .5 + 255 * .5).astype(np.uint8)
    if cost.max() > 0:
        heat = cv2.applyColorMap((cost * (255.0 / cost.max())).astype(np.uint8),
                                 cv2.COLORMAP_BONE)
        if cost_scale > 1:
            heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0)
    for x0, y0, x1, y1 in cells:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (235, 140, 0), 1)
    for x0, y0, x1, y1 in new_cells:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 160, 0), 2)
    lo, hi = cells[:, :2].min(0) - 25, cells[:, 2:].max(0) + 25
    cv2.imwrite(path, cv2.resize(vis[lo[1]:hi[1], lo[0]:hi[0]], None,
                                 fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))


if __name__ == '__main__':
    COST_SCALE = 4     # e.g. 4 -> coarse cost image, cheap to build
    SOLVE_SCALE = 4    # e.g. 4 -> coarse solve, cheap to optimize
    W_DEV = 0.5*1.0
    W_DEV_GROW = 0.5*0.5   # growth is 5x cheaper than shrinkage; try 1.0 for
                        # the old symmetric behavior, or 0 for free growth

    gray, cells, cost = make_test_data(cost_scale=COST_SCALE, peak=2)  # adjusting peak will speed up optimization
    print(f'{len(cells)} cells, cost image {cost.shape} at cost_scale={COST_SCALE}, '
          f'solving at solve_scale={SOLVE_SCALE}')
    new_cells, info = solve(cells, cost, cost_downscale=COST_SCALE, solve_downscale=SOLVE_SCALE,
                            max_expand=25, max_push=25, w_dev=W_DEV, w_dev_grow=W_DEV_GROW)
    print(info['status'], 'objective', info['objective'],
          '| edge_cost', info['edge_cost'],
          '| shrink', info['shrink'], '| grow', info['grow'],
          '| %.2fs' % info['wall'],
          f"| pairs constrained {info['pairs_constrained']}/{info['pairs_total']}",
          f"| lookup segs {info['lookup_segments']}/{info['lookup_positions']}"
          f" ({info['lookup_constant']}/{info['lookup_total']} constant)")
    draw(gray, cells, new_cells, cost, COST_SCALE, 'boxfit_output.png')