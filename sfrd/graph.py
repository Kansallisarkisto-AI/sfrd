import numpy as np
import heapq
from numba import njit, types
from numba.typed import List, Dict
import math

from .transforms import invert_affine_numba, compose_affine_numba, apply_affine, apply_affine_numba
from .unionfind import UnionFind

# ---- types (must be global) ----
mat_type = types.float64[:, ::1]

edge_tuple_type = types.Tuple((
    types.int64,   # neighbor
    mat_type,      # transform
    types.float64    # edge weight/cost
))

heap_tuple_type = types.Tuple((
    types.float64,  # priority
    types.int64,  # node id
    types.int64  # node id
))

@njit(cache=True, fastmath=True, nogil=True)
def propagate_transforms_multi_source_dijkstra_numba(
    m: int,
    roots: list[int],
    edges,
    hop_penalty: float = 0.0,
    prob_floor: float = 1e-6,
):
    """
    Multi-source shortest paths from all roots at once.
    Numba version that uses Numba types and typed edges.

    Edge probability:
        p_count = 1 - exp(-inliers / inlier_tau)
        p_ratio = inlier_ratio
        p = max(prob_floor, p_count * p_ratio)

    Edge cost:
        cost = -log(p) + hop_penalty

    Args:
        m: Number of graph nodes/images.
        roots: Root node indices used as propagation sources.
        edges: Iterable of graph edges containing affine transforms and
            matching statistics.
        hop_penalty: Additional cost added per traversed edge.
        prob_floor: Minimum probability value used to avoid log(0).

    Returns:
        tuple:
            T: Dict mapping node index to affine transform into the
                assigned root coordinate system.
            owner: Dict mapping node index to the root that reached it
                with minimum cost.
            dist: Dict mapping node index to shortest-path cost.
    """
    # initialize adjacency
    adj = List()
    for _ in range(m):
        adj.append(List.empty_list(edge_tuple_type))

    for (i, j, M_ij, inlier_pairs, w, inlier_ratio) in edges:
        M_ji = invert_affine_numba(M_ij)
        if M_ji is None:
            continue

        # probability from inlier ratio
        p = max(prob_floor, min(float(inlier_ratio), 1.0))

        cost = -math.log(p) + float(hop_penalty)

        # T[v] = T[u] @ (v -> u), so adjacency stores transform neighbor -> current
        adj[i].append((j, M_ji, cost))
        adj[j].append((i, M_ij, cost))

    # transforms dict
    T = Dict.empty(
        key_type=types.int64,
        value_type=mat_type,
    )
    owner = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    dist = Dict.empty(
        key_type=types.int64,
        value_type=types.float64,
    )

    heap = List.empty_list(heap_tuple_type)

    I = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)

    # imagine a super-source or virtual starting point
    for r in roots:
        T[r] = I.copy()
        owner[r] = r
        dist[r] = 0.0
        heapq.heappush(heap, (0.0, r, r))  # (dist, root_owner, node)

    heapq.heapify(heap)

    while heap:
        d_u, root_u, u = heapq.heappop(heap)

        if d_u > dist.get(u, float("inf")):
            continue
        if root_u != owner.get(u):
            continue

        for v, M_v_to_u, edge_cost in adj[u]:
            new_dist = d_u + edge_cost

            old_dist = dist.get(v, float("inf"))
            old_owner = owner.get(v, None)

            # tie-break by smaller root index for determinism
            if (new_dist < old_dist) or (
                abs(new_dist - old_dist) < 1e-12 and (old_owner is None or root_u < old_owner)
            ):
                dist[v] = new_dist
                owner[v] = root_u
                T[v] = compose_affine_numba(T[u], M_v_to_u)
                heapq.heappush(heap, (new_dist, root_u, v))

    return T, owner, dist


def build_tracks_and_canonical(feats, edges, T_to_root, min_support=3, strict=False):
    """
    Build tracks using all edges by default (strict=False). Optionally build tracks using only nodes
    with a transform to a root (strict=True).

    - A landmark is created if it has at least `min_support` observations
      from images with T_to_root.
    - Canonical position is computed from those registered observations.
    - ALL observations (including from unregistered images) are assigned
      to the landmark.

    Args:
        feats: Feature collections indexed by image.
        edges: Pairwise image matches and affine relationships.
        T_to_root: Mapping from image index to affine transform into
            root coordinates.
        min_support: Minimum number of registered observations required
            to create a landmark.
        strict: Whether to require both edge endpoints to have transforms
            to the root during track construction.

    Returns:
        tuple:
            kpnode_to_lm: Mapping from `(image_index, keypoint_index)` to
                landmark ID.
            lm_canon: Mapping from landmark ID to canonical 2D position
                in root coordinates.
    """
    uf = UnionFind()

    # Build full tracks (optional filtering)
    for i, j, M_ij, inlier_pairs, w1, w2 in edges:
        # optional strict filtering: only form tracks where all nodes have transform to root
        if strict and (i not in T_to_root or j not in T_to_root):
            continue
        for qi, tj in inlier_pairs:
            uf.union((i, int(qi)), (j, int(tj)))

    # Collect connected components
    comp = {}
    for node in list(uf.p.keys()):
        r = uf.find(node)
        comp.setdefault(r, []).append(node)

    kpnode_to_lm = {}
    lm_canon = {}
    lm_id = 0

    # Process each track
    for nodes in comp.values():
        # Split observations
        valid_obs = [(i, kp) for (i, kp) in nodes if i in T_to_root]

        # Require enough registered support
        if len(valid_obs) < min_support:
            continue

        # Compute canonical point from registered observations
        pts_root = []
        for i, kpidx in valid_obs:
            x, y = feats[i][0].keypoints[kpidx][0]
            xy = np.array([[x, y]], dtype=np.float64)
            xy_r = apply_affine(T_to_root[i], xy)[0]
            pts_root.append(xy_r)

        pts_root = np.asarray(pts_root, dtype=np.float64)
        canon = np.median(pts_root, axis=0)

        # Assign landmark to all nodes (registered + unregistered)
        for node in nodes:
            kpnode_to_lm[node] = lm_id

        lm_canon[lm_id] = canon
        lm_id += 1

    return kpnode_to_lm, lm_canon
