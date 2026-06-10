import json
from pathlib import Path
import shutil
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.auto import tqdm
from collections import defaultdict
import os
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

from .ioutils import read_image_cv
from .feats import SIFTMatcher, SIFTFeatures, SIFTFeaturesCopy, affine_is_valid, estimate_affine_from_landmarks, fullscale_bsplinegrid_from_landmarks
from .graph import propagate_transforms_multi_source_dijkstra_numba
from .graph import build_tracks_and_canonical
from .transforms import lift_affine_to_fullres
from .annotations_yolo import transform_yolo_obb_to_pages
from .page_similarity import phash
from .config import config

import cv2

from numba import types
from numba.typed import List

# Ensure process forking (important for CoW)
import multiprocessing as mp
from multiprocessing import Pool
import psutil
import gc

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#from mpire import WorkerPool
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

mat_type = types.float64[:, ::1]

edge_tuple_type = types.Tuple((
    types.int64,   # i
    types.int64,   # j
    mat_type,      # M_ij
    types.int64,   # dummy / weight placeholder if needed
    types.int64,    # alternative weight (inlier count)
    types.float64    # new weight (inlier ratio)
))


def build_edges_typed(edges_py):
    """Converts Python edge tuples into Numba-compatible typed edge tuples.

    Args:
        edges_py: Iterable of Python edge tuples containing affine transforms
            and matching metadata.

    Returns:
        numba.typed.List: Typed edge list compatible with Numba-compiled graph
        propagation functions.
    """
    edges_nb = List.empty_list(edge_tuple_type)

    for i, j, M_ij, inlier_pairs, w, inlier_ratio in edges_py:
        # critical: force contiguous memory instead of view
        M_clean = np.ascontiguousarray(M_ij.copy())

        edges_nb.append((
            np.int64(i),
            np.int64(j),
            M_clean,
            np.int64(0),     # placeholder instead of inlier_pairs
            np.int64(w),
            np.float64(inlier_ratio)
        ))

    return edges_nb


# Worker module-level variables, initialized by _init_worker()
_worker_matcher = None
_worker_ransac_thresh = None

'''
Global variables used for copy-on-write data sharing
between controller and worker processes. This is 
necessary, because it is the only efficient and easy way to distribute
such complex data structures among child processes 
(multiprocessing shared memory is too limited).
Copy-on-write only works on platforms with process forking, such as Linux!
'''
_worker_feats = None
_worker_shapes = None


def _init_worker(matcher_params, ransac_thresh):
    """
    Initialize worker processes (for ProcessPoolExecutor)

    Args:
        matcher_params: Parameters used to construct a SIFTMatcher instance.
        ransac_thresh: RANSAC reprojection threshold used during matching.
    """
    global _worker_matcher, _worker_ransac_thresh

    _worker_matcher = SIFTMatcher(**matcher_params)
    # _worker_matcher_cheap = SIFTMatcher(**matcher_cheap_params)
    _worker_ransac_thresh = ransac_thresh


def _reduce_feats(feats, k):
    """
    Selects only the k features with the highest "response", which
    is also used by OpenCV internally to rank features.
    For SIFT, this uses the built-in measure of contrast.
    Result should be roughly the same as lowering the nfeatures argument given to cv2.SIFT_create().
    Currently unused and unmaintained!

    Args:
        feats: SIFTFeatures object containing keypoints and descriptors.
        k: Number of strongest features to retain.

    Returns:
        SIFTFeaturesCopy: Reduced feature set containing only the top-k
        responses.
    """
    if feats.descriptors is None:
        return feats

    responses = np.array([kp[3] for kp in feats.keypoints])  # kp[3] = response
    idx = np.argsort(-responses)[:k]

    return SIFTFeaturesCopy(
        [feats.keypoints[i] for i in idx],
        feats.descriptors[idx]
    )


def bit_similarity(a, b):
    """Computes Hamming similarity between integers using XOR bit differences.

    Args:
        a: First integer hash value.
        b: Second integer hash value.

    Returns:
        int: Number of differing bits.
    """
    return (a ^ b).bit_count()


def _compute_edge_worker(args):
    """
    Align two pages
    Returns: affine transformation matrix, inlier pairs and their count

    Args:
        args: Tuple containing two page indices.

    Returns:
        tuple | None:
            Edge tuple containing page indices, affine transform,
            inlier correspondences, inlier count, and inlier ratio.

        Returns None if alignment fails or the affine transform is invalid.
    """
    global _worker_feats, _worker_shapes, _worker_ransac_thresh
    i, j = args

    fi_full = _worker_feats[i]
    fj_full = _worker_feats[j]
    shape_i = _worker_shapes[i]

    M_ij, inlier_pairs, inlier_ratio = _worker_matcher.match_ransac_affine(
        fi_full[0],
        fj_full[0],
        ransac_reproj_threshold=_worker_ransac_thresh,
        max_iters=config["ransac_graph"]["max_iters"],
        confidence=config["ransac_graph"]["confidence"]
    )

    if M_ij is None or not inlier_pairs or not affine_is_valid(M_ij, shape_i):
        return None
    
    inlier_pairs = np.asarray(inlier_pairs, dtype=np.int32)

    return (i, j, M_ij, inlier_pairs, int(inlier_pairs.shape[0]), inlier_ratio)

class NeighborPairs:
    """Iterator over candidate image pairs excluding root-root pairs."""

    def __init__(self, n, n_roots):
        """Initializes the pair iterator.

        Args:
            n: Total number of images.
            n_roots: Number of root/reference images.
        """
        self.n = n
        self.n_roots = n_roots

    def __iter__(self):
        """Iterates over valid image index pairs.

        Yields:
            tuple[int, int]: Pair of image indices.
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not (i < self.n_roots and j < self.n_roots):  # disallow pairs between two roots
                    yield (i, j)

    def __len__(self):
        """Returns the number of valid non-root image pairs that will be output by this iterator.

        Returns:
            int: Total number of candidate pairs.
        """
        total = self.n * (self.n - 1) // 2
        excluded = self.n_roots * (self.n_roots - 1) // 2
        return total - excluded

def make_gray_thumbnail(path, size):
    """
    Make grayscale (0...1) thumbnail from image
    """
    img = read_image_cv(path, max_dim=size)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    img = img.astype(np.float32) / 255.0

    return img

def build_edges_and_features_parallel(
    matcher_params: SIFTMatcher,
    images: list[Path],
    max_dim: int | None,
    ransac_thresh: float,
    neighbor_pairs: NeighborPairs,
    max_workers: int | None = None,
    prefilter_k: int = 24  # 24
):
    """
    Extracts features and computes pairwise alignment edges in parallel.

    Pipeline stages:
        1. Parallel SIFT feature extraction.
        2. nearest-neighbor prefiltering with efficient tree search, either between PCA representations or between pHashes of the images.
        3. Parallel affine RANSAC matching.

    Args:
        matcher_params: Parameters used to initialize SIFTMatcher.
        images: List of image paths.
        max_dim: Maximum image dimension used during feature extraction (images are downscaled to this).
        ransac_thresh: RANSAC reprojection threshold.
        neighbor_pairs: Iterable of candidate image pairs.
        max_workers: Number of worker processes/threads.
        prefilter_k: Number of nearest neighbors retained per image.

    Returns:
        tuple:
            - list: Extracted feature data for each image.
            - list: Valid alignment edges between images.
    """
    # copy on write globals, necessary to save memory in multiprocessing
    global _worker_feats, _worker_shapes

    if max_workers is None:
        max_workers = psutil.cpu_count(logical=False)  # use physical cpu count instead of logical/hyperthreads

    matcher = SIFTMatcher(**matcher_params)

    # Stage 1: feature extraction (threads, opencv will release GIL)
    def compute_feat(p):
        img_small, scale = read_image_cv(p, max_dim=max_dim, return_scale=True)

        if config["prefiltering"]["descriptor"] == "pca":
            thumb = make_gray_thumbnail(p, config["prefiltering"]["pca_image_size"])
            descriptor = thumb.flatten()
        else: # use phash
            # phash length is quite influential, think of it in terms of a 2D matrix: sqrt(32*8)=16 bins per side
            descriptor = phash(p, config["prefiltering"]["phash_bytes"])
        ret = (SIFTFeatures(matcher, img_small),
               img_small.shape[:2], scale, descriptor)
        return ret  # returns SIFTFeatures, shape, scale, phash_tuple

    print("Started feats")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        feats = list(tqdm(
            ex.map(compute_feat, images),
            total=len(images)
        ))
    print("Feats finished")

    shapes = [f[1] for f in feats]

    # Stage 2: prefilter
    print("Prefiltering pairs...")

    if config["prefiltering"]["descriptor"] == "pca":
        print("Starting PCA")
        # build float matrix from raw downscaled grayscale images
        X = np.stack([
            f[3]
            for f in feats
        ])

        n_components = min(len(X)//2, config["prefiltering"]["pca_components"])
        pca = PCA(
            n_components=n_components,
            whiten=True,
            svd_solver="randomized",
            random_state=42
        )

        X = pca.fit_transform(X)
        X = normalize(X)
        # Nearest neighbor search on PCA representations of thumbnail images
        nn = NearestNeighbors(
            n_neighbors=prefilter_k + 1,   # +1 because self-match included
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=max_workers
        )

        # rebuild features with new PCA descriptor
        feats = [
            (f[0], f[1], f[2], X[i].astype(np.float32))
            for i, f in enumerate(feats)
        ]
    else:
        # Build binary matrix from phashes
        X = np.stack([
            f[3].hash.reshape(-1)
            for f in feats
        ])

        # Nearest neighbor search on phashes
        nn = NearestNeighbors(
            n_neighbors=prefilter_k + 1,   # +1 because self-match included
            metric="hamming",
            algorithm="ball_tree",
            n_jobs=max_workers
        )

    # store in globals (copy-on-write memory for forked processes)
    _worker_feats = feats
    _worker_shapes = shapes
    
    print("Nearest neighbors tree started")
    nn.fit(X)
    print("Nearest neighbors tree complete, starting search")
    start_time = time.time()

    _, indices = nn.kneighbors(X)
    end_time = time.time()
    print(f"Nearest neighbors search complete, time: {end_time - start_time}")

    filtered_pairs = set()

    for i in range(len(feats)):
        added = 0

        for j_idx in range(1, len(indices[i])):  # skip self
            j = indices[i][j_idx]

            # disallow root-root pairs
            if i < neighbor_pairs.n_roots and j < neighbor_pairs.n_roots:
                continue

            a, b = sorted((i, j))

            filtered_pairs.add((a, b))
            added += 1

            if added >= prefilter_k:
                break

    filtered_pairs = list(filtered_pairs)
    del indices
    del X
    del nn

    print(f"Filtered {len(neighbor_pairs)} to {len(filtered_pairs)} pairs")

    edges = []

    gc.collect()
    #gc.freeze()  # try to prevent gc from clobbering forked CoW memory

    # Stage 3: RANSAC
    with Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(matcher_params, ransac_thresh),
        #maxtasksperchild=1000,
    ) as pool:
        for r in tqdm(
            pool.imap_unordered(_compute_edge_worker, filtered_pairs, chunksize=16),
            total=len(filtered_pairs),
        ):
            if r is not None:
                edges.append(r)

    print("Finished build_edges_and_features_parallel")

    return feats, edges


def mean_transform_diff(T1, T2):
    """
    Computes mean absolute difference between shared transforms.

    Args:
        T1: Mapping from node ID to affine transform.
        T2: Mapping from node ID to affine transform.

    Returns:
        float: Mean absolute element-wise transform difference.
    """
    keys = set(T1.keys()) & set(T2.keys())

    total = 0.0
    count = 0

    for k in keys:
        diff = T1[k] - T2[k]
        total += np.abs(diff).mean()
        count += 1

    return total / count if count > 0 else 0.0


def mean_relative_transform_error(T1, T2, eps=1e-12):
    """
    Computes mean relative error between shared transforms.

    Args:
        T1: Mapping from node ID to affine transform.
        T2: Mapping from node ID to affine transform.
        eps: Small constant preventing division by zero.

    Returns:
        float: Mean relative transform error.
    """
    keys = set(T1.keys()) & set(T2.keys())

    total = 0.0
    count = 0

    for k in keys:
        A = T1[k]
        B = T2[k]

        num = np.linalg.norm(A - B)          # difference
        den = np.linalg.norm(B) + eps        # scale

        total += num / den
        count += 1

    return total / count if count > 0 else 0.0

# alignment globals, necessary for CoW multiprocessing
_final_all_images = None
_final_feats = None
_final_kpnode_to_lm = None
_final_lm_canon = None
_final_T_to_root = None
_final_root_ph = None
_final_root_index = None
_final_s_root_small = None
_final_n_roots = None
_final_ransac_threshold = None
_final_min_matches = None

def _align_worker(page_idx):
    """
    Worker function for parallel final alignment.

    Args:
        page_idx: Index of the page to align.

    Returns:
        tuple | None:
            Page index and alignment result tuple.

        Returns None if alignment fails.
    """
    global _final_all_images, _final_feats, _final_kpnode_to_lm, _final_lm_canon
    global _final_T_to_root, _final_root_ph, _final_root_index, _final_s_root_small
    global _final_n_roots, _final_ransac_threshold, _final_min_matches

    if page_idx < _final_n_roots:
        return None

    page = _final_all_images[page_idx]
    s_mov_small = _final_feats[page_idx][2]
    page_ph = _final_feats[page_idx][3]

    if config["prefiltering"]["descriptor"] == "pca":
        similarity_to_root = np.linalg.norm(
            page_ph - _final_root_ph
        )
    else:  # use phash
        similarity_to_root = page_ph - _final_root_ph

    fi = _final_feats[page_idx][0]

    estimation_result = estimate_affine_from_landmarks(
        page_idx,
        fi,
        _final_kpnode_to_lm,
        _final_lm_canon,
        ransac_thresh=_final_ransac_threshold,
        min_matches=_final_min_matches,
        max_iters=config["ransac_final"]["max_iters"],
        confidence=config["ransac_final"]["confidence"],
    )

    if estimation_result is None:
        M_small = _final_T_to_root.get(page_idx)

        if M_small is None:
            return None

        M_full = lift_affine_to_fullres(
            M_small,
            s_mov=s_mov_small,
            s_ref=_final_s_root_small,
        )

        return page_idx, (
            _final_root_index,
            M_full,
            0,
            page,
            similarity_to_root,
        )

    M_small, inlier_count, mean_residual = estimation_result

    M_full = lift_affine_to_fullres(
        M_small,
        s_mov=s_mov_small,
        s_ref=_final_s_root_small,
    )

    if config["bspline"]["enabled"]:
        bspline_transformation = fullscale_bsplinegrid_from_landmarks(page, page_idx, M_full,
                                                s_mov_small, _final_s_root_small,
                                                fi, _final_kpnode_to_lm, _final_lm_canon)
    else:
        bspline_transformation = None

    return page_idx, (
        _final_root_index,
        M_full,
        inlier_count,
        page,
        mean_residual,
        bspline_transformation
    )


def final_alignment_parallel(
    all_images,
    feats,
    n_roots,
    root_index,
    kpnode_to_lm,
    lm_canon,
    T_to_root,
    min_matches=10,
    ransac_threshold=3.0,
    processes=None,
):
    """
    Runs landmark-based final alignment in parallel.

    Args:
        all_images: List of all image paths.
        feats: Extracted feature data.
        n_roots: Number of root/reference images.
        root_index: Root image index.
        kpnode_to_lm: Mapping from keypoints to landmark IDs.
        lm_canon: Canonical landmark coordinates.
        T_to_root: Mapping from image index to root transform.
        min_matches: Minimum landmark matches required.
        ransac_threshold: RANSAC reprojection threshold.
        processes: Number of worker processes.

    Returns:
        dict: Mapping from page index to alignment result tuple.
    """
    global _final_all_images, _final_feats, _final_kpnode_to_lm, _final_lm_canon
    global _final_T_to_root, _final_root_ph, _final_root_index, _final_s_root_small
    global _final_n_roots, _final_ransac_threshold, _final_min_matches

    _final_all_images = all_images
    _final_feats = feats
    _final_kpnode_to_lm = kpnode_to_lm
    _final_lm_canon = lm_canon
    _final_T_to_root = T_to_root
    _final_root_ph = feats[root_index][3]
    _final_root_index = root_index
    _final_s_root_small = feats[root_index][2]
    _final_n_roots = n_roots
    _final_ransac_threshold = ransac_threshold
    _final_min_matches = min_matches

    page_indices = range(n_roots, len(all_images))

    T_q = {}

    with Pool(processes or psutil.cpu_count(logical=False)) as pool:
        results = pool.imap_unordered(
            _align_worker,
            page_indices
        )

        for result in tqdm(results, total=len(all_images) - n_roots, desc="Final alignment"):
            if result is None:
                continue

            page_idx, value = result
            T_q[page_idx] = value

    return T_q

# ENTRY POINT
def align_pages(
    images_to_align: List[Path],
    roots: List[Path],
    yolo_obb_dir
):
    """
    Aligns pages into one or more root/template coordinate systems.

    The pipeline:
        1. Extracts SIFT features.
        2. Builds pairwise alignment edges.
        3. Propagates transforms from all roots, competitively selecting the best chain of transformations to reach every node possible.
        4. Builds landmark tracks.
        5. Performs final landmark-based alignment refinement.

    Args:
        images_to_align: Pages requiring alignment.
        roots: Root/reference page images.
        yolo_obb_dir: YOLO OBB annotation directory.

    Returns:
        tuple:
            - dict: Final alignment results indexed by image path.
            - list: Unaligned image paths.
    """
    all_images = roots + images_to_align
    n = len(all_images)
    n_roots = len(roots)

    neighbor_pairs = NeighborPairs(n, n_roots)

    feats, edges = build_edges_and_features_parallel(
        {"nfeatures": config["sift"]["nfeatures"], 
         "min_matches": config["ransac_graph"]["min_matches"],
         "lowes_ratio_threshold": config["sift"]["lowes_ratio_threshold"],
         "num_trees": config["flann"]["num_trees"]
        },
        all_images,
        config["page_alignment"]["max_alignment_dim"],
        config["ransac_graph"]["pixel_threshold"],
        neighbor_pairs,
        prefilter_k=config["prefiltering"]["k"]
    )

    if not edges:
        return None

    root_indices = list(range(n_roots))
    root_indices_numba = List(root_indices)  # numba typed list

    edges_numba = build_edges_typed(edges)

    print("Starting dijkstra")
    # One global propagation: all roots compete simultaneously
    T_all, owner, dist = propagate_transforms_multi_source_dijkstra_numba(
        len(feats),
        root_indices_numba,  # _numba
        edges_numba,  # _numba
        hop_penalty=config["transform_propagation"]["hop_penalty"],
        prob_floor=1e-6,
    )

    # numba variants not needed after this
    del edges_numba
    del root_indices_numba

    T_q_list = []

    print("Starting to build tracks and canonical, final alignment")
    for root_idx in root_indices:
        # Keep only nodes assigned to this root
        T_root = {node: T_all[node]
                  for node, r in owner.items() if r == root_idx}

        # print(len(T_root))

        # Build tracks only inside this root's owned subset
        kpnode_to_lm, lm_canon = build_tracks_and_canonical(
            feats, edges, T_root,
            min_support=config["page_alignment"]["min_support"]
        )

        print("Tracks complete, starting final alignment in parallel")

        final = final_alignment_parallel(
            all_images,
            feats,
            n_roots,
            root_idx,
            #config["page_alignment"]["max_alignment_dim"],
            kpnode_to_lm,
            lm_canon,
            T_root,
            min_matches=config["ransac_final"]["min_matches"],
            ransac_threshold=config["ransac_final"]["pixel_threshold"]
        )

        T_q_list.append(final)

        if config["debug"]["enabled"]:
            with open(f'T_q_{os.path.basename(roots[root_idx]).split(".")[0]}.json', 'w') as fp:
                json.dump(final, fp, default=lambda o: 'not serialized')

    T_q = {}
    for d in T_q_list:
        for key, value in d.items():
            # replace if doesn't exist or new value has higher inlier count, or >= inlier count and lower mean residual
            if key not in T_q or value[2] > T_q[key][2] or (value[2] >= T_q[key][2] and value[4] < T_q[key][4]):
                T_q[key] = value

    if config["debug"]["enabled"]:
        with open('T_q_final.json', 'w') as fp:
            json.dump(T_q, fp, default=lambda o: 'not serialized')
        
        transform_yolo_obb_to_pages(
            yolo_obb_dir,   # contains images/, labels/, classes.txt
            T_q,
            all_images,
            output_dir="aligned"
        )

    unaligned = []
    for pageidx in range(0, n):
        if pageidx >= n_roots and pageidx not in T_q:
            pagepath = all_images[pageidx]
            if config["debug"]["enabled"]:
                shutil.copy2(str(pagepath), str(
                    Path("unaligned") / Path(pagepath.name)))
            
            unaligned.append(pagepath)

    pretty_output_dict = {}
    for id, value in T_q.items():
        root_index, transformation_to_root, inlier_count, page, mean_residual, bspline = value
        pretty_output_dict[all_images[id]] = {"root_path": all_images[root_index],
                                               "transformation_to_root": transformation_to_root,
                                               "bspline_transformation_post": bspline,
                                               "inlier_count": inlier_count, "mean_residual": mean_residual}
    return pretty_output_dict, unaligned


# simple tests during development
if __name__ == "__main__":
    test_images = list(Path("tests/out").glob("*.jpg"))

    annotations_dir = "yolo-obb"
    roots = list((Path(annotations_dir) / Path("images")).glob("*.jpg"))

    # change config temporarily for testing purposes
    config["debug"]["enabled"] = True
    feats, edges, T_q, unaligned = align_pages(test_images, roots, annotations_dir)

    # print(T_q)

    print(len(edges))
