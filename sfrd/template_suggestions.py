import typing
from numba.typed import List
from pathlib import Path

import statistics
import math

from .page_alignment import build_edges_and_features_parallel, NeighborPairs, build_edges_typed
from .config import config
from .unionfind import UnionFind
from .graph import propagate_transforms_multi_source_dijkstra_numba

def find_connected_components(edges):
    """Groups nodes into connected components using union-find.
    
    Args:
        edges: List of tuples (a, b, ...) representing connected node pairs.
    
    Returns:
        List of lists, where each inner list contains node IDs in a component.
    """
    uf = UnionFind()

    # Union all nodes connected by an edge
    for a, b, *rest in edges:
        uf.union(a, b)

    # Group nodes by their root
    components = {}
    for node in uf.p:
        root = uf.find(node)
        components.setdefault(root, []).append(node)

    return list(components.values())

def suggest_templates(all_images: typing.List[Path], image_count_threshold=2, samples_per_component=1):
    """Selects representative template images from clustered image groups.
    
    Builds the collection alignment graph. identifies connected components, 
    and returns well-spread samples from each sufficiently large component 
    using a farthest-point heuristic.
    
    Args:
        all_images: List of image file paths.
        image_count_threshold: Minimum images required in a component to suggest templates.
        samples_per_component: Number of representative samples to select per component.
    
    Returns:
        List of lists of Path objects, where each inner list contains template
        samples from one component, or None if no templates were found.
    """
    n = len(all_images)

    neighbor_pairs = NeighborPairs(n, 0)

    # build features
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

    components = find_connected_components(edges)

    edges_numba = build_edges_typed(edges)

    best_central_nodes = []

    for component in components:
        if len(component) >= image_count_threshold:
            best_distance = math.inf
            best_node = None
            best_dist_arr = None

            for node in component:
                root_indices_numba = List([node])  # numba typed list

                # try using node as root
                _, _, dist = propagate_transforms_multi_source_dijkstra_numba(
                    len(feats),
                    root_indices_numba,
                    edges_numba,
                    hop_penalty=config["transform_propagation"]["hop_penalty"],
                    prob_floor=1e-6,
                )
                distances = [dist[m] for m in component 
                             if (m != node and not math.isinf(dist[m]))]
                
                if not distances:
                    continue
                
                average_dist = statistics.mean(distances)

                if average_dist < best_distance:
                    best_node = node
                    best_distance = average_dist
                    best_dist_arr = dist  # keep distance array

            if best_node is not None:
                # Heuristic farthest-point (k-center) sampling within this component,
                # starting from the most central node
                samples = [best_node]
                min_dist = best_dist_arr

                for _ in range(samples_per_component - 1):
                    candidates = [m for m in component if m not in samples]
                    if not candidates:
                        break

                    # find next node that is farthest away from last sample
                    # (min_dist from last sample is set above and after every dijkstra call)
                    next_node = max(
                        candidates,
                        key=lambda m: min_dist[m] if not math.isinf(min_dist[m]) else -1
                    )
                    samples.append(next_node)

                    # only recompute if there's another iteration left
                    if len(samples) < samples_per_component:
                        root_indices_numba = List(samples)
                        _, _, min_dist = propagate_transforms_multi_source_dijkstra_numba(
                            len(feats),
                            root_indices_numba,
                            edges_numba,
                            hop_penalty=config["transform_propagation"]["hop_penalty"],
                            prob_floor=1e-6,
                        )

                # append a new component as a list of samples that are well spread out
                best_central_nodes.append(samples)

    return [[all_images[i] for i in x] for x in best_central_nodes]