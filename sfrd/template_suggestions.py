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

def suggest_templates(all_images: typing.List[Path], image_count_threshold=2):
    n = len(all_images)

    neighbor_pairs = NeighborPairs(n, 0)

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

            for node in component:
                root_indices_numba = List([node])  # numba typed list

                # try using node as root
                T_all, owner, dist = propagate_transforms_multi_source_dijkstra_numba(
                    len(feats),
                    root_indices_numba,  # _numba
                    edges_numba,  # _numba
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

            if best_node is not None:
                best_central_nodes.append(best_node)

    return [all_images[i] for i in best_central_nodes]