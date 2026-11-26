import numpy as np
from ortools.sat.python import cp_model

from ..unionfind import UnionFind
from .potentials import BACKGROUND_LABEL


def fit_model(unary_potentials, pairwise_potentials, unary_weight=1.0, pairwise_weight=0.01, pair_top_k=50,
              min_item_fields=0, dropped_penalty=1.0):
    """
    Label every box with exactly one label (a field label or BACKGROUND_LABEL), maximizing the weighted unary
    potentials of the chosen labels plus the weighted pairwise potentials of the chosen edges.

    Labels are of three kinds:
    - line item labels: the labels in pairwise_potentials. A box can only have one if it is connected by an edge.
      An edge from box p to box q with label pair (a, b) gives p the label a and q the label b.
    - other field labels: fields that are not part of line items. They have their own unary potential, need no edge
      and can be given to any number of boxes.
    - BACKGROUND_LABEL: OCR units that belong to no field (added with potential 0 if it is not in labels).
    Line items are the connected components of the chosen edges, so any number of line items (and any number of
    boxes per field, e.g. several tokens) can be found. There is no transitivity: two boxes of a line item
    do not have to be connected by a direct edge, a path of edges is enough.

    unary_potentials: (U, labels) from unary_potential()
    pairwise_potentials: {(label_a, label_b): P} from pairwise_potential(). Only these label pairs can be edges.
    The potentials are filtered first to keep the model small:
    pair_top_k: every box can only be in its pair_top_k edges with the highest pairwise potential (edges between
    boxes and labels, with a finite potential). An edge is possible if it is in the top pair_top_k of either of
    its boxes.
    min_item_fields: smallest number of distinct fields in a training line item, from min_line_item_fields().
    dropped_penalty: cost of every field missing below min_item_fields. It is counted at every box of a line item:
    a box with a line item label should be linked by chosen edges directly to boxes of min_item_fields - 1 other
    fields (only direct edges count, not paths). 0 (or min_item_fields <= 1) turns the penalty off.
    Returns: (label of every box, line items as lists of box indices, total value)
    """
    U, labels = unary_potentials
    n = len(U)
    index = {label: a for a, label in enumerate(labels)}
    line_labels = {index[label] for pair in pairwise_potentials for label in pair}

    # filter the edges: (p, q, a, b, potential) means box p has label a and box q has label b.
    # For each label pair keep the pair_top_k best edges of every row and of every column, then the pair_top_k best
    # edges of every box over all label pairs
    edges = []
    for (label_a, label_b), P in pairwise_potentials.items():
        a, b = index[label_a], index[label_b]
        scores = P.astype(float)
        scores[np.arange(n)[:, None] == np.arange(n)[None, :]] = -np.inf  # no edge from a box to itself
        keep = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(keep, np.argsort(-scores, axis=1)[:, :pair_top_k], True, axis=1)
        np.put_along_axis(keep, np.argsort(-scores, axis=0)[:pair_top_k, :], True, axis=0)
        for i, j in zip(*np.nonzero(keep & np.isfinite(scores))):
            edges.append((int(i), int(j), a, b, scores[i, j]))
    seen = np.zeros(n, dtype=int)  # number of better edges seen at each box
    kept = []
    for edge in sorted(edges, key=lambda edge: -edge[4]):
        p, q = edge[:2]
        if seen[p] < pair_top_k or seen[q] < pair_top_k:
            kept.append(edge)
        seen[p] += 1
        seen[q] += 1
    edges = kept

    model = cp_model.CpModel()

    # box_labels[p, a]: box p has label labels[a]. Every box has exactly one label
    box_labels = {(p, a): model.new_bool_var(f"box{p}_{labels[a]}") for p in range(n) for a in range(len(labels))}
    for p in range(n):
        model.add_exactly_one(box_labels[p, a] for a in range(len(labels)))

    # an edge gives both of its boxes their labels, and a box is only given a line item label by an edge
    edge_vars = [model.new_bool_var(f"edge{p}_{q}_{labels[a]}_{labels[b]}") for p, q, a, b, _ in edges]
    incident = {key: [] for key in box_labels if key[1] in line_labels}
    for var, (p, q, a, b, _) in zip(edge_vars, edges):
        model.add_implication(var, box_labels[p, a])
        model.add_implication(var, box_labels[q, b])
        incident[p, a].append(var)
        incident[q, b].append(var)
    for key, vars in incident.items():
        model.add_bool_or(vars + [box_labels[key].Not()])

    # dropped fields: has[(p, a)] holds a bool for each other field b that is linked to box p with label a by a chosen
    # edge, dropped = max(0, min_item_fields - 1 - number of linked fields) when p has label a, otherwise 0
    dropped = []
    if min_item_fields > 1 and dropped_penalty:
        neighbour_edges = {key: {} for key in incident}  # (p, a) -> {b: edges linking p to a box with label b}
        for var, (p, q, a, b, _) in zip(edge_vars, edges):
            if a != b:  # edges inside one field (several tokens) do not add a field
                neighbour_edges[p, a].setdefault(b, []).append(var)
                neighbour_edges[q, b].setdefault(a, []).append(var)
        for (p, a), by_field in neighbour_edges.items():
            linked = []
            for b, vars in by_field.items():
                has = model.new_bool_var(f"linked{p}_{labels[a]}_{labels[b]}")
                model.add_bool_or(vars).only_enforce_if(has)
                linked.append(has)
            d = model.new_int_var(0, min_item_fields - 1, f"dropped{p}_{labels[a]}")
            model.add(d >= (min_item_fields - 1) * box_labels[p, a] - sum(linked))
            dropped.append(d)

    accumulated_unary_potential = sum(float(unary_weight * U[p, a]) * var for (p, a), var in box_labels.items())
    accumulated_pairwise_potential = sum(float(pairwise_weight * potential) * var
                                         for var, (_, _, _, _, potential) in zip(edge_vars, edges))
    model.maximize(accumulated_unary_potential + accumulated_pairwise_potential - dropped_penalty * sum(dropped))

    solver = cp_model.CpSolver()
    solver.solve(model)

    result = [next(labels[a] for a in range(len(labels)) if solver.value(box_labels[p, a])) for p in range(n)]

    # line items: connected components of the chosen edges
    uf = UnionFind()
    for var, (p, q, _, _, _) in zip(edge_vars, edges):
        if solver.value(var):
            uf.union(p, q)
    components = {}
    for p in uf.p:
        components.setdefault(uf.find(p), []).append(p)
    line_items = sorted(sorted(component) for component in components.values())

    return result, line_items, solver.objective_value