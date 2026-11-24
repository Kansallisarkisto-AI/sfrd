# Unary and pairwise potentials for assigning labels to bounding boxes and linking them into line items.
#
# Page annotation format (see also sfrd/annotations_vrdu.py), pages are assumed to be aligned with SfRD beforehand:
#   {"image": BGR np.ndarray,
#    "boxes": [{"label": str, "x": float, "y": float, "w": float, "h": float}, ...],
#    "line_items": [[box, box, ...], ...]}
# x, y is the top left corner and all coordinates are relative to page size (0..1).
# Boxes inside a line item are in canonical field order.
# A test annotation only needs "image" and "boxes" (box labels are not used).
#
# Potentials are log probabilities (or log densities), indexed by the labels returned by label_list().

import cv2
import numpy as np
from sklearn.decomposition import PCA


# multiplier weights for the k-NN features in unary_potential()
# x, y, w, h are relative to page size, PCA features are scaled to unit total variance before weighting.
# Position differences between fields are ~0.01-0.1, so the PCA weight is small: on aligned pages geometry is the
# strongest cue (on raw VRDU pages higher PCA weights reduced accuracy), appearance mostly breaks ties
DEFAULT_KNN_WEIGHTS = {"x": 1.0, "y": 1.0, "w": 1.0, "h": 1.0, "pca": 0.05}


def pooled_line_items(pooled_training_annotations):
    """All line items of all training pages in one list"""
    return [item for ann in pooled_training_annotations for item in ann["line_items"]]


def label_list(pooled_training_annotations):
    """Sorted labels found in training line items, defines the label axes of the potentials"""
    return sorted({box["label"] for item in pooled_line_items(pooled_training_annotations) for box in item})


def pairwise_offset_distributions(pooled_line_items, min_std=0.005):
    """
    Learn normal distribution of x and y offset between transitions inside line items

    pooled_line_items: list of line items, each a list of dicts with "label", "x" and "y" (relative position inside page)

    Returns: {(label_a, label_b): (mean, std)} with mean and std as arrays [dx, dy] of the offset b - a,
    for all possible pairs inside line items
    (uses canonical order inside line item so pairs are only made in the forward direction, with two for loops, for i in range (0,n): for j in range(i+1,n))
    std is at least min_std, so pairs seen only once (or always at the same offset) still get a usable distribution
    """
    offsets = {}
    for item in pooled_line_items:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                a, b = item[i], item[j]
                offsets.setdefault((a["label"], b["label"]), []).append((b["x"] - a["x"], b["y"] - a["y"]))

    return {pair: (np.mean(d, axis=0), np.maximum(np.std(d, axis=0), min_std))
            for pair, d in offsets.items()}


def allowed_pairs(pooled_training_annotations):
    """
    Allowed pairs are the pairs in the original pool of line items and all pairs that drop any number of nodes. So similar two for loops, for i in range (0,n): for j in range(i+1,n)

    Returns: set of (label_a, label_b) where label_a comes before label_b in canonical order
    """
    pairs = set()
    for item in pooled_line_items(pooled_training_annotations):
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                pairs.add((item[i]["label"], item[j]["label"]))
    return pairs


def pairwise_potential(test_annotation, pooled_training_annotations, min_std=0.005):
    """
    Pairwise potentials between allowed pairs in test_annotation, based on line items in pooled training annotations.
    Calculated with pairwise_offset_distributions()

    Returns: (P, labels) where P[p, q, a, b] is the log density of the offset from box p (with label a) to box q (with label b)
    in the same line item. Shape (n_boxes, n_boxes, n_labels, n_labels), -inf for p == q and for pairs that are not allowed.
    """
    labels = label_list(pooled_training_annotations)
    distributions = pairwise_offset_distributions(pooled_line_items(pooled_training_annotations), min_std)

    boxes = test_annotation["boxes"]
    n = len(boxes)
    xy = np.array([[box["x"], box["y"]] for box in boxes]).reshape(n, 2)
    offsets = xy[None, :, :] - xy[:, None, :]  # offsets[p, q] = position of q - position of p

    P = np.full((n, n, len(labels), len(labels)), -np.inf)
    for a, b in allowed_pairs(pooled_training_annotations):
        mean, std = distributions[(a, b)]
        z = (offsets - mean) / std
        P[:, :, labels.index(a), labels.index(b)] = np.sum(-0.5 * z ** 2 - np.log(std * np.sqrt(2 * np.pi)), axis=-1)

    P[np.arange(n), np.arange(n)] = -np.inf  # a box can not pair with itself
    return P, labels


def box_crop(image, box, context=1.0, size=(64, 16)):
    """
    Grayscale crop of box (extended by context * box size on each side to capture some context),
    resized to size (width, height) and flattened, values 0..1
    """
    H, W = image.shape[:2]
    x0 = int(np.clip((box["x"] - context * box["w"]) * W, 0, W - 1))
    y0 = int(np.clip((box["y"] - context * box["h"]) * H, 0, H - 1))
    x1 = int(np.clip((box["x"] + (1 + context) * box["w"]) * W, x0 + 1, W))
    y1 = int(np.clip((box["y"] + (1 + context) * box["h"]) * H, y0 + 1, H))

    crop = image[y0:y1, x0:x1]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA).ravel() / 255.0


def unary_potential(test_annotation, pooled_training_annotations, k=9, weights=None,
                    context=1.0, crop_size=(64, 16), pca_components=16, smoothing=0.01):
    """
    Calculate unary potential for different possible labels of bounding boxes in test_annotation based on k-NN to the training annotation

    k-NN features: top left location (x, y), width and height (w, h) and the PCA of the box image crop (with context),
    each multiplied by its weight (weights overrides DEFAULT_KNN_WEIGHTS key by key)

    Returns: (U, labels) where U[p, a] is the log probability of box p having label a. Shape (n_boxes, n_labels)
    """
    weights = {**DEFAULT_KNN_WEIGHTS, **(weights or {})}
    labels = label_list(pooled_training_annotations)

    # training boxes are the labeled boxes inside line items
    train = [(ann["image"], box) for ann in pooled_training_annotations for item in ann["line_items"] for box in item]
    test = [(test_annotation["image"], box) for box in test_annotation["boxes"]]
    if not test:
        return np.zeros((0, len(labels))), labels

    def geometry(boxes):
        return np.array([[box["x"] * weights["x"], box["y"] * weights["y"],
                          box["w"] * weights["w"], box["h"] * weights["h"]] for _, box in boxes])

    def crops(boxes):
        return np.array([box_crop(image, box, context, crop_size) for image, box in boxes])

    # PCA fitted on training crops, scaled so that the PCA features have unit total variance
    pca = PCA(n_components=min(pca_components, len(train))).fit(crops(train))
    scale = weights["pca"] / np.sqrt(max(pca.explained_variance_.sum(), 1e-12))

    train_feats = np.hstack([geometry(train), pca.transform(crops(train)) * scale])
    test_feats = np.hstack([geometry(test), pca.transform(crops(test)) * scale])
    train_labels = np.array([labels.index(box["label"]) for _, box in train])

    # k-NN turned into log probabilities by weighing with distance
    k = min(k, len(train))
    distances = np.linalg.norm(test_feats[:, None, :] - train_feats[None, :, :], axis=-1)
    nearest = np.argsort(distances, axis=1)[:, :k]

    U = np.zeros((len(test), len(labels)))
    for p in range(len(test)):
        w = 1.0 / (distances[p, nearest[p]] + 1e-6)  # inverse distance weights
        np.add.at(U[p], train_labels[nearest[p]], w / w.sum())

    # smoothing keeps labels without neighbours possible (finite log probability)
    U = (U + smoothing) / (1 + len(labels) * smoothing)
    return np.log(U), labels
