# Unary and pairwise potentials for labeling OCR units (tokens, or larger units identified by the OCR engine)
# and linking them into line items.
#
# Page annotation format (see also sfrd/annotations_vrdu.py), pages are assumed to be aligned with SfRD beforehand:
#   {"image": BGR np.ndarray,
#    "boxes": [{"label": str, "x": float, "y": float, "w": float, "h": float}, ...],
#    "line_items": [[box, box, ...], ...]}
# "boxes" holds every OCR unit on the page, units belonging to no field are labeled BACKGROUND_LABEL.
# x, y is the top left corner and all coordinates are relative to page size (0..1).
# Boxes inside a line item are in canonical field order (and in reading order inside a field).
# A test annotation is only the OCR output: "image" and unlabeled "boxes".
#
# Potentials are log probabilities (or log densities).

import cv2
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


# label of OCR units that belong to no field
BACKGROUND_LABEL = "background"

# multiplier weights for the k-NN features in unary_potential()
# x, y, w, h are relative to page size, PCA features are scaled to unit total variance before weighting.
# y is weighted down because line item rows move vertically (the table can also start at a different height on
# different pages). Position differences are ~0.01-0.1, so the PCA weight is small, appearance mostly breaks ties.
# Defaults were chosen on raw VRDU ad-buy form pages (tokens): field accuracy 0.89, background accuracy 0.99
DEFAULT_KNN_WEIGHTS = {"x": 1.0, "y": 0.1, "w": 1.0, "h": 1.0, "pca": 0.05}


def pooled_line_items(pooled_training_annotations):
    """All line items of all training pages in one list"""
    return [item for ann in pooled_training_annotations for item in ann["line_items"]]


def label_list(pooled_training_annotations):
    """Sorted labels of training boxes (including BACKGROUND_LABEL if present), the label axis of the unary potential"""
    return sorted({box["label"] for ann in pooled_training_annotations for box in ann["boxes"]})


def pairwise_offset_distributions(pooled_line_items, min_std=0.005):
    """
    Learn normal distribution of x and y offset between transitions inside line items

    pooled_line_items: list of line items, each a list of dicts with "label", "x" and "y" (relative position inside page)

    Returns: {(label_a, label_b): (mean, std)} with mean and std as arrays [dx, dy] of the offset b - a,
    for all possible pairs inside line items, including pairs of units with the same label (e.g. words of one field)
    (uses canonical order inside line item so pairs are only made in the forward direction, with two for loops, for i in range (0,n): for j in range(i+1,n))
    std is at least min_std, so pairs seen only once (or always at the same offset) still get a usable distribution
    """
    offsets = {}
    for item in pooled_line_items:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                a, b = item[i], item[j]
                offsets.setdefault((a["label"], b["label"]), []).append((b["x"] - a["x"], b["y"] - a["y"]))

    distributions = {}
    for pair, d in offsets.items():
        d = np.array(d)
        (mean_x, std_x), (mean_y, std_y) = norm.fit(d[:, 0]), norm.fit(d[:, 1])  # maximum likelihood fit per axis
        distributions[pair] = (np.array([mean_x, mean_y]), np.maximum([std_x, std_y], min_std))
    return distributions


def allowed_pairs(pooled_training_annotations):
    """
    Allowed pairs are the pairs in the original pool of line items and all pairs that drop any number of nodes. So similar two for loops, for i in range (0,n): for j in range(i+1,n)

    Returns: set of (label_a, label_b) where label_a comes before label_b in canonical order (or label_a == label_b).
    BACKGROUND_LABEL is never in a line item, so it is never in an allowed pair.
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

    Returns: {(label_a, label_b): P} for each allowed pair, where P[p, q] is the log density of the offset from box p
    (with label_a) to box q (with label_b) in the same line item, shape (n_boxes, n_boxes) and -inf on the diagonal.
    Pairs that are not in the dict (including all pairs with BACKGROUND_LABEL) are not allowed (potential -inf).
    """
    distributions = pairwise_offset_distributions(pooled_line_items(pooled_training_annotations), min_std)

    boxes = test_annotation["boxes"]
    n = len(boxes)
    xy = np.array([[box["x"], box["y"]] for box in boxes]).reshape(n, 2)
    offsets = xy[None, :, :] - xy[:, None, :]  # offsets[p, q] = position of q - position of p

    potentials = {}
    for a, b in allowed_pairs(pooled_training_annotations):
        mean, std = distributions[(a, b)]
        P = norm.logpdf(offsets, loc=mean, scale=std).sum(axis=-1).astype(np.float32)
        np.fill_diagonal(P, -np.inf)  # a box can not pair with itself
        potentials[(a, b)] = P
    return potentials


def box_crop(image, box, context=1.0, size=(32, 16)):
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
                    context=1.0, crop_size=(32, 16), pca_components=16, smoothing=0.01):
    """
    Calculate unary potential for different possible labels of bounding boxes in test_annotation based on k-NN to the training annotation

    k-NN features: top left location (x, y), width and height (w, h) and the PCA of the box image crop (with context),
    each multiplied by its weight (weights overrides DEFAULT_KNN_WEIGHTS key by key)

    Returns: (U, labels) where U[p, a] is the log probability of box p having label labels[a], shape (n_boxes, n_labels)
    """
    weights = {**DEFAULT_KNN_WEIGHTS, **(weights or {})}
    labels = label_list(pooled_training_annotations)

    # training boxes are all labeled boxes of the training pages, including background
    train = [(ann["image"], box) for ann in pooled_training_annotations for box in ann["boxes"]]
    test = [(test_annotation["image"], box) for box in test_annotation["boxes"]]
    if not test:
        return np.zeros((0, len(labels))), labels

    def geometry(boxes):
        return np.array([[box["x"] * weights["x"], box["y"] * weights["y"],
                          box["w"] * weights["w"], box["h"] * weights["h"]] for _, box in boxes])

    def crops(boxes):
        return np.array([box_crop(image, box, context, crop_size) for image, box in boxes])

    # PCA fitted on training crops, scaled so that the PCA features have unit total variance
    train_crops = crops(train)
    pca = PCA(n_components=min(pca_components, len(train))).fit(train_crops)
    scale = weights["pca"] / np.sqrt(max(pca.explained_variance_.sum(), 1e-12))

    train_feats = np.hstack([geometry(train), pca.transform(train_crops) * scale])
    test_feats = np.hstack([geometry(test), pca.transform(crops(test)) * scale])

    # k-NN turned into probabilities by weighing with inverse distance (columns of predict_proba are the sorted labels)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(train)), weights="distance")
    knn.fit(train_feats, [box["label"] for _, box in train])
    U = knn.predict_proba(test_feats)

    # smoothing keeps labels without neighbours possible (finite log probability)
    U = (U + smoothing) / (1 + len(labels) * smoothing)
    return np.log(U), labels
