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
#
# Works down to one training page with a single line item: offset spreads fall back on a prior,
# k is capped by the number of training boxes, HoG appearance features need no fitting,
# and PCA appearance features are skipped (with a warning) when there are too few training boxes.

import warnings

import cv2
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


# label of OCR units that belong to no field
BACKGROUND_LABEL = "background"

# multiplier weights for the k-NN features in unary_potential()
# x, y, w, h are relative to page size. Appearance features have roughly unit scale: HoG vectors are L2 normalized,
# PCA features are scaled to unit total variance. Only the weight of the chosen appearance mode is used.
# y is weighted down because line item rows move vertically (the table can also start at a different height on
# different pages). Position differences are ~0.01-0.1, so appearance weights are small.
# Defaults were chosen on raw VRDU ad-buy form pages (tokens, default crop window, label_balance 0.5;
# the PCA weight was chosen without label balance)
DEFAULT_KNN_WEIGHTS = {"x": 1.0, "y": 0.1, "w": 1.0, "h": 1.0, "hog": 0.4, "pca": 0.1}

# prior std of line item offsets [dx, dy] relative to page size, used when a pair has few samples.
# x: median spread of pairs with >= 5 samples in VRDU ad-buy forms (~0.015). y: spreads inside a row are tiny,
# so the prior is the min_std floor, leaving room for residual misalignment
PRIOR_OFFSET_STD = (0.015, 0.005)


def pooled_line_items(pooled_training_annotations):
    """All line items of all training pages in one list"""
    return [item for ann in pooled_training_annotations for item in ann["line_items"]]


def label_list(pooled_training_annotations):
    """Sorted labels of training boxes (including BACKGROUND_LABEL if present), the label axis of the unary potential"""
    return sorted({box["label"] for ann in pooled_training_annotations for box in ann["boxes"]})


def pairwise_offset_distributions(pooled_line_items, min_std=0.005, prior_std=PRIOR_OFFSET_STD, prior_weight=2):
    """
    Learn normal distribution of x and y offset between transitions inside line items

    pooled_line_items: list of line items, each a list of dicts with "label", "x" and "y" (relative position inside page)

    Returns: {(label_a, label_b): (mean, std)} with mean and std as arrays [dx, dy] of the offset b - a,
    for all possible pairs inside line items, including pairs of units with the same label (e.g. words of one field)
    (uses canonical order inside line item so pairs are only made in the forward direction, with two for loops, for i in range (0,n): for j in range(i+1,n))
    The fitted variance is combined with prior_std as prior_weight pseudo-samples:
    std^2 = (n * fitted_std^2 + prior_weight * prior_std^2) / (n + prior_weight),
    so a pair seen once gets a std near the prior and many samples override it. std is at least min_std.
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
        n = len(d)
        std = np.sqrt((n * np.array([std_x, std_y]) ** 2 + prior_weight * np.array(prior_std) ** 2) / (n + prior_weight))
        distributions[pair] = (np.array([mean_x, mean_y]), np.maximum(std, min_std))
    return distributions


def allowed_pairs(pooled_training_annotations):
    """
    Allowed pairs are the pairs in the original pool of line items and all pairs that drop any number of nodes. So similar two for loops, for i in range (0,n): for j in range(i+1,n)

    Returns: set of (label_a, label_b) where label_a comes before label_b in canonical order (or label_a == label_b).
    BACKGROUND_LABEL is never in a line item, so it is never in an allowed pair.
    With few training line items, pairs of fields that never appear together are not allowed.
    """
    pairs = set()
    for item in pooled_line_items(pooled_training_annotations):
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                pairs.add((item[i]["label"], item[j]["label"]))
    return pairs


def pairwise_potential(test_annotation, pooled_training_annotations, min_std=0.005,
                       prior_std=PRIOR_OFFSET_STD, prior_weight=2):
    """
    Pairwise potentials between allowed pairs in test_annotation, based on line items in pooled training annotations.
    Calculated with pairwise_offset_distributions()

    Returns: {(label_a, label_b): P} for each allowed pair, where P[p, q] is the log density of the offset from box p
    (with label_a) to box q (with label_b) in the same line item, shape (n_boxes, n_boxes) and -inf on the diagonal.
    Pairs that are not in the dict (including all pairs with BACKGROUND_LABEL) are not allowed (potential -inf).
    """
    distributions = pairwise_offset_distributions(pooled_line_items(pooled_training_annotations),
                                                  min_std, prior_std, prior_weight)

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


def box_crop(image, box, crop_box=(0.3, 0.06), size=(64, 16)):
    """
    Grayscale crop of a fixed size window centered on the box, the same size for all boxes so that the crops are
    aligned for appearance features. The window is much larger than a token (typically 0.02 x 0.01 of the page),
    so it captures context: neighbouring words, column and line above/below.
    crop_box: window (width, height) relative to page size. size: output (width, height) in pixels.
    Returns the crop resized to size, uint8 (outside the page is filled by replicating the border)
    """
    H, W = image.shape[:2]
    window = (max(1, round(crop_box[0] * W)), max(1, round(crop_box[1] * H)))
    center = ((box["x"] + box["w"] / 2) * W, (box["y"] + box["h"] / 2) * H)
    crop = cv2.getRectSubPix(image, window, center)
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def hog_features(crops, size=(64, 16), cell=8):
    """
    HoG of each crop (blocks of 2 x 2 cells, 9 orientations), L2 normalized per crop. Needs no training data.
    size must be a multiple of cell and at least 2 cells in both directions
    """
    hog = cv2.HOGDescriptor(size, (2 * cell, 2 * cell), (cell, cell), (cell, cell), 9)
    features = np.array([hog.compute(crop).ravel() for crop in crops]).reshape(len(crops), -1)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def unary_potential(test_annotation, pooled_training_annotations, k=9, weights=None, appearance="hog",
                    crop_box=(0.3, 0.06), crop_size=(64, 16), hog_cell=8, pca_components=16, min_pca_samples=32,
                    label_balance=0.5, smoothing=0.01):
    """
    Calculate unary potential for different possible labels of bounding boxes in test_annotation based on k-NN to the training annotation

    k-NN features: top left location (x, y), width and height (w, h) and appearance of a fixed size image crop around
    the box (see box_crop()), each multiplied by its weight (weights overrides DEFAULT_KNN_WEIGHTS key by key).
    appearance: "hog" (default, no fitting, works with any number of training boxes),
                "pca" (PCA fitted on training crops, skipped with a warning if there are fewer than min_pca_samples
                training boxes) or "none"
    label_balance: label prior correction, k-NN votes are divided by the label's training frequency ** label_balance
                   (0 = plain k-NN, 1 = all labels equally likely a priori). With few training line items, fields are
                   rare among training boxes compared to background (and compared to test pages with more rows)

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

    train_crops = [box_crop(image, box, crop_box, crop_size) for image, box in train]
    test_crops = [box_crop(image, box, crop_box, crop_size) for image, box in test]

    if appearance == "pca" and len(train) < min_pca_samples:
        warnings.warn(f"{len(train)} training boxes is too few for PCA (min_pca_samples={min_pca_samples}), "
                      "using position and size only")
        appearance = "none"

    if appearance == "hog":
        train_appearance = hog_features(train_crops, crop_size, hog_cell) * weights["hog"]
        test_appearance = hog_features(test_crops, crop_size, hog_cell) * weights["hog"]
    elif appearance == "pca":
        # PCA fitted on training crops, scaled so that the PCA features have unit total variance
        train_pixels = np.array([crop.ravel() / 255.0 for crop in train_crops])
        test_pixels = np.array([crop.ravel() / 255.0 for crop in test_crops])
        pca = PCA(n_components=min(pca_components, len(train))).fit(train_pixels)
        scale = weights["pca"] / np.sqrt(max(pca.explained_variance_.sum(), 1e-12))
        train_appearance = pca.transform(train_pixels) * scale
        test_appearance = pca.transform(test_pixels) * scale
    elif appearance == "none":
        train_appearance, test_appearance = np.zeros((len(train), 0)), np.zeros((len(test), 0))
    else:
        raise ValueError(f"unknown appearance mode {appearance!r}, use 'hog', 'pca' or 'none'")

    train_feats = np.hstack([geometry(train), train_appearance])
    test_feats = np.hstack([geometry(test), test_appearance])

    # k-NN turned into probabilities by weighing with inverse distance (columns of predict_proba are the sorted labels)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(train)), weights="distance")
    knn.fit(train_feats, [box["label"] for _, box in train])
    U = knn.predict_proba(test_feats)

    # label prior correction for rare labels
    counts = np.array([sum(box["label"] == label for _, box in train) for label in labels])
    U = U / counts ** label_balance
    U = U / U.sum(axis=1, keepdims=True)

    # smoothing keeps labels without neighbours possible (finite log probability)
    U = (U + smoothing) / (1 + len(labels) * smoothing)
    return np.log(U), labels
