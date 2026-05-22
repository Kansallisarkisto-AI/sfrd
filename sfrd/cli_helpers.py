import os
import glob
import hashlib

# Shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Numpy with 1 thread per process
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

import cv2

# opencv with 1 thread per process because we are using multiprocessing
cv2.setNumThreads(1)

def load_image_paths(input_folder, extensions=None):
    """
    Glob image paths from a folder.

    Args:
        input_folder: Path to the folder containing images.
        extensions: Set or iterable of allowed file extensions.
            Defaults to common image formats.

    Returns:
        list[str]: Sorted list of image file paths.
    """
    if extensions is None:
        extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp'}

    return sorted([
        path for path in glob.glob(
            os.path.join(input_folder, '**', '*'),
            recursive=True
        )
        if os.path.isfile(path)
        and path.split('.')[-1].lower() in extensions
    ])

def draw_debug_overlay(image, regions, alpha=0.4):
    """
    Draw polygons + labels on a transparent overlay and blend with image.

    Args:
        image: Input image as a NumPy array.
        regions: Iterable of region dictionaries containing polygons
            and labels.
        alpha: Overlay transparency factor.

    Returns:
        np.ndarray: Debug visualization image with overlays applied.
    """
    overlay = image.copy()

    for region in regions:
        lines = region.get("lines", [])
        labels = region.get("line_labels", ["None"] * len(lines))

        for poly, label in zip(lines, labels):
            pts = poly.astype(np.int32)

            # random-ish but deterministic color per label
            digest = hashlib.md5(label.encode()).digest()
            color = (digest[0], digest[1], digest[2])  # hash bytes have range 0...255, perfect for colors

            # fill polygon
            cv2.fillPoly(overlay, [pts], color)

            # draw border
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

            # label position (top-left of polygon)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(
                overlay,
                label,
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # blend overlay with original
    debug_img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return debug_img

def crop_polygon_region(image, polygon):
    """
    Extracts a polygon-masked crop from an image.

    Args:
        image: Input image as a NumPy array.
        polygon: Polygon vertices as an `(N, 2)` array.

    Returns:
        tuple:
            - np.ndarray: Cropped image region.
            - np.ndarray: Corresponding cropped binary mask.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = polygon.astype(np.int32)

    cv2.fillPoly(mask, [pts], 255)

    x, y, w, h = cv2.boundingRect(pts)
    cropped_img = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    return cropped_img, cropped_mask

def compute_blackout_metric(gray, mask, thresh, method="black_ratio", dilate=True):
    """
    Computes darkness metrics inside a masked image region.

    Args:
        gray: Grayscale image.
        mask: Binary mask selecting valid pixels.
        thresh: Threshold used for black pixel classification.
        method: Metric type. Supported values are `"mean"` and
            `"black_ratio"`.
        dilate: Whether to dilate black regions before computing the
            black ratio.

    Returns:
        float: Computed blackout metric.

    Raises:
        ValueError: If an unknown metric method is requested.
    """

    pixels = gray[mask == 255]

    if len(pixels) < 5:
        return np.nan

    if method == "mean":
        return float(np.mean(pixels))
    elif method == "black_ratio":
        if dilate:
            # binarize (black = 1)
            binary = (gray < thresh).astype(np.uint8)

            # dilate black regions
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

            # Step 3: apply mask
            black_pixels = np.sum(binary[mask == 255])
            return float(black_pixels) / np.sum(mask == 255)
        else:
            # Use Otsu threshold instead of fixed threshold
            black_pixels = np.sum(pixels < thresh)
            return float(black_pixels) / len(pixels)
    else:
        raise ValueError("Unknown method")

def as_valid_polygon(poly):
    """
    Converts list of coordinates into a valid Shapely polygon.

    Args:
        poly: Polygon coordinates.

    Returns:
        shapely.geometry.Polygon: Valid polygon geometry.
    """
    g = Polygon(poly)
    if not g.is_valid:
        g = make_valid(g)
    return g

def line_overlap_score(line_poly, ann_poly, line_relative_threshold=0.25, ann_relative_threshold=0.25):
    """
    Computes overlap score between a text line polygon and annotation.

    Args:
        line_poly: Text line polygon coordinates.
        ann_poly: Annotation polygon coordinates.
        line_relative_threshold: Minimum relative overlap required for
            the line polygon.
        ann_relative_threshold: Minimum relative overlap required for
            the annotation polygon.

    Returns:
        float: Symmetric overlap score between the polygons.
    """
    line_g = as_valid_polygon(line_poly)
    ann_g = as_valid_polygon(ann_poly)

    if line_g.is_empty or ann_g.is_empty:
        return 0.0
    if not line_g.intersects(ann_g):
        return 0.0

    inter = line_g.intersection(ann_g).area
    if line_g.area <= 0:
        return 0.0

    # way 1
    #return inter / ann_g.area
    # way 2
    line_relative = inter / line_g.area
    ann_relative = inter / ann_g.area

    if line_relative < line_relative_threshold:
        line_relative = 0.0
    if ann_relative < ann_relative_threshold:
        ann_relative = 0.0

    return 0.5*line_relative + 0.5*ann_relative
    # return measure of polygon containment
    #return inter / min(line_g.area, ann_g.area)  

def clip_line_to_ann_complex(line_poly, ann_poly, extend=10000):
    """
    Clip polygon between the two side lines of YOLO-OBB annotation.

    ann_poly ordering:
        p1, p2, p3, p4

    side lines:
        left  = p4 -> p1
        right = p3 -> p2

    Args:
        line_poly: Text line polygon coordinates.
        ann_poly: YOLO-OBB annotation polygon coordinates.
        extend: Distance used to extend clipping boundary lines.

    Returns:
        np.ndarray: Clipped polygon coordinates as an `(N, 2)` array.
    """

    p1 = ann_poly[0]
    p2 = ann_poly[1]
    p3 = ann_poly[2]
    p4 = ann_poly[3]

    # Direction along the side edges
    d_left = p1 - p4
    d_left = d_left / np.linalg.norm(d_left)

    d_right = p2 - p3
    d_right = d_right / np.linalg.norm(d_right)

    # Extend both lines far beyond image
    a = p4 - d_left * extend
    b = p1 + d_left * extend

    c = p3 - d_right * extend
    d = p2 + d_right * extend

    # Region between the two infinite (/very long) lines
    clip_region = Polygon([a, b, d, c])

    poly = as_valid_polygon(line_poly)

    try:
        clipped = poly.intersection(clip_region)
    except:
        return np.empty((0, 2), dtype=np.float64)

    if clipped.is_empty:
        return np.empty((0, 2), dtype=np.float64)

    # Handle GeometryCollection
    if isinstance(clipped, GeometryCollection):
        polys = [
            g for g in clipped.geoms
            if isinstance(g, Polygon)
        ]

        if len(polys) == 0:
            return np.empty((0, 2), dtype=np.float64)

        clipped = unary_union(polys)

    # Handle MultiPolygon if needed
    if clipped.geom_type == "MultiPolygon":
        clipped = clipped.convex_hull

    coords = np.array(clipped.exterior.coords[:-1], dtype=np.float64)

    return coords

def clip_line_to_ann_simple(line_poly, ann_poly):
    """
    Clip text line to YOLO-OBB annotation, using simple x-coordinate clipping

    Args:
        line_poly: Text line polygon coordinates.
        ann_poly: YOLO-OBB annotation polygon coordinates.

    Returns:
        np.ndarray: Clipped polygon coordinates.
    """
    ann_xmin = np.min(ann_poly[:, 0])
    ann_xmax = np.max(ann_poly[:, 0])

    clipped = line_poly.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], ann_xmin, ann_xmax)

    return clipped

def interpolate_color(c1, c2, t):
    """Linearly interpolates between two RGB colors.

    Args:
        c1: Starting RGB color tuple.
        c2: Ending RGB color tuple.
        t: Interpolation parameter in the range `[0, 1]`.

    Returns:
        tuple[int, int, int]: Interpolated RGB color.
    """
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def rgb_to_hex(rgb):
    """Converts an RGB color tuple to hexadecimal format.

    Args:
        rgb: RGB color tuple.

    Returns:
        str: Hexadecimal color string.
    """
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def make_gradient(n):
    """Generates an Excel-style red-yellow-green color gradient.

    Args:
        n: Number of colors to generate.

    Returns:
        list[str]: List of hexadecimal color strings.
    """
    red = (248, 105, 107)    # Excel-ish red
    yellow = (255, 235, 132) # Excel-ish yellow
    green = (99, 190, 123)   # Excel-ish green

    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0

        if t < 0.5:
            # red → yellow
            c = interpolate_color(red, yellow, t * 2)
        else:
            # yellow → green
            c = interpolate_color(yellow, green, (t - 0.5) * 2)

        colors.append(rgb_to_hex(c))

    return colors

def get_thresholds(series, n):
    """Computes quantile thresholds from a numeric series.

    Args:
        series: Numeric series supporting quantile operations.
        n: Number of quantile intervals.

    Returns:
        list[float]: Quantile threshold values.
    """
    return [float(series.drop_nulls().drop_nans().quantile(i / n)) for i in range(n + 1)]

def get_thresholds_linear(n, start=0.85):
    """Generates evenly spaced linear thresholds.

    Args:
        n: Number of intervals.
        start: Starting threshold value.

    Returns:
        list[float]: Linearly spaced threshold values.
    """
    return [start + (1.0 - start) * (i / n) for i in range(n + 1)]

def min_without_outliers_std(vals, n_std=2.0, min_count=3):
    """Computes the minimum value after removing statistical outliers.

    Outliers are removed using a standard deviation filter.

    Args:
        vals: Iterable of numeric values.
        n_std: Standard deviation multiplier used for filtering.
        min_count: Minimum number of valid values required.

    Returns:
        float: Minimum filtered value, or NaN if insufficient data exists.
    """
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]

    if len(vals) < min_count:
        return np.nan

    mean = np.mean(vals)
    std = np.std(vals)

    if std < 1e-12:
        return float(np.min(vals))

    filtered = vals[np.abs(vals - mean) <= n_std * std]

    if len(filtered) == 0:
        return float(np.min(vals))  # fallback

    return float(np.min(filtered))