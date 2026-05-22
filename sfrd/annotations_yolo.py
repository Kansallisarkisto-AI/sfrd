import numpy as np
import cv2
from pathlib import Path
import colorsys
import unicodedata

from .transforms import apply_affine_numba, invert_affine_numba
from .config import config


def load_classes(classes_path):
    with open(classes_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_yolo_obb_labels(label_path, img_w, img_h):
    """
    Returns list of (class_id, polygon[4x2]) in PIXEL coords
    """
    objects = []

    if not label_path.exists():
        raise IOError(f"Label path does not exist: {label_path}")

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                raise ValueError(f"Number of items per line differs from 9,\
                                  is the file in YOLOv8-OBB format?\nLabel path: {label_path}")

            class_id = int(parts[0])

            coords = np.array(list(map(float, parts[1:])), dtype=np.float64)
            poly = coords.reshape(4, 2)

            # ---- denormalize ----
            poly[:, 0] *= img_w
            poly[:, 1] *= img_h

            objects.append((class_id, poly))

    return objects


def to_ascii(text):
    return unicodedata.normalize('NFKD', text) \
        .encode('ascii', 'ignore') \
        .decode('ascii')


def draw_polygons_with_labels(image, polygons, labels, colors=None, thickness=2):
    for i, (poly, label) in enumerate(zip(polygons, labels)):
        pts = poly.astype(np.int32).reshape((-1, 1, 2))

        color = colors[i] if colors is not None else (0, 255, 0)

        cv2.polylines(image, [pts], isClosed=True,
                      color=color, thickness=thickness)

        cx, cy = poly.mean(axis=0)

        cv2.putText(
            image,
            to_ascii(label),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    return image


def color_from_id(cid):
    np.random.seed(cid)
    h = np.random.rand()
    s = 0.6 + 0.4 * np.random.rand()   # high saturation
    v = 0.8 + 0.2 * np.random.rand()   # high brightness
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def transform_yolo_obb_to_pages(
    dataset_dir,   # contains images/, labels/, classes.txt
    T_q,
    all_images,
    output_dir="aligned"
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    classes = load_classes(dataset_dir / "classes.txt")

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    # map root_idx to label file
    image_name_to_label = {
        p.stem: labels_dir / f"{p.stem}.txt"
        for p in images_dir.glob("*")
    }

    output_pages = {}

    for page_idx, (root_idx, M_full, inlier_count, page_path, reprojection_error) in T_q.items():
        img_path = all_images[page_idx]

        # find corresponding root label file
        root_img_path = all_images[root_idx]
        root_name = Path(root_img_path).stem
        label_path = image_name_to_label.get(root_name)

        if label_path is None:
            print(f"[WARN] No label file for {root_name}")
            continue

        h_root, w_root = cv2.imread(str(root_img_path)).shape[:2]

        objects = load_yolo_obb_labels(label_path, w_root, h_root)

        if not objects:
            continue

        # invert transform
        M_root_to_page = invert_affine_numba(M_full)

        transformed_polygons = []
        labels = []
        colors = []

        for class_id, poly in objects:
            poly_t = apply_affine_numba(M_root_to_page, poly)

            transformed_polygons.append(poly_t)

            # label text
            if class_id < len(classes):
                label = classes[class_id]
            else:
                label = str(class_id)

            labels.append(label)
            colors.append(color_from_id(class_id))

        
        # optional debug draw
        if config["debug"]["enabled"]:
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"[WARN] Could not read {img_path}")
                continue

            img_drawn = draw_polygons_with_labels(
                img.copy(),
                transformed_polygons,
                labels,
                colors
            )

            cv2.putText(
                img_drawn,
                str(page_idx),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

            # save
            out_path = output_dir / Path(img_path).name
            if not cv2.imwrite(str(out_path), img_drawn):
                raise IOError("cv2.imwrite error")
        
        output_pages[img_path] = {"labels": labels, "transformed_polygons": transformed_polygons}

    return output_pages