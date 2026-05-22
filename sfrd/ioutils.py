import cv2
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def _scale_smaller_dimension(h: int, w: int, max_dim: int | None) -> float:
    '''
    Scale the smaller of height and width to max_dim
    '''
    if max_dim is None:
        return 1.0
    m = min(h, w)  # scale the smaller of height and width to max_dim
    if m <= 0:
        return 1.0
    scale = float(max_dim) / float(m)
    return scale if scale < 1.0 else 1.0


def read_image_cv(path: Path, max_dim: int | None = None, return_scale: bool = False):
    '''
    Read image and downscale
    '''
    # np.fromfile seems to be faster than providing a path
    img = cv2.imdecode(np.fromfile(
        str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    s = _scale_smaller_dimension(h, w, max_dim)

    if s < 1.0:  # interpolation=cv2.INTER_AREA is especially suitable for downscaling
        img = cv2.resize(
            img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)

    if return_scale:
        return img, s
    return img


def tokenize(text: str):
    cur = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                yield "".join(cur)
                cur = []
    if cur:
        yield "".join(cur)


def parse_page_xml(xml_path: Path, bbox_max: int):
    """Original: returns BOW + list of normalized bboxes (x1,y1,x2,y2) in page coords."""
    if not xml_path.exists():
        return Counter(), []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return Counter(), []

    page_el = root.find(".//p:Page", PAGE_NS)
    w = float(page_el.get("imageWidth", "1")) if page_el is not None else 1.0
    h = float(page_el.get("imageHeight", "1")) if page_el is not None else 1.0
    w = max(w, 1.0)
    h = max(h, 1.0)

    tok_counter = Counter()
    for uni in root.findall(".//p:TextEquiv/p:Unicode", PAGE_NS):
        if uni.text:
            for t in tokenize(uni.text):
                tok_counter[t] += 1

    items = []

    def add_coords_elements(elems):
        for el in elems:
            coords = el.find("./p:Coords", PAGE_NS)
            if coords is None:
                continue
            pts = coords.get("points", "")
            if not pts:
                continue

            xs, ys = [], []
            for pair in pts.split():
                if "," not in pair:
                    continue
                try:
                    x_str, y_str = pair.split(",", 1)
                    xs.append(float(x_str))
                    ys.append(float(y_str))
                except Exception:
                    continue
            if not xs or not ys:
                continue

            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            ro = el.get("custom", "") or ""
            idx = None
            if "readingOrder" in ro and "index:" in ro:
                try:
                    idx_part = ro.split("index:", 1)[1]
                    idx_num = ""
                    for ch in idx_part:
                        if ch.isdigit():
                            idx_num += ch
                        else:
                            break
                    idx = int(idx_num) if idx_num else None
                except Exception:
                    idx = None

            sort_key = (idx if idx is not None else 10**9, y1, x1)
            items.append((sort_key, (x1 / w, y1 / h, x2 / w, y2 / h)))

    textlines = root.findall(".//p:TextLine", PAGE_NS)
    if textlines:
        add_coords_elements(textlines)
    else:
        textregions = root.findall(".//p:TextRegion", PAGE_NS)
        add_coords_elements(textregions)

    items.sort(key=lambda t: t[0])
    boxes = [b for _, b in items]
    boxes = boxes[:bbox_max] if bbox_max is not None else boxes
    return tok_counter, boxes
