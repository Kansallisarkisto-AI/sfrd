# Load VRDU documents (https://arxiv.org/abs/2211.15421) as line item annotations for sfrd.lineitems.
#
# One page annotation is a dict (the format used by sfrd.lineitems):
#   {"filename": str, "page": int, "image": BGR np.ndarray,
#    "boxes": [{"label": str, "x": float, "y": float, "w": float, "h": float, "text": str}, ...],
#    "line_items": [[box, box, ...], ...]}
# x, y is the top left corner and all coordinates are relative to page size (0..1).
# The boxes inside each line item are sorted into canonical field order (the order of meta.json).
#
# SfRD expects pages to be aligned to a common frame beforehand. Raw pages of one VRDU document
# share a template, so they are used as is for testing.

import gzip
import json
from pathlib import Path

import cv2
import numpy as np


def load_vrdu_records(dataset_dir):
    """Yield raw records of <dataset_dir>/main/dataset.jsonl.gz, e.g. dataset_dir = vrdu/ad-buy-form"""
    with gzip.open(Path(dataset_dir) / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def line_item_fields(dataset_dir):
    """Canonical order of line item fields, taken from meta.json"""
    with open(Path(dataset_dir) / "main" / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    patterns = meta.get("entity_appearance_pattern", {})
    return [name for name, pattern in patterns.items() if pattern == "line_item"]


def record_to_pages(record, field_order):
    """
    Turn the line item annotations of one VRDU record into page annotations without images.
    Returns {page_idx: {"filename", "page", "boxes", "line_items"}} for pages with at least one line item.
    Fields that are not in field_order are dropped.
    """
    pages = {}

    for names, values in record["annotations"]:
        if not isinstance(names, list):  # unrepeated field, not a line item
            continue

        for item in values:  # in practice one line item per annotation entry
            boxes = []
            for name, (text, (page, x1, y1, x2, y2), _segments) in zip(names, item):
                if name in field_order:
                    boxes.append({"label": name, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                                  "text": text, "page": page})

            boxes.sort(key=lambda b: field_order.index(b["label"]))

            # line items do not span pages in VRDU, but split by page just in case
            for page in sorted({b["page"] for b in boxes}):
                page_boxes = [b for b in boxes if b["page"] == page]
                ann = pages.setdefault(page, {"filename": record["filename"], "page": page,
                                              "boxes": [], "line_items": []})
                ann["boxes"].extend(page_boxes)
                ann["line_items"].append(page_boxes)

    return pages


def render_pdf_page(pdf_path, page_idx, dpi=100):
    """Render one PDF page as a BGR image (pypdfium2 is only needed for VRDU, so it is imported here)"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        rgb = np.array(pdf[page_idx].render(scale=dpi / 72).to_pil().convert("RGB"))
    finally:
        pdf.close()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_line_item_documents(dataset_dir, min_pages=2):
    """List (filename, pages) for documents with line items on at least min_pages pages (pages without images)"""
    field_order = line_item_fields(dataset_dir)
    documents = []
    for record in load_vrdu_records(dataset_dir):
        pages = record_to_pages(record, field_order)
        if len(pages) >= min_pages:
            documents.append((record["filename"], [pages[p] for p in sorted(pages)]))
    return documents


def add_images(dataset_dir, pages, dpi=100):
    """Render the PDF page image of each page annotation into page["image"]"""
    for page in pages:
        pdf_path = Path(dataset_dir) / "main" / "pdfs" / page["filename"]
        page["image"] = render_pdf_page(pdf_path, page["page"], dpi=dpi)
    return pages
