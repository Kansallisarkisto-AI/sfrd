# Load VRDU documents (https://arxiv.org/abs/2211.15421) as line item annotations for sfrd.lineitems.
#
# Classification happens on OCR units (tokens by default, or larger units pre-identified by the OCR engine:
# lines, paragraphs or blocks), because at inference only the OCR output is available.
# One page annotation is a dict (the format used by sfrd.lineitems):
#   {"filename": str, "page": int, "image": BGR np.ndarray,
#    "boxes": [{"label": str, "x": float, "y": float, "w": float, "h": float, "text": str, "segments": list}, ...],
#    "line_items": [[box, box, ...], ...]}
# "boxes" holds every OCR unit on the page. x, y is the top left corner, coordinates are relative to page size (0..1).
# Each unit is labeled with the line item field it overlaps most (in OCR text characters), or BACKGROUND_LABEL.
# A line item holds its units in canonical field order (the order of meta.json), and in OCR order inside a field.
# Tokens of unrepeated fields (e.g. advertiser) are background, since they are not line item fields.
#
# SfRD expects pages to be aligned to a common frame beforehand. Raw pages of one VRDU document
# share a template, so they are used as is for testing.

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from .lineitems.potentials import BACKGROUND_LABEL


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


def ocr_boxes(ocr_page, unit="tokens"):
    """Unlabeled boxes of the OCR units (tokens | lines | paragraphs | blocks) of one VRDU OCR page, no annotations used"""
    boxes = []
    for ocr_unit in ocr_page[unit]:
        if not ocr_unit["text"].strip():
            continue
        _, x1, y1, x2, y2 = ocr_unit["bbox"]
        boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                      "text": ocr_unit["text"], "segments": ocr_unit["segments"]})
    return boxes


def record_to_pages(record, field_order, unit="tokens"):
    """
    Label the OCR units of one VRDU record with its line item annotations, without images.
    Returns {page_idx: {"filename", "page", "boxes", "line_items"}} for pages with at least one line item.
    Fields that are not in field_order are treated as background.
    """
    # owner[c] = index of the line item field that OCR text character c belongs to, -1 for none
    owner = np.full(len(record["ocr"]["text"]), -1)
    fields = []  # (label, line item index)
    n_items = 0
    for names, values in record["annotations"]:
        if not isinstance(names, list):  # unrepeated field, not a line item
            continue
        for item in values:  # in practice one line item per annotation entry
            for name, (_text, _bbox, segments) in zip(names, item):
                if name in field_order:
                    for start, end in segments:
                        owner[start:end] = len(fields)
                    fields.append((name, n_items))
            n_items += 1

    pages = {}
    for ocr_page in record["ocr"]["pages"]:
        boxes = ocr_boxes(ocr_page, unit)
        items = {}
        for box in boxes:
            chars = np.concatenate([owner[start:end] for start, end in box["segments"]])
            chars = chars[chars >= 0]
            if len(chars) == 0:
                box["label"] = BACKGROUND_LABEL
            else:  # the field covering most characters of the unit
                label, item_idx = fields[np.bincount(chars).argmax()]
                box["label"] = label
                items.setdefault(item_idx, []).append(box)

        if items:
            line_items = [sorted(item, key=lambda b: (field_order.index(b["label"]), b["segments"][0][0]))
                          for _, item in sorted(items.items())]
            pages[ocr_page["page_id"]] = {"filename": record["filename"], "page": ocr_page["page_id"],
                                          "boxes": boxes, "line_items": line_items}

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


def find_line_item_documents(dataset_dir, min_pages=2, unit="tokens"):
    """List (filename, pages) for documents with line items on at least min_pages pages (pages without images)"""
    field_order = line_item_fields(dataset_dir)
    documents = []
    for record in load_vrdu_records(dataset_dir):
        pages = record_to_pages(record, field_order, unit)
        if len(pages) >= min_pages:
            documents.append((record["filename"], [pages[p] for p in sorted(pages)]))
    return documents


def add_images(dataset_dir, pages, dpi=100):
    """Render the PDF page image of each page annotation into page["image"]"""
    for page in pages:
        pdf_path = Path(dataset_dir) / "main" / "pdfs" / page["filename"]
        page["image"] = render_pdf_page(pdf_path, page["page"], dpi=dpi)
    return pages
