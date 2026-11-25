# Test by loading a random VRDU document with at least two pages with line items, selecting one at random as the test page
# and calculating unary and pairwise potentials based on the rest.
# The test page is given only its OCR units (tokens by default) and image, as at inference: its annotations are not
# passed to the potentials and are only used afterwards as ground truth for scoring.
# Raw VRDU pages are used as is (pages of one document share a template, in practice they would be aligned with SfRD first).
#
# Usage: python -m sfrd.lineitems.tests /path/to/vrdu/ad-buy-form [--seed 0] [--runs 1] [--unit tokens]

import argparse
import random

import numpy as np

from sfrd.annotations_vrdu import find_line_item_documents, add_images
from sfrd.lineitems import unary_potential, pairwise_potential, allowed_pairs, BACKGROUND_LABEL


def test_potentials(dataset_dir, documents, rng):
    filename, pages = rng.choice(documents)
    pages = add_images(dataset_dir, pages)
    test_page = rng.choice(pages)
    train_pages = [page for page in pages if page is not test_page]

    # inference input: OCR units and image only, annotations discarded
    test_annotation = {"image": test_page["image"],
                       "boxes": [{key: box[key] for key in ("x", "y", "w", "h", "text")} for box in test_page["boxes"]]}

    # ground truth, only for scoring
    true_labels = [box["label"] for box in test_page["boxes"]]
    true_item = {}  # box index -> line item index
    index = {id(box): p for p, box in enumerate(test_page["boxes"])}
    for item_idx, item in enumerate(test_page["line_items"]):
        for box in item:
            true_item[index[id(box)]] = item_idx
    n = len(true_labels)

    U, labels = unary_potential(test_annotation, train_pages)
    P = pairwise_potential(test_annotation, train_pages)
    pairs = allowed_pairs(train_pages)

    # shapes, finiteness and normalization
    assert U.shape == (n, len(labels)) and np.all(np.isfinite(U))
    assert np.allclose(np.exp(U).sum(axis=1), 1.0)
    assert set(P) == pairs
    for (a, b), potential in P.items():
        assert BACKGROUND_LABEL not in (a, b)
        assert potential.shape == (n, n) and np.all(np.isneginf(np.diag(potential)))

    # unary: how often the most probable label is the true label (labels unseen in training can not be right),
    # separately for field units and background units
    predicted = [labels[np.argmax(U[p])] for p in range(n)]
    fields = [p for p in range(n) if true_labels[p] in labels and true_labels[p] != BACKGROUND_LABEL]
    background = [p for p in range(n) if true_labels[p] == BACKGROUND_LABEL]
    field_correct = sum(predicted[p] == true_labels[p] for p in fields)
    background_correct = sum(predicted[p] == BACKGROUND_LABEL for p in background)

    # pairwise: for each true pair (p with label a, q with label b) inside a test line item, is the best scoring
    # partner of p (pairwise + unary of the partner) a unit with label b in the same line item
    pair_total = pair_correct = 0
    for item in test_page["line_items"]:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                a, b = item[i]["label"], item[j]["label"]
                if (a, b) not in P:
                    continue
                p = index[id(item[i])]
                best = np.argmax(P[(a, b)][p] + U[:, labels.index(b)])
                pair_total += 1
                pair_correct += true_item.get(best) == true_item[p] and true_labels[best] == b

    print(f"{filename} page {test_page['page']} (trained on {len(train_pages)} pages): "
          f"{len(fields)} field units, {len(background)} background units, {len(test_page['line_items'])} line items")
    print(f"  unary accuracy: fields {field_correct}/{len(fields)}, background {background_correct}/{len(background)}; "
          f"pairwise best partner accuracy {pair_correct}/{pair_total}")
    return field_correct, len(fields), background_correct, len(background), pair_correct, pair_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="VRDU dataset with line items, e.g. vrdu/ad-buy-form")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1, help="number of random documents to test")
    parser.add_argument("--unit", default="tokens", help="OCR unit to classify: tokens | lines | paragraphs | blocks")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    documents = find_line_item_documents(args.dataset_dir, min_pages=2, unit=args.unit)
    print(f"{len(documents)} documents with line items on at least two pages")

    totals = np.zeros(6, dtype=int)
    for _ in range(args.runs):
        totals += test_potentials(args.dataset_dir, documents, rng)

    if args.runs > 1:
        print(f"total: unary accuracy: fields {totals[0]}/{totals[1]}, background {totals[2]}/{totals[3]}; "
              f"pairwise best partner accuracy {totals[4]}/{totals[5]}")


if __name__ == "__main__":
    main()
