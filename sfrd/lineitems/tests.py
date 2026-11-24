# Test by loading a random VRDU document with at least two pages with line items, selecting one at random as the test page
# and calculating unary and pairwise potentials based on the rest.
# Raw VRDU pages are used as is (pages of one document share a template, in practice they would be aligned with SfRD first).
#
# Usage: python -m sfrd.lineitems.tests /path/to/vrdu/ad-buy-form [--seed 0] [--runs 1]

import argparse
import random

import numpy as np

from sfrd.annotations_vrdu import find_line_item_documents, add_images
from sfrd.lineitems import unary_potential, pairwise_potential, allowed_pairs


def test_potentials(dataset_dir, documents, rng):
    filename, pages = rng.choice(documents)
    pages = add_images(dataset_dir, pages)
    test_page = rng.choice(pages)
    train_pages = [page for page in pages if page is not test_page]

    # hide labels from the test page
    test_annotation = {"image": test_page["image"],
                       "boxes": [{key: box[key] for key in ("x", "y", "w", "h")} for box in test_page["boxes"]]}
    true_labels = [box["label"] for box in test_page["boxes"]]
    n = len(true_labels)

    U, labels = unary_potential(test_annotation, train_pages)
    P, pairwise_labels = pairwise_potential(test_annotation, train_pages)
    pairs = allowed_pairs(train_pages)

    # shapes, finiteness and normalization
    assert labels == pairwise_labels
    assert U.shape == (n, len(labels)) and np.all(np.isfinite(U))
    assert np.allclose(np.exp(U).sum(axis=1), 1.0)
    assert P.shape == (n, n, len(labels), len(labels))
    assert np.all(np.isneginf(P[np.arange(n), np.arange(n)]))
    for a in range(len(labels)):
        for b in range(len(labels)):
            if (labels[a], labels[b]) not in pairs:
                assert np.all(np.isneginf(P[:, :, a, b]))

    # unary: how often the most probable label is the true label (labels unseen in training can not be right)
    known = [p for p in range(n) if true_labels[p] in labels]
    unary_correct = sum(labels[np.argmax(U[p])] == true_labels[p] for p in known)

    # pairwise: for each true pair (p, q) inside a test line item, is q the best partner of p for that label pair
    index = {id(box): p for p, box in enumerate(test_page["boxes"])}
    pair_total = pair_correct = 0
    for item in test_page["line_items"]:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                a, b = item[i]["label"], item[j]["label"]
                if (a, b) not in pairs:
                    continue
                p, q = index[id(item[i])], index[id(item[j])]
                pair_total += 1
                pair_correct += np.argmax(P[p, :, labels.index(a), labels.index(b)]) == q

    print(f"{filename} page {test_page['page']} (trained on {len(train_pages)} pages): "
          f"{n} boxes, {len(test_page['line_items'])} line items, labels {labels}")
    print(f"  unary accuracy {unary_correct}/{len(known)}, pairwise best partner accuracy {pair_correct}/{pair_total}")
    return unary_correct, len(known), pair_correct, pair_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="VRDU dataset with line items, e.g. vrdu/ad-buy-form")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1, help="number of random documents to test")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    documents = find_line_item_documents(args.dataset_dir, min_pages=2)
    print(f"{len(documents)} documents with line items on at least two pages")

    totals = np.zeros(4, dtype=int)
    for _ in range(args.runs):
        totals += test_potentials(args.dataset_dir, documents, rng)

    if args.runs > 1:
        print(f"total: unary accuracy {totals[0]}/{totals[1]}, pairwise best partner accuracy {totals[2]}/{totals[3]}")


if __name__ == "__main__":
    main()
