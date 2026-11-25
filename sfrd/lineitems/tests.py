# Test by loading a random VRDU document with at least two pages with line items, selecting one at random as the test page
# and calculating unary and pairwise potentials based on the rest.
# The test page is given only its OCR units (tokens by default) and image, as at inference: its annotations are not
# passed to the potentials and are only used afterwards as ground truth for scoring.
# Training can be limited to a number of pages and line items per page, down to one page with one line item
# (the tokens of removed line items are removed from the page, as if it had fewer rows).
# test_minimal_sample() also checks every appearance mode with one line item and no other boxes (too few for PCA).
# Raw VRDU pages are used as is (pages of one document share a template, in practice they would be aligned with SfRD first).
#
# Usage: python -m sfrd.lineitems.tests /path/to/vrdu/ad-buy-form [--seed 0] [--runs 1] [--unit tokens]
#            [--appearance hog] [--train_pages N] [--train_line_items M] [--benchmark]

import argparse
import random
import warnings

import numpy as np

from sfrd.annotations_vrdu import find_line_item_documents, add_images
from sfrd.lineitems import unary_potential, pairwise_potential, allowed_pairs, BACKGROUND_LABEL


def limit_training(pages, max_pages=None, max_line_items=None):
    """Keep the first max_pages pages and their first max_line_items line items (None keeps all)"""
    limited = []
    for page in pages[:max_pages]:
        kept = page["line_items"][:max_line_items]
        kept_ids = {id(box) for item in kept for box in item}
        dropped = {id(box) for item in page["line_items"] for box in item if id(box) not in kept_ids}
        limited.append({**page, "boxes": [box for box in page["boxes"] if id(box) not in dropped], "line_items": kept})
    return limited


def check_potentials(U, labels, P, pairs, n):
    """Shapes, finiteness and normalization"""
    assert U.shape == (n, len(labels)) and np.all(np.isfinite(U))
    assert np.allclose(np.exp(U).sum(axis=1), 1.0)
    assert set(P) == pairs
    for (a, b), potential in P.items():
        assert BACKGROUND_LABEL not in (a, b)
        assert potential.shape == (n, n)


def evaluate(test_page, train_pages, unary=unary_potential, pairwise=pairwise_potential, **unary_kwargs):
    """
    Potentials for test_page from train_pages, scored against the test page annotations.
    Returns [field correct, field total, background correct, background total, pairs correct, pairs total]
    """
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

    U, labels = unary(test_annotation, train_pages, **unary_kwargs)
    P = pairwise(test_annotation, train_pages)
    check_potentials(U, labels, P, allowed_pairs(train_pages), n)

    # unary: how often the most probable label is the true label, separately for field and background units
    # (field labels never seen in training count as misses)
    predicted = [labels[np.argmax(U[p])] for p in range(n)]
    fields = [p for p in range(n) if true_labels[p] != BACKGROUND_LABEL]
    background = [p for p in range(n) if true_labels[p] == BACKGROUND_LABEL]
    field_correct = sum(predicted[p] == true_labels[p] for p in fields)
    background_correct = sum(predicted[p] == BACKGROUND_LABEL for p in background)

    # pairwise: for each true pair (p with label a, q with label b) inside a test line item, is the best scoring
    # partner of p (pairwise + unary of the partner) a unit with label b in the same line item
    # (pairs that are not allowed count as misses)
    pair_total = pair_correct = 0
    for item in test_page["line_items"]:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                a, b = item[i]["label"], item[j]["label"]
                pair_total += 1
                if (a, b) not in P:
                    continue
                p = index[id(item[i])]
                best = np.argmax(P[(a, b)][p] + U[:, labels.index(b)])
                pair_correct += true_item.get(best) == true_item[p] and true_labels[best] == b

    return np.array([field_correct, len(fields), background_correct, len(background), pair_correct, pair_total])


def format_scores(scores):
    return (f"unary accuracy: fields {scores[0]}/{scores[1]}, background {scores[2]}/{scores[3]}; "
            f"pairwise best partner accuracy {scores[4]}/{scores[5]}")


def test_potentials(dataset_dir, documents, rng, max_pages=None, max_line_items=None, appearance="hog"):
    filename, pages = rng.choice(documents)
    pages = add_images(dataset_dir, pages)
    test_page = rng.choice(pages)
    train_pages = limit_training([page for page in pages if page is not test_page], max_pages, max_line_items)

    scores = evaluate(test_page, train_pages, appearance=appearance)
    print(f"{filename} page {test_page['page']} (trained on {len(train_pages)} pages, "
          f"{sum(len(page['line_items']) for page in train_pages)} line items): {len(test_page['boxes'])} units, "
          f"{len(test_page['line_items'])} line items")
    print("  " + format_scores(scores))
    return scores


def benchmark(dataset_dir, documents, rng, runs, appearances=("hog", "pca", "none"),
              scenarios=((None, None), (1, None), (1, 1))):
    """Compare appearance modes with all / limited training (max pages, max line items) on the same random test pages"""
    cases = []
    for filename, pages in rng.sample(documents, min(runs, len(documents))):
        pages = add_images(dataset_dir, pages)
        test_page = rng.choice(pages)
        cases.append((test_page, [page for page in pages if page is not test_page]))

    print(f"benchmark on {len(cases)} documents (fraction correct: fields, background, pairwise best partner)")
    for max_pages, max_line_items in scenarios:
        for appearance in appearances:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = sum(evaluate(test_page, limit_training(train_pages, max_pages, max_line_items),
                                      appearance=appearance) for test_page, train_pages in cases)
            print(f"  pages {max_pages or 'all'}, line items {max_line_items or 'all'}, {appearance:4s}: "
                  f"{scores[0] / scores[1]:.3f} {scores[2] / scores[3]:.3f} {scores[4] / scores[5]:.3f}")


def test_minimal_sample(dataset_dir, documents, rng):
    """One training page with one line item and nothing else on it: every appearance mode must still work"""
    filename, pages = rng.choice(documents)
    pages = add_images(dataset_dir, pages[:2])
    train = limit_training(pages[:1], 1, 1)
    train[0]["boxes"] = list(train[0]["line_items"][0])  # only the boxes of the one line item
    test = {"image": pages[1]["image"], "boxes": [{key: box[key] for key in "xywh"} for box in pages[1]["boxes"]]}

    pairs = allowed_pairs(train)
    P = pairwise_potential(test, train)
    for appearance in ("hog", "pca", "none"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            U, labels = unary_potential(test, train, appearance=appearance)
        check_potentials(U, labels, P, pairs, len(test["boxes"]))
        assert (appearance == "pca") == bool(caught)  # PCA is skipped with a warning
    print(f"minimal sample ok ({filename}: {len(train[0]['boxes'])} training boxes, {len(labels)} labels)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="VRDU dataset with line items, e.g. vrdu/ad-buy-form")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1, help="number of random documents to test")
    parser.add_argument("--unit", default="tokens", help="OCR unit to classify: tokens | lines | paragraphs | blocks")
    parser.add_argument("--appearance", default="hog", help="appearance features: hog | pca | none")
    parser.add_argument("--train_pages", type=int, default=None, help="max number of training pages")
    parser.add_argument("--train_line_items", type=int, default=None, help="max number of line items per training page")
    parser.add_argument("--benchmark", action="store_true",
                        help="compare appearance modes with all training pages, 1 page, and 1 page with 1 line item")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    documents = find_line_item_documents(args.dataset_dir, min_pages=2, unit=args.unit)
    print(f"{len(documents)} documents with line items on at least two pages")

    test_minimal_sample(args.dataset_dir, documents, rng)

    if args.benchmark:
        benchmark(args.dataset_dir, documents, rng, args.runs)
        return

    totals = np.zeros(6, dtype=int)
    for _ in range(args.runs):
        totals += test_potentials(args.dataset_dir, documents, rng, args.train_pages, args.train_line_items,
                                  args.appearance)

    if args.runs > 1:
        print("total: " + format_scores(totals))


if __name__ == "__main__":
    main()
