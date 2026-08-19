from sfrd import align_pages, load_yolo_obb_labels, load_classes, \
                 invert_affine_numba, apply_affine_numba, apply_sitk_transform_to_points, \
                 config, apply_tps

from nafhtr import *
import argparse
import torch
import glob
import os
from pathlib import Path
import cv2
import lzma

import numpy as np

import polars as pl
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
import re
import pickle
from multiprocessing import get_context
from tqdm import tqdm
import json

import itertools
import time
import traceback
import psutil

from .cli_helpers import *
from .boxfit import fit_middle_axis_aligned_box, build_cost_image, \
                    solve, cells_to_polygons
from skimage.filters import threshold_otsu

# disable opencv threading because we are using multiprocessing
cv2.setNumThreads(1)


def parse_args():
    """Parse command-line arguments for the structuralization pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run structuralization")

    parser.add_argument(
        "--config",
        type=str,
        default='',
        help="Path to config file"
    )

    parser.add_argument(
        "--detection_model_path",
        type=str,
        default='/path/to/rfdetr/model.pth',
        help="Path to the detection model file"
    )
    parser.add_argument(
        "--recognition_model_path",
        type=str,
        default='/path/to/trocr/model/folder/',
        help="Path to the recognition model folder"
    )
    parser.add_argument(
        "--recognition_model_revision",
        type=str,
        default='main',
        help="Recognition model revision"
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        required=True,
        help="Directory with images to align"
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory for structured output"
    )

    parser.add_argument(
        "--annotation_directory",
        type=str,
        required=True,
        help="Directory to YOLO-OBBv8-style annotations"
    )

    # Thresholds
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--line_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small line polygons"
    )
    parser.add_argument(
        "--region_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small region polygons"
    )
    parser.add_argument(
        "--line_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--line_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--association_overlap_threshold",
        type=float,
        default=0.03,
        help=(
            "Threshold value for the minimum overlap "
            "(proportion of text line area overlapping with annotation) "
            "required for associating text lines with labels "
            "in the annotation"
        ),
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=8,
        help="Batch size for text recognition"
    )
    parser.add_argument("--disregard_regions", action="store_true",
                        help="Do not use segmentation model for text regions")
    parser.add_argument("--disregard_lines", action="store_true", 
                        help="Do not use segmentation model for text lines (or at all if also disregarding regions)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug visualizations with polygons and labels"
    )

    parser.add_argument(
        "--align_only",
        action="store_true",
        help="Align only, store in output directory."
    )

    parser.add_argument(
        "--alignment_batch_size",
        type=int,
        default=500,
        help="Batch size for page alignment"
    )

    parser.add_argument(
        "--refit_boundaries",
        action="store_true",
        help="Whether to refit cell/line boundaries, avoiding high-frequency content (e.g. written text) at cell boundaries."
    )

    parser.add_argument(
        "--refit_scale",
        type=int,
        default=4,
        help="Downscaling factor for refit_boundaries"
    )

    parser.add_argument(
        "--process_only",
        action="store_true",
        help="Process only (with GPU, after alignment), store in output directory."
    )

    parser.add_argument("--crossout_threshold", type=float, default=0.0,
                        help="Proportion of annotated box required to be filled for crossing out to be detected")
    parser.add_argument("--underline_threshold", type=float, default=0.0,
                        help="Proportion of annotated box required to be filled for underline to be detected")

    parser.add_argument(
        "--pixel_metric",
        type=str,
        choices=["mean", "black_ratio"],
        default="black_ratio",
        help="Metric to compute inside cropped boxes"
    )

    parser.add_argument(
        "--black_threshold",
        type=int,
        default=50,
        help="Pixel intensity threshold for 'black' (0–255)"
    )

    parser.add_argument(
        "--cpu_processes",
        type=int,
        default=0,
        help="Number of CPU preprocessing/postprocessing workers. Default: min(os.cpu_count(), 6 * number of GPUs)."
    )

    parser.add_argument(
        "--gpu_in_flight_limit",
        type=int,
        default=32,
        help="Maximum number of GPU requests queued/in flight at once."
    )

    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of GPU workers per GPU"
    )

    parser.add_argument(
        "--segmentation_tilesize",
        type=int,
        default=0,
        help="Tile size for segmentation (e.g. 768px), 0 to disable"
    )

    parser.add_argument(
        "--segmentation_tileoverlap",
        type=int,
        default=128,
        help="Tile overlap for segmentation (e.g. 128px), 0 to disable"
    )

    parser.add_argument(
        "--segmentation_overallsize",
        type=int,
        default=768,
        help="Overall size of image fed into segmentation (when tiling, _subtract_ segmentation_tileoverlap)"
    )

    return parser.parse_args()


def align_only(args):
    """
    Finds images and templates from given directories and performs alignment of images to templates.

    Args:
        args (argparse.Namespace): Parsed CLI arguments containing
            input/output paths and alignment settings.

    Returns:
        dict[str, dict]: Mapping of aligned image paths to alignment
        metadata including transformation matrices and alignment scores.

    """
    images_to_align = load_image_paths(args.input_directory)
    root_images = load_image_paths(
        str(Path(args.annotation_directory) / Path("images")))

    print(f"Annotated images: {len(root_images)}")
    print(f"Images to align: {len(images_to_align)}")

    batch_size = args.alignment_batch_size

    pretty_output_dict = {}
    unaligned_all = []

    batch_ranges = [
        (i, min(i + batch_size, len(images_to_align)))
        for i in range(0, len(images_to_align), batch_size)
    ]

    # If last resulting batch is less than a third of batch_size,
    # merge the last and the second to last batches.
    if (
        len(batch_ranges) > 1
        and batch_ranges[-1][1] - batch_ranges[-1][0] < batch_size * 0.33333
    ):
        print("Merged last two batches")
        # start index of second to last batch, end index of last batch
        batch_ranges[-2] = (batch_ranges[-2][0], batch_ranges[-1][1])
        batch_ranges.pop()  # remove unused last batch from list

    for batch_idx, (start_idx, end_idx) in enumerate(batch_ranges, 1):
        batch = images_to_align[start_idx:end_idx]

        print(f"Aligning batch {batch_idx} ({start_idx}:{end_idx})")

        batch_output_dict, batch_unaligned = align_pages(
            batch,
            root_images,
            args.annotation_directory
        )

        pretty_output_dict.update(batch_output_dict)
        unaligned_all.extend(batch_unaligned)

        print(
            f"Batch aligned: {len(batch_output_dict)} | "
            f"Batch unaligned: {len(batch_unaligned)}"
        )

    print(
        f"Aligned {len(pretty_output_dict.keys())} images, "
        f"{len(unaligned_all)} images left unaligned."
    )

    with open(Path(args.output_directory) / "pretty_output_dict.pickle", 'wb') as f:
        pickle.dump(pretty_output_dict, f)

    return pretty_output_dict


_GPU_REQUEST_COUNTER = itertools.count()


def gpu_worker_loop(args, rank, gpu_task_queue, gpu_results):
    """
    Run a GPU worker process for detection and recognition tasks.

    Each worker initializes OCR and detection models on a dedicated GPU
    and continuously processes queued inference requests.

    Supported task types:
    - predict_polygons
    - get_text_predictions

    Args:
        args (argparse.Namespace): Parsed runtime arguments.
        rank (int): CUDA device index assigned to this worker.
        gpu_task_queue (multiprocessing.Queue): Queue containing GPU tasks.
        gpu_results (multiprocessing.Manager.dict): Shared dictionary for
            returning inference results and exceptions.
    """
    torch.cuda.set_device(rank)
    device_string = f"cuda:{rank}"

    if args.disregard_regions and args.disregard_lines:
        detection_model = None
    else:
        detection_model = load_rfdetr_model(
            args.detection_model_path,
            batch_size=1,
            device=torch.device(device_string),
        )

    recognition_model, processor = load_trocr_model(
        args.recognition_model_path,
        args.recognition_model_path,
        device=device_string,
        revision=args.recognition_model_revision
    )

    print(f"Loading {device_string}")

    while True:
        task = gpu_task_queue.get()

        try:
            if task is None:
                return

            kind, request_id, payload = task

            if kind == "predict_polygons":
                image_path = payload["image_path"]

                result = predict_polygons(
                    detection_model,
                    image_path,
                    max_size=args.segmentation_overallsize,
                    confidence_threshold=args.confidence_threshold,
                    line_percentage_threshold=args.line_percentage_threshold,
                    region_percentage_threshold=args.region_percentage_threshold,
                    line_iou=args.line_iou,
                    region_iou=args.region_iou,
                    line_overlap_threshold=args.line_overlap_threshold,
                    region_overlap_threshold=args.region_overlap_threshold,
                    tile_size=args.segmentation_tilesize,
                    tile_overlap=args.segmentation_tileoverlap
                )

            elif kind == "get_text_predictions":
                input_data = TextPredictionInput(
                    image_path=payload["image_path"],
                    line_threshold=args.text_batch_size,
                )

                result = get_text_predictions(
                    input_data,
                    payload["ordered_lines"],
                    recognition_model,
                    processor,
                )

            else:
                raise ValueError(f"Unknown GPU task kind: {kind}")

            gpu_results[request_id] = {
                "ok": True,
                "result": result,
            }

        except Exception:
            gpu_results[request_id] = {
                "ok": False,
                "error": traceback.format_exc(),
            }

        finally:
            gpu_task_queue.task_done()


def run_gpu_task(gpu_task_queue, gpu_results, gpu_slots, kind, payload):
    """
    Submit a task to a GPU worker and wait for completion.

    Handles synchronization and bounded concurrency using semaphores.

    Args:
        gpu_task_queue (multiprocessing.Queue): Queue for GPU tasks.
        gpu_results (multiprocessing.Manager.dict): Shared result store.
        gpu_slots (multiprocessing.BoundedSemaphore): Semaphore limiting
            concurrent GPU requests.
        kind (str): Task type identifier.
        payload (dict): Task-specific input payload.

    Returns:
        Any: Result returned by the GPU worker.

    Raises:
        RuntimeError: If the GPU worker reports an exception.
    """
    request_id = f"{os.getpid()}-{next(_GPU_REQUEST_COUNTER)}"

    gpu_slots.acquire()

    try:
        gpu_task_queue.put((kind, request_id, payload))

        while True:
            result = gpu_results.pop(request_id, None)

            if result is not None:
                if result["ok"]:
                    return result["result"]

                raise RuntimeError(result["error"])

            time.sleep(0.05)

    finally:
        gpu_slots.release()

def round_polygon(poly):
    return [round(x) for x in poly]

def merge_results_parallel(results):
    """Merge outputs from parallel worker processes.

    Combines structured OCR outputs and annotation label orders into
    unified dictionaries.

    Args:
        results (list[tuple]): Iterable of worker outputs containing
            final_outputs and final_label_orders.

    Returns:
        tuple[dict, dict]:
            - Merged output records.
            - Merged annotation label ordering.
    """
    merged_outputs = {}
    merged_label_orders = {}

    for final_outputs, final_label_orders in results:
        for key, rows in final_outputs.items():
            merged_outputs.setdefault(key, []).extend(rows)

        merged_label_orders.update(final_label_orders)

    return merged_outputs, merged_label_orders


def finalize_blackout_logic_parallel(final_outputs, args):
    """
    Finalize underline and crossout classification logic.

    Computes normalized thresholded metrics for special annotation labels
    beginning with '__' or '||', then converts them into boolean flags
    indicating underline or crossout activation.

    Args:
        final_outputs (dict): Structured OCR output grouped by template.
        args (argparse.Namespace): Runtime arguments containing threshold
            configuration values.
    """

    metrics = {}

    for rows in final_outputs.values():
        for row_dict in rows:
            row = row_dict["values"]
            for k, v in row.items():
                if (isinstance(v, float)
                        and (k.startswith("||") or k.startswith("__"))
                        and not np.isnan(v)
                    ):
                    metrics.setdefault(k, []).append(v)

    thresholds = {}

    for k, vals in metrics.items():
        thresholds[k] = min_without_outliers_std(vals)

    for rows in final_outputs.values():
        for row_dict in rows:
            row = row_dict["values"]
            groups = {}

            for key in row.keys():
                if not (key.startswith("__") or key.startswith("||")):
                    continue

                if key not in thresholds:
                    continue

                row[key] = row[key] - thresholds[key]

                try:
                    group_number = re.search(r"[0-9]+", key).group()

                    if (
                        group_number in groups
                        and groups[group_number][-1].startswith(key[:2])
                    ):
                        groups[group_number].append(key)
                    else:
                        groups[group_number] = [key]

                except (IndexError, AttributeError):
                    pass

            for group_number, group_members in groups.items():
                if len(group_members) <= 1:
                    continue

                vals = [row[key] for key in group_members]

                if group_members[0].startswith("__"):
                    try:
                        max_index = np.nanargmax(vals)
                    except ValueError:
                        min_index = -1

                    for i, key in enumerate(group_members):
                        try:
                            row["CONF_" + key] = row[key]
                        except:
                            pass
                        row[key] = bool(
                            i == max_index
                            and row[key] > args.underline_threshold
                        )

                elif group_members[0].startswith("||"):
                    try:
                        min_index = np.nanargmin(vals)
                    except ValueError:
                        min_index = -1

                    for i, key in enumerate(group_members):
                        try:
                            row["CONF_" + key] = row[key]
                        except:
                            pass
                        row[key] = bool(
                            i == min_index
                            and row[key] > args.crossout_threshold
                        )


GLOBAL_CLASSES = None


def init_worker(annotation_directory, config_file):
    """
    Initialize multiprocessing worker state.

    Loads annotation class definitions into a process-global variable
    for reuse across worker tasks.

    Args:
        annotation_directory (str | Path): Directory containing
            the classes.txt annotation file.
    """
    global GLOBAL_CLASSES

    GLOBAL_CLASSES = load_classes(
        Path(annotation_directory) / "classes.txt"
    )

    if config_file:
        config.load_config(config_file)

def refit_lines(ordered_lines):
    boxes = []



def process_single_image(worker_args):
    """
    Process a single aligned image through the OCR pipeline.

    Performs:
    - Text region and line detection
    - Associates text lines with YOLO-OBB annotations
    - Text recognition
    - Crossout/underline extraction
    - Structured output generation

    Args:
        worker_args (tuple): Worker input tuple containing:
            - image path
            - alignment metadata
            - runtime arguments
            - GPU task queue
            - GPU result dictionary
            - GPU semaphore

    Returns:
        tuple[dict, dict]:
            - Structured OCR outputs grouped by template image.
            - Annotation label ordering metadata.
    """
    (
        image_path,
        alignment_result,
        args,
        gpu_task_queue,
        gpu_results,
        gpu_slots,
    ) = worker_args

    global GLOBAL_CLASSES
    classes = GLOBAL_CLASSES
    final_outputs = {}
    final_label_orders = {}

    if args.disregard_regions and args.disregard_lines:
        # create dummy line preds if we disregard all segmentation predictions
        line_preds = {
            'coords': [],
            'max_min': [],
            'confs': []
        }
        region_polygons = []
        image_shape = cv2.imread(image_path).shape[0:2]
    else:
        # predict regions and lines normally using segmentation model
        line_polygons, line_confs, line_max_mins, region_polygons, \
            region_confs, region_max_mins, image_shape = run_gpu_task(
                gpu_task_queue,
                gpu_results,
                gpu_slots,
                "predict_polygons",
                {"image_path": image_path})

        line_preds = {
            'coords': line_polygons,
            'max_min': line_max_mins,
            'confs': line_confs
        }

    if len(region_polygons) > 0 and not args.disregard_regions:
        region_preds = []

        for num, (region_polygon, region_conf, region_max_min) in enumerate(
            zip(region_polygons, region_confs, region_max_mins)
        ):
            region_preds.append({
                'coords': region_polygon,
                'id': str(num),
                'max_min': region_max_min,
                'name': 'paragraph',
                'img_shape': image_shape,
                'conf': region_conf
            })
    else:
        region_preds = get_default_region(image_shape=image_shape)

    if not args.disregard_lines:
        lines_connected_to_regions = get_line_regions(
            lines=line_preds,
            regions=region_preds
        )

        ordered_lines = order_regions_lines(
            lines=lines_connected_to_regions,
            regions=region_preds
        )
    else:
        ordered_lines = region_preds

    label_path = (
        Path(args.annotation_directory)
        / "labels"
        / (Path(alignment_result["root_path"]).stem + ".txt")
    )

    h_root, w_root = cv2.imread(
        str(alignment_result["root_path"])
    ).shape[:2]

    img = cv2.imread(image_path)

    h_page, w_page = img.shape[:2]

    annotations = load_yolo_obb_labels(
        label_path,
        w_root,
        h_root
    )

    annotation_label_order = []
    seen = set()

    annotations2 = []

    for class_ids, poly in annotations:
        label = "->".join(classes[class_id] if class_id < len(classes) else str(class_id) for class_id in class_ids)

        annotations2.append((class_ids, label, poly))

        if label not in seen:
            annotation_label_order.append(label)
            annotation_label_order.append("CONF_" + label)
            seen.add(label)
    
    annotations = annotations2

    M_root_to_page = invert_affine_numba(
        alignment_result["transformation_to_root"]
    )

    if config["thinplate"]["enabled"]:
        tps = alignment_result["post_transformation"]
        # TRANSFORM ANNOTATION POLYGONS
        # first apply affine transformation from root to page,
        # then apply B-Spline transformation (in page coordinates, nothing happens if it is None)
        annotations = [
            (
                class_ids,
                label,
                apply_tps(tps["tps"], apply_affine_numba(M_root_to_page, polygon).astype(np.float32), tps["page_shape"][1], tps["page_shape"][0])
            )
            for class_ids, label, polygon in annotations
        ]
    else:
        bspline = alignment_result["post_transformation"]

        # TRANSFORM ANNOTATION POLYGONS
        # first apply affine transformation from root to page,
        # then apply B-Spline transformation (in page coordinates, nothing happens if it is None)
        annotations = [
            (
                class_ids,
                label,
                apply_sitk_transform_to_points(
                    bspline,
                    apply_affine_numba(M_root_to_page, polygon)
                )
            )
            for class_ids, label, polygon in annotations
        ]

    if args.debug:
        debug_img = draw_debug_overlay(img, ordered_lines)

        debug_dir = Path(args.output_directory) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        out_path = debug_dir / (
            Path(image_path).stem + "_debug_pre.jpg"
        )

        cv2.imwrite(str(out_path), debug_img)

    for region in ordered_lines:
        new_lines = []
        new_line_labels = []
        new_confs = []

        for class_ids, class_label, ann_poly in annotations:
            if args.disregard_lines:
                new_lines.append(ann_poly)
                new_line_labels.append(class_label)
                new_confs.append(1.0)
            else:
                for line_index, line_poly in enumerate(region['lines']):
                    score = line_overlap_score(
                        line_poly,
                        ann_poly,
                        line_relative_threshold=args.association_overlap_threshold,
                        ann_relative_threshold=args.association_overlap_threshold,
                    )

                    if score < args.association_overlap_threshold:
                        continue

                    clipped = clip_line_to_ann_complex(line_poly, ann_poly)

                    if len(clipped) < 3:
                        continue

                    new_lines.append(clipped)
                    new_line_labels.append(class_label)
                    new_confs.append(region['line_confs'][line_index])

        region['lines'] = new_lines
        region['line_labels'] = new_line_labels
        region['line_confs'] = new_confs

    if args.refit_boundaries:
        cost_image = build_cost_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                                      downscale=args.refit_scale,
                                      peak=config['refit_boundaries']['peak'])

        for region in ordered_lines:
            # create integer-coordinate representative middle-of-side boxes
            boxes = [round_polygon(fit_middle_axis_aligned_box(x)) for x in region['lines']]
            #print(boxes)
            new_boxes, refit_info = solve(boxes, cost_image, cost_downscale=args.refit_scale,
                                          solve_downscale=args.refit_scale,
                                          workers=config['refit_boundaries']['workers'], 
                                          w_dev=config['refit_boundaries']['w_dev'],
                                          w_dev_grow=config['refit_boundaries']['w_dev_grow'],
                                          max_expand=config['refit_boundaries']['max_expand'],
                                          max_push=config['refit_boundaries']['max_push'],
                                          min_size=config['refit_boundaries']['min_size'],
                                          gap=config['refit_boundaries']['gap'],
                                          max_time=config['refit_boundaries']['max_time'])

            # replace lines with refitted axis-aligned boxes
            region['lines'] = list(cells_to_polygons(new_boxes))

    if args.debug:
        #img = cv2.imread(image_path)

        debug_img = draw_debug_overlay(img, ordered_lines)

        debug_dir = Path(args.output_directory) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        out_path = debug_dir / (
            Path(image_path).stem + "_debug.jpg"
        )

        cv2.imwrite(str(out_path), debug_img)

    if ordered_lines:
        output_values = {}
        output_polygons = {}
        output_confidences = {}
        output_conf_count = {}

        output_values["image_identifier"] = image_path.removeprefix(
            args.input_directory).removeprefix("/")

        text_predictions = run_gpu_task(
            gpu_task_queue,
            gpu_results,
            gpu_slots,
            "get_text_predictions",
            {
                "image_path": image_path,
                "ordered_lines": ordered_lines,
            },
        )

        for region_predictions, original_region in zip(
            text_predictions,
            ordered_lines
        ):
            region_predictions["line_labels"] = (
                original_region["line_labels"]
            )

            for label, recognition_result in zip(
                region_predictions["line_labels"],
                region_predictions["text_lines"]
            ):
                if label.startswith("||") or label.startswith("__"):
                    continue

                text = recognition_result["text"]
                conf = recognition_result["text_conf"]
                polygon = recognition_result["polygon"]

                conf_label = "CONF_" + label

                if label in output_values:
                    output_values[label] += "\n" + text
                else:
                    output_values[label] = text

                if label in output_polygons:
                    output_polygons[label].append(polygon)
                else:
                    output_polygons[label] = [polygon]

                if conf_label in output_values:
                    output_values[conf_label] += conf
                    output_conf_count[conf_label] += 1
                else:
                    output_values[conf_label] = conf
                    output_conf_count[conf_label] = 1

        for label in annotation_label_order:
            if label not in output_values:
                output_values[label] = None

        for key in output_conf_count.keys():
            output_values[key] = (
                output_values[key] / output_conf_count[key]
            )

        img = cv2.imread(image_path)

        # get otsu threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = threshold_otsu(gray)

        for class_ids, label, poly in annotations:
            if not (
                label.startswith("||")
                or label.startswith("__")
            ):
                continue

            mask = np.zeros(gray.shape[:2], dtype=np.uint8)

            pts = poly.astype(np.int32)

            cv2.fillPoly(mask, [pts], 255)

            x, y, w, h = cv2.boundingRect(pts)

            crop = gray[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]

            val = compute_blackout_metric(
                crop,
                mask_crop,
                thresh
            )

            output_values[label] = val

        output_key = alignment_result["root_path"]

        # construct super-dict here
        final_dict = {"values": output_values, "polygons": output_polygons,
                      "page_otsu_threshold": float(thresh),
                      "inliers_with_root": int(alignment_result["inlier_count"]),
                      "mean_residual_to_root": float(alignment_result["mean_residual"]),
                      "transformation_to_root": np.asarray(alignment_result["transformation_to_root"]).tolist()}
        # final_dict = {"values": output_values, "polygons": output_polygons}

        if output_key in final_outputs:
            final_outputs[output_key].append(final_dict)
        else:
            final_outputs[output_key] = [final_dict]

        final_label_orders[output_key] = annotation_label_order

    return final_outputs, final_label_orders


def json_default(o):
    """
    Convert NumPy arrays and single values into JSON-serializable representations.

    Args:
        o (Any): Object to serialize.

    Returns:
        Any: JSON-compatible representation.

    Raises:
        TypeError: If the object type is unsupported.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()

    if isinstance(o, np.generic):
        return o.item()

    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def main():
    """
    Run the full structuralization pipeline.

    The pipeline supports:
    - Alignment-only execution
    - Processing-only execution
    - Both alignment and processing consecutively

    Main stages include:
    - Alignment
    - Text line detection with RF-DETR and association with aligned YOLO-OBB annotation
    - Text recognition
    - Crossout (label starting with "||") and underline processing (labels starting with "__")
    - Structured export to compressed NDJSON and XLSX
    """
    args = parse_args()

    if args.config:
        config.load_config(args.config)
        print(f"Set custom config {args.config}")

    output_dir_path = Path(args.output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if args.align_only:
        align_only(args)
        return

    elif args.process_only:
        # disable threading for numpy and possibly other libraries, because we are using multiprocessing
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        with open(
            Path(args.output_directory)
            / "pretty_output_dict.pickle",
            'rb'
        ) as f:
            pretty_output_dict = pickle.load(f)
            # limit
            #pretty_output_dict = {k: pretty_output_dict[k] for k in list(pretty_output_dict.keys())[:25]}
            filtering = ["4292860141", "4292893670", "4292296198"]
            pretty_output_dict = {
                k: v
                for k, v in pretty_output_dict.items()
                if any(x in k for x in filtering)
            }

    else:
        pretty_output_dict = align_only(args)
        # disable threading for numpy and possibly other libraries, because we are using multiprocessing
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    n_gpus = torch.cuda.device_count()

    if n_gpus < 1:
        raise RuntimeError("No CUDA GPUs detected")

    cpu_processes = args.cpu_processes or min(
        psutil.cpu_count(logical=False) or 1,
        max(1, 6 * n_gpus),
    )

    print(f"{n_gpus} GPUs detected")
    print(f"Starting {cpu_processes} CPU workers")

    ctx = get_context("spawn")

    manager = ctx.Manager()

    gpu_task_queue = manager.JoinableQueue(
        maxsize=args.gpu_in_flight_limit
    )

    gpu_results = manager.dict()

    gpu_slots = manager.BoundedSemaphore(
        args.gpu_in_flight_limit
    )

    gpu_workers = [
        ctx.Process(
            target=gpu_worker_loop,
            args=(args, gpu, gpu_task_queue, gpu_results),
        )
        for gpu in range(n_gpus) for _ in range(0, args.workers_per_gpu)
    ]

    for worker in gpu_workers:
        worker.start()

    worker_args = (
        (
            image_path,
            alignment_result,
            args,
            gpu_task_queue,
            gpu_results,
            gpu_slots,
        )
        for image_path, alignment_result in pretty_output_dict.items()
    )

    results = []

    try:
        with ctx.Pool(
            processes=cpu_processes,
            initializer=init_worker,
            initargs=(args.annotation_directory, args.config),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    process_single_image,
                    worker_args,
                    chunksize=1,
                ),
                total=len(pretty_output_dict),
                desc="Processing images",
                dynamic_ncols=True,
            ):
                results.append(result)

        final_outputs, final_label_orders = merge_results_parallel(
            results
        )

    finally:
        for _ in gpu_workers:
            gpu_task_queue.put(None)

        gpu_task_queue.join()

        for worker in gpu_workers:
            worker.join()

    finalize_blackout_logic_parallel(
        final_outputs,
        args,
    )

    # sort by image id inside each output key (output table)
    for key in final_outputs:
        final_outputs[key].sort(
            key=lambda r: r["values"]["image_identifier"]
        )

    for output_key, final_dicts in final_outputs.items():
        ordered_cols = (
            ["image_identifier"]
            + final_label_orders[output_key]
        )
        # write with polygons
        with lzma.open(
            output_dir_path /
                Path(Path(output_key).stem + "_complex.ndjson.xz"),
            "wt",
            encoding="utf-8",
        ) as f:
            for row in final_dicts:
                f.write(json.dumps(row, ensure_ascii=False,
                        default=json_default) + "\n")

        with open(
            output_dir_path / Path(Path(output_key).stem + "_extrainfo.json"),
            'w',
            encoding="utf-8"
        ) as f:
            json.dump({"column_order": ordered_cols}, f, ensure_ascii=False)

        output_dicts = (x['values'] for x in final_dicts)
        df = pl.from_dicts(output_dicts, infer_schema_length=10000)

        extra_cols = [
            c for c in df.columns
            if c not in ordered_cols
        ]

        all_cols = ordered_cols + extra_cols

        df_json = df.select(all_cols)
        df_json.write_ndjson(
            output_dir_path / Path(Path(output_key).stem + ".ndjson.zst"), compression="zstd")

        with xlsxwriter.Workbook(
            output_dir_path
            / Path(Path(output_key).stem + ".xlsx"),
            {"nan_inf_to_errors": True, "strings_to_urls": False,
                "strings_to_formulas": False}
        ) as workbook:
            df2 = df.select([
                x for x in all_cols
                if not x.startswith("CONF_")
            ])

            df2.write_excel(
                workbook=workbook,
                worksheet="Simple",
                autofit=True
            )

            df = df.select(all_cols)

            df.write_excel(
                workbook=workbook,
                worksheet="Complex",
                autofit=True
            )

            worksheet = workbook.get_worksheet_by_name("Complex")

            for i in range(0, len(all_cols)):
                if all_cols[i].startswith("CONF_"):
                    worksheet.set_column(
                        i,
                        i,
                        None,
                        None,
                        {"hidden": True, "level": 1}
                    )

            n_bands = 5
            colors = make_gradient(n_bands)

            n_rows = len(df)

            for i, col in enumerate(all_cols):
                if col.startswith("CONF_") and not col.startswith("CONF_||") and not col.startswith("CONF___"):
                    target_col = i - 1

                    conf_letter = xl_col_to_name(i)
                    target_letter = xl_col_to_name(target_col)

                    try:
                        thresholds = get_thresholds_linear(n_bands)
                    except:
                        continue

                    cell_range = (
                        f"{target_letter}2:"
                        f"{target_letter}{n_rows+1}"
                    )

                    for j in range(n_bands):
                        low = f"{thresholds[j]:.6f}"
                        high = f"{thresholds[j + 1]:.6f}"

                        fmt = workbook.add_format({
                            "bg_color": colors[j]
                        })

                        if j == 0:
                            formula = (
                                f"=${conf_letter}2<{high}"
                            )

                        elif j == n_bands - 1:
                            formula = (
                                f"=${conf_letter}2>={low}"
                            )

                        else:
                            formula = (
                                f"=AND("
                                f"${conf_letter}2>={low}, "
                                f"${conf_letter}2<{high}"
                                f")"
                            )

                        worksheet.conditional_format(
                            cell_range,
                            {
                                "type": "formula",
                                "criteria": formula,
                                "format": fmt,
                            }
                        )


if __name__ == "__main__":
    main()
