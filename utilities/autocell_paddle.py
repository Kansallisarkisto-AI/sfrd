from typing import List, Sequence

from cvat_sdk import make_client
import cvat_sdk.models as models
import cvat_sdk.auto_annotation as cvataa
import PIL.Image

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import math

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from paddlex import create_model

from collections import defaultdict, deque

class GraphBasedOrdering:
    """
    Graph-Based Reading Order Algorithm
    
    Determines the reading order of polygons by:
    1. Comparing each pair of polygons to determine precedence
    2. Building a directed graph of these relationships
    3. Finding a topological ordering of the graph
    
    Args:
        text_direction (str): Reading direction, either 'lr' (left-to-right) or 'rl' (right-to-left). Default: 'lr'
    """
    def __init__(self, text_direction='lr'):
        self.text_direction = text_direction
    
    def _get_features(self, line):
        """
        Extract spatial features from a text line's bounding box.
        
        Args:
            line (list or tuple): Bounding box coordinates in format:
                                 [x_min, y_min, x_max, y_max]
                                 where:
                                   - x_min: leftmost x-coordinate
                                   - x_max: rightmost x-coordinate  
                                   - y_min: topmost y-coordinate
                                   - y_max: bottommost y-coordinate
        
        Returns:
            dict: Dictionary containing extracted features
        """
        x_min, y_min, x_max, y_max = line

        return {
            'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
            'anchor': (x_min, y_min), # use top left corner as anchor
            'x_min': x_min, 
            'x_max': x_max,
            'y_min': y_min, 
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def _should_precede(self, u_feat, v_feat):
        """"
        Determine if line u should come before line v in reading order.
        
        The logic follows natural reading patterns:
        1. If lines are on the same row (vertical overlap) -> use horizontal order
        2. If lines are on different rows -> use vertical order (top to bottom)
        
        Args:
            u_feat (dict): Feature dictionary for line u (from _get_features)
                          Must contain: 'center', 'y_min', 'y_max', 'height'
            v_feat (dict): Feature dictionary for line v (from _get_features)
                          Must contain: 'center', 'y_min', 'y_max', 'height'
        
        Returns:
            bool: True if line u should come before line v in reading order
                  False otherwise
        """
        u_anchor = u_feat['anchor']
        v_anchor = v_feat['anchor']
        
        # Vertical overlap threshold
        v_overlap = min(u_feat['y_max'], v_feat['y_max']) - max(u_feat['y_min'], v_feat['y_min'])
        avg_height = (u_feat['height'] + v_feat['height']) / 2
        
        # If significant vertical overlap, use horizontal order
        if v_overlap > 0.5 * avg_height:
            if self.text_direction == 'lr':
                return u_anchor[0] < v_anchor[0]
            else:
                return u_anchor[0] > v_anchor[0]
        
        # Otherwise, use vertical order (top to bottom)
        return u_anchor[1] < v_anchor[1]
    
    def order(self, lines):
        """
        Compute the reading order of text lines using graph-based approach.
        
        Args:
            lines (list): List of bounding boxes, where each bounding box is:
                         [x_min, x_max, y_min, y_max] NOW x_min,y_min,x_max,y_max
                         
                         Example input:
                         [
                             [50, 250, 100, 130],    # Line 0
                             [300, 500, 100, 130],   # Line 1
                             [50, 250, 180, 210],    # Line 2
                             [300, 500, 180, 210]    # Line 3
                         ]
                         
                         This represents a 2x2 grid:
                         [Line 0]  [Line 1]
                         [Line 2]  [Line 3]
        
        Returns:
            list: Indices of lines in reading order
                  
                  Example output: [0, 1, 2, 3]
                  Meaning: Read line 0, then 1, then 2, then 3
                  
                  If input is empty, returns []
        
        Algorithm Steps:
            1. Handle empty input
            2. Extract features for all lines
            3. Build directed graph:
               - Nodes = line indices (0, 1, 2, ...)
               - Edges = precedence relationships (i→j means "i before j")
            4. Calculate in-degrees (number of predecessors for each node)
            5. Perform topological sort using Kahn's algorithm
            6. Return the sorted order
        """
        if not lines:
            return []
        
        n = len(lines)
        features = [self._get_features(line) for line in lines]
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_precede(features[i], features[j]):
                    graph[i].append(j)
                    in_degree[j] += 1
                else:  
                    graph[j].append(i)
                    in_degree[i] += 1
        
        # Kahn's algorithm for topological sort
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []
        
        while queue or len(result) < n:
            # check special case requiring cycle break: queue empty but all nodes haven't been processed yet
            if not queue:
                # cycle-break: force-pick the next most plausible node, sorted by x and y
                remaining = set(range(n)) - set(result)
                node = min(
                    remaining,
                    key=lambda i: (
                        features[i]['anchor'][1],
                        features[i]['anchor'][0] if self.text_direction == 'lr' else -features[i]['anchor'][0]
                    )
                )
            else:
                # regular case
                node = queue.popleft()
            
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result

def layout_table_nms(boxes, ios_threshold=0.5):
    """
    NMS for PaddleX layout boxes using IoS (intersection over smaller area).

    Keeps boxes whose label is "table", "image", or "figure", then suppresses
    lower-score boxes when their overlap with a higher-score box exceeds the
    IoS threshold.

    IoS = intersection_area / min(area_a, area_b)
    """
    table_boxes = [
        box for box in boxes
        if box.get("label", "").lower() in {"table", "image", "figure"}
    ]

    if not table_boxes:
        return []

    xyxy = np.asarray(
        [box["coordinate"] for box in table_boxes],
        dtype=np.float32,
    )
    scores = np.asarray(
        [box["score"] for box in table_boxes],
        dtype=np.float32,
    )

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    # Protect against malformed / reversed coordinates.
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep_indices = []

    while order.size > 0:
        current = order[0]
        keep_indices.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        inter_x1 = np.maximum(x1[current], x1[remaining])
        inter_y1 = np.maximum(y1[current], y1[remaining])
        inter_x2 = np.minimum(x2[current], x2[remaining])
        inter_y2 = np.minimum(y2[current], y2[remaining])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        smaller_area = np.minimum(areas[current], areas[remaining])

        ios = np.divide(
            intersection,
            smaller_area,
            out=np.zeros_like(intersection),
            where=smaller_area > 0,
        )

        # Retain boxes not sufficiently contained/overlapped.
        order = remaining[ios <= ios_threshold]

    return [table_boxes[i] for i in sorted(keep_indices)]


layout_model = create_model(model_name="PP-DocLayout_plus-L")


class PaddleDetectionFunction:
    def __init__(
        self,
        wired=True,
        **kwargs
    ) -> None:
        self.wired = wired

        if self.wired:
            model_path = "PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors"
            print("Wired model")
        else:
            model_path = "PaddlePaddle/RT-DETR-L_wireless_table_cell_det_safetensors"
            print("Wireless model")
        
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=model_path,
            confidence_threshold=0.3,
            image_size=640,
            device="cpu",  # Change to "cuda" when available.
        )

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        # describe the annotations
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(f"Table{i+1}", i, type="rectangle") 
                for i in range(0, 10)
            ]
        )

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[models.LabeledShapeRequest]:
        # determine the threshold for filtering results
        conf_threshold = context.conf_threshold or self.threshold
        print(f"Conf threshold: {conf_threshold:.2f}")

        # convert the input into a form the model can understand
        original_rgb = np.array(image.convert("RGB"))
        #original_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # layout inference
        layout_output = layout_model.predict(
            original_rgb,
            batch_size=1,
            layout_nms=True,
            threshold=0.3,
        )

        res = next(layout_output)

        # PaddleX results normally support mapping-style access, as reflected in
        # the JSON structure in the question.
        layout_boxes = res["boxes"]

        # Extra NMS over table detections before any crops are created.
        table_boxes_nms = layout_table_nms(
            layout_boxes,
            ios_threshold=0.5,
        )

        # Order retained layout regions in natural page reading order.
        table_orderer = GraphBasedOrdering(text_direction="lr")

        table_box_coords = [
            box["coordinate"]  # [x_min, y_min, x_max, y_max]
            for box in table_boxes_nms
        ]

        reading_order = table_orderer.order(table_box_coords)

        # Reorder the original box dictionaries, preserving labels/scores/etc.
        table_boxes_nms = [
            table_boxes_nms[i]
            for i in reading_order
        ]

        image_h, image_w = original_rgb.shape[:2]

        # This is the final coordinate-space result list: every box is relative
        # to the original, uncropped image.
        global_cell_predictions = []

        # Optional: add a few pixels around each layout table crop so border cells
        # are less likely to be cut off. Set to 0 for exact layout boxes only.
        TABLE_CROP_PADDING = 0

        # Keep the same large SAHI slice size you used before.
        MAX_SLICE_HEIGHT = max(1920, image_h // 6)
        MAX_SLICE_WIDTH = max(1920, image_w // 6)

        # ----------------------------
        # 4. Infer cells separately for every table crop
        # ----------------------------
        for table_index, layout_box in enumerate(table_boxes_nms):
            #if layout_box.get("label", "").lower() not in ["table", "image", "figure"]:  # table and image
            #    continue

            x1, y1, x2, y2 = layout_box["coordinate"]

            # Expand / clamp layout coordinates and convert floating layout boxes
            # into valid NumPy image indices.
            crop_x1 = max(0, int(math.floor(x1)) - TABLE_CROP_PADDING)
            crop_y1 = max(0, int(math.floor(y1)) - TABLE_CROP_PADDING)
            crop_x2 = min(image_w, int(math.ceil(x2)) + TABLE_CROP_PADDING)
            crop_y2 = min(image_h, int(math.ceil(y2)) + TABLE_CROP_PADDING)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                print(f"Skipping invalid table box {table_index}: {layout_box['coordinate']}")
                continue

            table_crop = original_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = table_crop.shape[:2]

            print(
                f"Running SAHI on table {table_index}: "
                f"original coordinates=({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}), "
                f"crop size={crop_w}x{crop_h}"
            )

            # SAHI accepts a NumPy image directly. Its returned cell boxes are
            # coordinates relative to `table_crop`.
            crop_result = get_sliced_prediction(
                table_crop,
                self.detection_model,
                slice_height=min(MAX_SLICE_HEIGHT, crop_h),
                slice_width=min(MAX_SLICE_WIDTH, crop_w),
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
                postprocess_type="NMM"
            )

            # ----------------------------
            # 5. Reproject crop coordinates to original image coordinates
            # ----------------------------
            for pred in crop_result.object_prediction_list:
                local_x1, local_y1, local_x2, local_y2 = pred.bbox.to_xyxy()

                global_x1 = max(0, min(image_w, local_x1 + crop_x1))
                global_y1 = max(0, min(image_h, local_y1 + crop_y1))
                global_x2 = max(0, min(image_w, local_x2 + crop_x1))
                global_y2 = max(0, min(image_h, local_y2 + crop_y1))

                if global_x2 <= global_x1 or global_y2 <= global_y1:
                    continue

                global_cell_predictions.append(
                    {
                        "table_index": table_index,
                        "category_id": int(pred.category.id),
                        "category_name": pred.category.name,
                        "score": float(pred.score.value),
                        "bbox_xyxy": [
                            float(global_x1),
                            float(global_y1),
                            float(global_x2),
                            float(global_y2),
                        ],
                    }
                )
        i = 0
        shapes = []
        for pred in global_cell_predictions:
            points = pred["bbox_xyxy"]

            shapes.append(
                cvataa.rectangle(
                    min(pred["table_index"], 9),  # max index 9 for predefined labels
                    points
                )
            )

        return shapes

def create(**kwargs) -> cvataa.DetectionFunction:
    return PaddleDetectionFunction(**kwargs)