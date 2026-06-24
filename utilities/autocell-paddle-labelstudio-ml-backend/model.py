from __future__ import annotations

import math
import uuid
from collections import defaultdict, deque
from typing import Any, List, Dict, Optional

import numpy as np
from PIL import Image
from paddlex import create_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import os

os.environ["HF_HOME"] = "/data/models/hfhome"
os.environ["HF_HUB_CACHE"] = "/data/models/hfcache"
os.environ["PADDLE_PDX_CACHE_HOME"] = "/data/models/paddlex"

UNDEFINED_LABEL = "blank"


class GraphBasedOrdering:
    def __init__(self, text_direction: str = "lr") -> None:
        self.text_direction = text_direction

    @staticmethod
    def _get_features(box):
        x_min, y_min, x_max, y_max = box
        return {
            "anchor": (x_min, y_min),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }

    def _should_precede(self, first, second) -> bool:
        first_anchor = first["anchor"]
        second_anchor = second["anchor"]

        overlap = min(first["y_max"], second["y_max"]) - max(first["y_min"], second["y_min"])
        average_height = (first["height"] + second["height"]) / 2

        if overlap > 0.5 * average_height:
            return first_anchor[0] < second_anchor[0]

        return first_anchor[1] < second_anchor[1]

    def order(self, boxes):
        if not boxes:
            return []

        count = len(boxes)
        features = [self._get_features(box) for box in boxes]
        graph = defaultdict(list)
        in_degree = [0] * count

        for left in range(count):
            for right in range(left + 1, count):
                if self._should_precede(features[left], features[right]):
                    graph[left].append(right)
                    in_degree[right] += 1
                else:
                    graph[right].append(left)
                    in_degree[left] += 1

        queue = deque(i for i in range(count) if in_degree[i] == 0)
        result = []

        while queue or len(result) < count:
            if queue:
                node = queue.popleft()
            else:
                remaining = set(range(count)) - set(result)
                node = min(
                    remaining,
                    key=lambda i: (features[i]["anchor"][1], features[i]["anchor"][0]),
                )

            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


def layout_table_nms(boxes, ios_threshold: float = 0.5):
    candidate_boxes = [
        box for box in boxes
        if str(box.get("label", "")).lower() in {"table", "image", "figure"}
    ]

    if not candidate_boxes:
        return []

    xyxy = np.asarray([box["coordinate"] for box in candidate_boxes], dtype=np.float32)
    scores = np.asarray([box["score"] for box in candidate_boxes], dtype=np.float32)

    x1, y1, x2, y2 = xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep_indices = []

    while order.size:
        current = int(order[0])
        keep_indices.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining]) - np.maximum(x1[current], x1[remaining]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining]) - np.maximum(y1[current], y1[remaining]),
        )

        intersection = intersection_width * intersection_height
        smaller_area = np.minimum(areas[current], areas[remaining])

        ios = np.divide(
            intersection,
            smaller_area,
            out=np.zeros_like(intersection),
            where=smaller_area > 0,
        )

        order = remaining[ios <= ios_threshold]

    return [candidate_boxes[index] for index in sorted(keep_indices)]


class TableCellPredictor:
    def __init__(
        self,
        wired: bool = True,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        model_path = (
            "PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors"
            if wired
            else "PaddlePaddle/RT-DETR-L_wireless_table_cell_det_safetensors"
        )

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            image_size=640,
            device=device,
        )

        self.layout_model = create_model(model_name="PP-DocLayout_plus-L")
        self.orderer = GraphBasedOrdering(text_direction="lr")

    def predict(self, image: Image.Image):
        original_rgb = np.asarray(image.convert("RGB"))
        image_height, image_width = original_rgb.shape[:2]

        layout_result = next(
            self.layout_model.predict(
                original_rgb,
                batch_size=1,
                layout_nms=True,
                threshold=0.3,
            )
        )

        layout_boxes = layout_table_nms(layout_result["boxes"], ios_threshold=0.5)
        reading_order = self.orderer.order(
            [list(map(float, box["coordinate"])) for box in layout_boxes]
        )
        layout_boxes = [layout_boxes[index] for index in reading_order]

        max_slice_height = max(1920, image_height // 6)
        max_slice_width = max(1920, image_width // 6)

        predictions = []

        for layout_box in layout_boxes:
            x1, y1, x2, y2 = map(float, layout_box["coordinate"])

            crop_x1 = max(0, int(math.floor(x1)))
            crop_y1 = max(0, int(math.floor(y1)))
            crop_x2 = min(image_width, int(math.ceil(x2)))
            crop_y2 = min(image_height, int(math.ceil(y2)))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = original_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_height, crop_width = crop.shape[:2]

            sliced_prediction = get_sliced_prediction(
                crop,
                self.detection_model,
                slice_height=min(max_slice_height, crop_height),
                slice_width=min(max_slice_width, crop_width),
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
                postprocess_type="NMM",
            )

            for detected_cell in sliced_prediction.object_prediction_list:
                local_x1, local_y1, local_x2, local_y2 = detected_cell.bbox.to_xyxy()

                global_x1 = max(0.0, min(float(image_width), local_x1 + crop_x1))
                global_y1 = max(0.0, min(float(image_height), local_y1 + crop_y1))
                global_x2 = max(0.0, min(float(image_width), local_x2 + crop_x1))
                global_y2 = max(0.0, min(float(image_height), local_y2 + crop_y1))

                if global_x2 <= global_x1 or global_y2 <= global_y1:
                    continue

                predictions.append({
                    "score": float(detected_cell.score.value),
                    "bbox_xyxy": [global_x1, global_y1, global_x2, global_y2],
                })

        return predictions

def ls_rectangle_result(
    bbox_xyxy,
    score: float,
    image_width: int,
    image_height: int,
    from_name: str,
    to_name: str,
):
    x1, y1, x2, y2 = bbox_xyxy

    return {
        "id": str(uuid.uuid4()),
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "score": score,
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x1 / image_width,
            "y": 100.0 * y1 / image_height,
            "width": 100.0 * (x2 - x1) / image_width,
            "height": 100.0 * (y2 - y1) / image_height,
            "rotation": 0,
            "rectanglelabels": [UNDEFINED_LABEL],
        },
    }

# initialize once
wired_predictor = TableCellPredictor(
    wired=True,
    confidence_threshold=0.3,
    device="cpu",  # change to "cuda" if available
)

'''wireless_predictor = TableCellPredictor(
    wired=False,
    confidence_threshold=0.3,
    device="cpu",  # change to "cuda" if available
)'''

class NewModel(LabelStudioMLBase):
    def setup(self):
        self.set("model_version", "table-cell-detector-v1")

        '''self.predictor = TableCellPredictor(
            wired=True,
            confidence_threshold=0.3,
            device="cpu",  # change to "cuda" if available
        )'''
        self.predictor = wired_predictor

    def _get_label_studio_names(self):
        for from_name, config in self.parsed_label_config.items():
            if config.get("type") == "RectangleLabels":
                to_name = config["to_name"][0]
                data_key = self.parsed_label_config[to_name]["value"].lstrip("$")
                return from_name, to_name, data_key

        return "label", "image", "image"

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        #from_name, to_name, data_key = self._get_label_studio_names()
        from_name = "label"
        to_name = "image"
        data_key = "image"
        predictions = []

        for task in tasks:
            image_url = task["data"][data_key]
            image_path = self.get_local_path(image_url, task_id=task.get("id"))

            with Image.open(image_path) as image:
                image_width, image_height = image.size
                cells = self.predictor.predict(image)

            results = [
                ls_rectangle_result(
                    bbox_xyxy=cell["bbox_xyxy"],
                    score=cell["score"],
                    image_width=image_width,
                    image_height=image_height,
                    from_name=from_name,
                    to_name=to_name,
                )
                for cell in cells
            ]

            mean_score = float(np.mean([cell["score"] for cell in cells])) if cells else 0.0

            predictions.append({
                "model_version": self.get("model_version"),
                "score": mean_score,
                "result": results,
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        old_model_version = self.get("model_version")
        print(f"Old model version: {old_model_version}")
        print("fit() completed successfully.")