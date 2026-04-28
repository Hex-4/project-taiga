"""yolov8 detector wrapper. loads the model once, runs inference in a thread pool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO


# handy COCO class ids for defaults and docs. the full list is available at
# runtime via Detector.available_classes() once the model is loaded.
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16


@dataclass(frozen=True)
class Detection:
    """one bounding box from the detector.

    coordinates are absolute pixels in the source image (not normalised),
    top-left origin. x2 >= x1 and y2 >= y1 always hold.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def as_dict(self) -> dict:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "conf": round(self.confidence, 3),
            "cls": self.class_name,
            "cls_id": self.class_id,
        }


class Detector:
    """thin wrapper around a yolov8 model.

    the first `predict` call materialises the model (and downloads weights on
    the first ever run). subsequent calls are fast. inference is cpu-bound and
    blocking, so we always run it via asyncio.to_thread to avoid stalling the
    event loop.

    `target_class_ids` is a mutable set of COCO class ids the detector will
    return. changing it is cheap (no model reload). defaults to {cat}.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        target_class_ids: set[int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class_ids: set[int] = (
            set(target_class_ids) if target_class_ids is not None else {CAT_CLASS_ID}
        )
        self._model: YOLO | None = None
        self._model_lock = asyncio.Lock()

    async def _ensure_model(self) -> "YOLO":
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                # import is deferred because ultralytics is heavy (pulls torch).
                from ultralytics import YOLO

                self._model = await asyncio.to_thread(YOLO, self.model_name)
        return self._model

    async def available_classes(self) -> dict[int, str]:
        """returns the full class_id -> name mapping from the model (COCO = 80 entries)."""
        model = await self._ensure_model()
        return dict(model.names)

    async def detect_jpeg(self, jpeg: bytes) -> tuple[tuple[int, int], list[Detection]]:
        """decode a jpeg buffer and return (image size wh, detections)."""
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return (0, 0), []
        return await self.detect_image(img)

    async def detect_image(self, img: "np.ndarray") -> tuple[tuple[int, int], list[Detection]]:
        h, w = img.shape[:2]
        model = await self._ensure_model()
        # capture the target set at the start of inference so a mid-request
        # change from /classes doesn't skew results mid-frame.
        targets = set(self.target_class_ids)
        if not targets:
            # no classes selected = system is "blind"; skip inference entirely.
            return (w, h), []

        results = await asyncio.to_thread(
            model.predict, img, conf=self.conf_threshold, verbose=False
        )

        dets: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id not in targets:
                    continue
                xyxy = box.xyxy[0].tolist()
                dets.append(
                    Detection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(box.conf.item()),
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                    )
                )
        return (w, h), dets
