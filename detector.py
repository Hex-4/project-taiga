"""yolov8 detector using onnxruntime for fast cpu inference.

we used to load the .pt with ultralytics+torch (~150-300ms/frame on a t430).
the same model run through onnxruntime + numpy preprocessing + cv2 NMS lands
at ~50-100ms/frame on the same hardware - 2-3x faster, no torch in the hot
path. the .onnx is exported once from the .pt at startup if missing.

public surface unchanged:
    Detector(model_name="yolov8n", conf_threshold=0.25, target_class_ids={15})
    Detector.detect_jpeg(jpeg_bytes) -> ((w, h), [Detection, ...])
    Detector.available_classes() -> {class_id: name}

so the rest of the project (server, tracker, ui) is oblivious to the swap.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

# COCO 80 class names. yolov8n is trained on these. used to be pulled from
# the ultralytics model object; we hardcode here so we can drop ultralytics
# from the runtime path.
COCO_CLASSES: dict[int, str] = {i: n for i, n in enumerate([
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
])}

# default inference size for yolov8n. matches the export's static input shape.
INPUT_SIZE = 640
NMS_IOU_THRESHOLD = 0.45


@dataclass(frozen=True)
class Detection:
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


def _ensure_onnx_export(pt_path: Path, onnx_path: Path) -> None:
    """if .onnx doesn't exist next to the .pt, export it once.

    this is the only place we still touch ultralytics; it's a build-time
    operation, not a runtime one. once `yolov8n.onnx` exists, subsequent
    runs skip the import entirely.
    """
    if onnx_path.exists():
        return
    if not pt_path.exists():
        raise FileNotFoundError(
            f"{pt_path} not found and {onnx_path} doesn't exist either - "
            "fetch the .pt first (any predict/export call with ultralytics will "
            "auto-download it) or copy a .onnx into place."
        )
    print(f"[detector] exporting {pt_path.name} -> {onnx_path.name} (one-time)")
    from ultralytics import YOLO  # heavy import, only when needed
    YOLO(str(pt_path)).export(format="onnx", opset=12, simplify=True, dynamic=False)


def _letterbox(img: np.ndarray, target: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """resize-and-pad to a square target size, preserving aspect ratio.

    returns (padded_image, scale, pad_left, pad_top). pad colour 114/114/114
    matches yolov8's training augmentation so the model sees what it expects.
    """
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h
    left = pad_w // 2
    top = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, left, top


class Detector:
    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.4,
        target_class_ids: set[int] | None = None,
    ) -> None:
        # accept "yolov8n.pt", "yolov8n.onnx", or bare "yolov8n"
        stem = model_name.replace(".pt", "").replace(".onnx", "")
        self.model_stem = stem
        self.conf_threshold = conf_threshold
        self.target_class_ids: set[int] = (
            set(target_class_ids) if target_class_ids is not None else {CAT_CLASS_ID}
        )
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is None:
                pt_path = Path(f"{self.model_stem}.pt")
                onnx_path = Path(f"{self.model_stem}.onnx")
                # export off the event loop - it's slow on first run
                await asyncio.to_thread(_ensure_onnx_export, pt_path, onnx_path)

                # yolov8n is small; benchmarking shows it tops out at ~4
                # threads. on a 12-core box, more threads = oversubscription =
                # *slower*. on a 4-thread t430, threads=4 saturates fully.
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 4))

                def _make() -> ort.InferenceSession:
                    return ort.InferenceSession(
                        str(onnx_path),
                        sess_options=so,
                        providers=["CPUExecutionProvider"],
                    )

                self._session = await asyncio.to_thread(_make)
                self._input_name = self._session.get_inputs()[0].name
                print(f"[detector] onnxruntime ready, threads={so.intra_op_num_threads}")
        return self._session

    async def available_classes(self) -> dict[int, str]:
        # ensure session loads (also validates the model is real)
        await self._ensure_session()
        return dict(COCO_CLASSES)

    async def set_model(self, stem: str) -> None:
        """swap to a different model (e.g. fp32 -> int8). next inference reloads."""
        async with self._lock:
            self.model_stem = stem.replace(".pt", "").replace(".onnx", "")
            self._session = None
            self._input_name = None

    async def detect_jpeg(self, jpeg: bytes) -> tuple[tuple[int, int], list[Detection]]:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return (0, 0), []
        return await self.detect_image(img)

    async def detect_image(self, img: "np.ndarray") -> tuple[tuple[int, int], list[Detection]]:
        h, w = img.shape[:2]
        targets = set(self.target_class_ids)
        if not targets:
            return (w, h), []
        session = await self._ensure_session()
        # all numerics are cpu-bound; offload off the event loop.
        return await asyncio.to_thread(
            self._run_sync, session, img, targets, float(self.conf_threshold), w, h,
        )

    def _run_sync(
        self,
        session: ort.InferenceSession,
        img: "np.ndarray",
        targets: set[int],
        conf_thr: float,
        w: int,
        h: int,
    ) -> tuple[tuple[int, int], list[Detection]]:
        padded, scale, pad_l, pad_t = _letterbox(img)
        # BGR -> RGB and HWC -> CHW; normalise to [0, 1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        x = np.expand_dims(chw, axis=0)

        # raw output: (1, 84, 8400). 84 = 4 cxcywh + 80 class probs.
        y = session.run(None, {self._input_name: x})[0][0]  # -> (84, 8400)
        # transpose to (n_anchors, 84) - faster downstream slicing
        pred = y.transpose(1, 0)
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), class_ids]

        # confidence + class filter together
        mask = confs > conf_thr
        if mask.any() and len(targets) < 80:
            target_arr = np.fromiter(targets, dtype=np.int64)
            mask &= np.isin(class_ids, target_arr)
        if not mask.any():
            return (w, h), []

        boxes_cxcywh = boxes_cxcywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        # cxcywh -> xyxy in the letterboxed coordinate space
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5

        # cv2 NMS wants [x, y, w, h] top-left format
        nms_in = np.column_stack([x1, y1, bw, bh]).astype(np.float32)
        keep = cv2.dnn.NMSBoxes(
            nms_in.tolist(), confs.astype(np.float32).tolist(),
            conf_thr, NMS_IOU_THRESHOLD,
        )
        if len(keep) == 0:
            return (w, h), []
        # cv2 returns either np.ndarray or list-of-lists depending on version
        keep = np.array(keep).flatten()

        # un-letterbox back into the original image's pixel space
        inv = 1.0 / scale
        out: list[Detection] = []
        for i in keep:
            ox1 = float((x1[i] - pad_l) * inv)
            oy1 = float((y1[i] - pad_t) * inv)
            ox2 = float((x2[i] - pad_l) * inv)
            oy2 = float((y2[i] - pad_t) * inv)
            # clamp to image bounds (boxes can spill into the letterbox padding)
            ox1 = max(0.0, min(w - 1, ox1))
            oy1 = max(0.0, min(h - 1, oy1))
            ox2 = max(0.0, min(w - 1, ox2))
            oy2 = max(0.0, min(h - 1, oy2))
            cls_id = int(class_ids[i])
            out.append(Detection(
                x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                confidence=float(confs[i]),
                class_id=cls_id,
                class_name=COCO_CLASSES.get(cls_id, str(cls_id)),
            ))
        return (w, h), out
