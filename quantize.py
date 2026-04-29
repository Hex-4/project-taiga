"""static int8 quantization of the yolov8 onnx model.

we use static (post-training) quantization with calibration data drawn from
the user's actual camera feed. this gives ~1.5-3x speedup on cpu, especially
on older hardware (avx1/avx2 without int8-specific instructions). quality
drop is typically 1-3% map; for our "is there a target in the frame" task
that's noise.

usage:
    quantize_with_frames("yolov8n.onnx", "yolov8n_int8.onnx", frames=[bgr_imgs...])

requires onnxruntime>=1.14 with the `quantization` submodule.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# import the same letterbox so calibration sees inputs identical to inference
from detector import _letterbox, INPUT_SIZE


class FrameCalibrationReader(CalibrationDataReader):
    """yields preprocessed frames as {input_name: chw_float32} for ort.quantize."""

    def __init__(self, frames: list[np.ndarray], input_name: str = "images") -> None:
        self.input_name = input_name
        # generator state - quantize_static iterates until None
        self._iter = iter(self._generate(frames))

    def _generate(self, frames: list[np.ndarray]):
        for img in frames:
            padded, _, _, _ = _letterbox(img, INPUT_SIZE)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
            yield {self.input_name: np.expand_dims(chw, axis=0)}

    def get_next(self) -> dict | None:
        return next(self._iter, None)


def quantize_with_frames(
    src_onnx: str | Path,
    out_onnx: str | Path,
    frames: list[np.ndarray],
    *,
    input_name: str = "images",
) -> dict:
    """run static int8 quantization. blocking - call in a thread.

    returns a small dict with file sizes for reporting back to the UI.
    """
    src = Path(src_onnx)
    out = Path(out_onnx)
    if not src.exists():
        raise FileNotFoundError(f"{src} doesn't exist")
    if not frames:
        raise ValueError("need at least 1 calibration frame; ideally 20+")

    # ort's quantizer wants the model pre-processed to fold constants and
    # infer shapes. this writes a temp file that we throw away after quantizing.
    preproc = src.with_name(src.stem + "_preproc.onnx")
    quant_pre_process(str(src), str(preproc), skip_optimization=False)

    try:
        reader = FrameCalibrationReader(frames, input_name=input_name)
        # QDQ format with QInt8 weights + QUInt8 activations is the recommended
        # combo for cpu inference. it's well-supported and fast.
        quantize_static(
            str(preproc), str(out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=False,  # per-tensor is simpler + faster on cpu
        )
    finally:
        if preproc.exists():
            preproc.unlink()

    return {
        "src_size_mb": round(src.stat().st_size / 1e6, 2),
        "out_size_mb": round(out.stat().st_size / 1e6, 2),
        "frames_used": len(frames),
    }
