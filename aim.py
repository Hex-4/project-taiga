"""coordinate mapping: pixel in camera frame -> turret aim (pitch_angle, yaw_ms).

how calibration works:
  1. user homes the turret (yaw_ms = 0 by dead-reckoning, pitch = 100 default)
  2. for each known spot on the floor:
     - user clicks that spot in the camera view, capturing its pixel position
     - user aims the turret at the physical spot using the aim pad
     - the server records (pixel_xy, turret.pitch, turret.yaw_ms) as a point
  3. at runtime we use inverse-distance-weighted (IDW) interpolation to estimate
     aim commands for arbitrary pixels

why IDW: points are scattered (not a regular grid), IDW handles that without a
heavy dep like scipy, and it degrades to "nearest neighbor" when you extrapolate
which is perfectly acceptable for nerf darts.

storage: a plain json file on disk. rewritten atomically every time something
changes. survives restarts. the `yaw_ms` coordinate is always interpreted
relative to the turret's "home" (yaw_ms=0), so a calibration is invalidated
if the turret is ever rotated by hand without re-homing after.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationPoint:
    px: int
    py: int
    pitch: int       # servo angle (33-150)
    yaw_ms: int      # cumulative ms from home (+ = ccw, - = cw)
    # optional metadata
    ts: float = field(default_factory=time.time)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AimEstimate:
    """output of `Calibration.estimate` - a proposed turret pose."""
    pitch: int
    yaw_ms: int
    # how many calibration points contributed, and how "extrapolated" this is.
    # a high max_distance means the target is far from any known point, and
    # the estimate is really a nearest-neighbor guess.
    n_used: int
    max_distance_px: float
    exact_match: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class Calibration:
    """persistent list of (pixel, turret-pose) correspondences."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.points: list[CalibrationPoint] = []
        self.load()

    # --- persistence ---

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            # corrupt file; start fresh but don't clobber until save()
            return
        self.points = [CalibrationPoint(**p) for p in data.get("points", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # write atomically so a crash mid-save doesn't wipe the calibration
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {"points": [p.as_dict() for p in self.points]},
                    f,
                    indent=2,
                )
            os.replace(tmp_path, self.path)
        except Exception:
            with contextlib_suppress(OSError):
                os.unlink(tmp_path)
            raise

    # --- mutation ---

    def add(self, point: CalibrationPoint) -> CalibrationPoint:
        self.points.append(point)
        self.save()
        return point

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.save()

    def clear(self) -> None:
        self.points.clear()
        self.save()

    # --- interpolation ---

    def estimate(self, px: float, py: float, *, power: float = 2.0) -> Optional[AimEstimate]:
        """idw estimate of turret pose for an arbitrary pixel.

        returns None if there are no calibration points at all. when 1 point
        exists it degrades to that single value.
        """
        if not self.points:
            return None
        # check for an exact or near-exact hit first
        for p in self.points:
            if (p.px - px) ** 2 + (p.py - py) ** 2 < 2.0:
                return AimEstimate(
                    pitch=int(round(p.pitch)),
                    yaw_ms=int(round(p.yaw_ms)),
                    n_used=1,
                    max_distance_px=0.0,
                    exact_match=True,
                )

        # inverse-distance weighted average. epsilon avoids div-by-zero when a
        # point is ~1px away, though we already handled the exact case above.
        eps = 1e-6
        w_sum = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0
        max_d = 0.0
        for p in self.points:
            dx = p.px - px
            dy = p.py - py
            d = (dx * dx + dy * dy) ** 0.5
            max_d = max(max_d, d)
            w = 1.0 / ((d + eps) ** power)
            w_sum += w
            pitch_acc += w * p.pitch
            yaw_acc += w * p.yaw_ms

        return AimEstimate(
            pitch=int(round(pitch_acc / w_sum)),
            yaw_ms=int(round(yaw_acc / w_sum)),
            n_used=len(self.points),
            max_distance_px=round(max_d, 1),
            exact_match=False,
        )


# tiny stdlib-dodge so we don't have to import contextlib just for one call
import contextlib as _cl
contextlib_suppress = _cl.suppress
