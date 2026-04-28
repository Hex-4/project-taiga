"""centroid tracker + stillness detector for the primary cat in frame.

design: we don't need multi-object tracking. we pick the single "most prominent"
cat detection in each frame (largest bounding box = closest / most central) and
track its centroid across frames. "stillness" is measured as: how many seconds
has the centroid stayed within `threshold_px` of its current position.

yolov8n flickers on partial or weirdly-angled views of cats, so we also
"bridge" short detection gaps: if we recently saw the cat and now don't,
we carry the last known position forward for up to `bridge_s` seconds.
this stops micro-movements (head turns, tail flicks, brief occlusions) from
resetting the stillness timer to zero.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from detector import Detection


@dataclass(frozen=True)
class TrackState:
    active: bool
    center: tuple[float, float] | None
    bbox: tuple[float, float, float, float] | None  # x1,y1,x2,y2 of tracked cat
    stillness_s: float  # how long the cat has been stationary (0 if moving/absent)
    stationary: bool  # true once stillness_s >= still_duration_s
    image_size: tuple[int, int] | None  # (w, h) of frame this state refers to
    # true if this frame had no fresh detection but we're coasting on a
    # recent one (within bridge_s). clients can render the overlay in a
    # different style to make "we're guessing" visible.
    bridged: bool = False

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "center": self.center,
            "bbox": self.bbox,
            "stillness_s": round(self.stillness_s, 2),
            "stationary": self.stationary,
            "bridged": self.bridged,
        }


class StillnessTracker:
    """tracks centroid of the largest cat detection across frames.

    parameters:
      still_threshold_frac: how far (as fraction of the shorter image edge) the
        centroid may wander and still count as "not moving". 0.05 = 5% of the
        shorter edge, which is roughly the width of a cat's head.
      still_duration_s: how long the cat must stay within the threshold before
        we consider it "stationary" (about to do its business).
      history_s: how much position history to retain. should be >= still_duration_s.
      bridge_s: how long to keep a track alive after the last detection. during
        this window we reuse the last bbox and keep the stillness timer running.
        if the gap exceeds this, the track is cleared.
    """

    def __init__(
        self,
        still_threshold_frac: float = 0.05,
        still_duration_s: float = 10.0,
        history_s: float = 15.0,
        bridge_s: float = 1.5,
    ) -> None:
        self.still_threshold_frac = still_threshold_frac
        self.still_duration_s = still_duration_s
        self.history_s = history_s
        self.bridge_s = bridge_s
        # history entries: (timestamp, cx, cy)
        self._history: deque[tuple[float, float, float]] = deque()
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_detection_ts: float | None = None
        self._last_state = TrackState(
            active=False, center=None, bbox=None,
            stillness_s=0.0, stationary=False, image_size=None,
        )

    def update(
        self,
        detections: list[Detection],
        image_size: tuple[int, int],
        now: float | None = None,
    ) -> TrackState:
        now = now if now is not None else time.time()
        # detections have already been filtered by the detector to enabled
        # target classes, so we just pick the most prominent target by area.
        if detections:
            largest = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
            bbox = (largest.x1, largest.y1, largest.x2, largest.y2)
            cx = (largest.x1 + largest.x2) / 2.0
            cy = (largest.y1 + largest.y2) / 2.0
            self._last_bbox = bbox
            self._last_detection_ts = now
            self._push_history(now, cx, cy)
            return self._compute_state(
                now, image_size, bbox=bbox, center=(cx, cy), bridged=False,
            )

        # no fresh detection. if we saw a target recently, carry the old bbox
        # forward and KEEP the stillness timer running - the cat is almost
        # certainly still there, the detector just blinked.
        gap = (
            now - self._last_detection_ts
            if self._last_detection_ts is not None
            else float("inf")
        )
        if gap <= self.bridge_s and self._last_bbox is not None:
            bx1, by1, bx2, by2 = self._last_bbox
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            # don't append to history during bridging - we don't have a real
            # observation to record. stillness is still computed against the
            # historical positions so the timer keeps climbing.
            return self._compute_state(
                now, image_size, bbox=self._last_bbox, center=(cx, cy), bridged=True,
            )

        # gap too long; give up on the track.
        self._history.clear()
        self._last_bbox = None
        self._last_detection_ts = None
        self._last_state = TrackState(
            active=False, center=None, bbox=None,
            stillness_s=0.0, stationary=False, image_size=image_size,
        )
        return self._last_state

    def _push_history(self, now: float, cx: float, cy: float) -> None:
        self._history.append((now, cx, cy))
        cutoff = now - self.history_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _compute_state(
        self,
        now: float,
        image_size: tuple[int, int],
        *,
        bbox: tuple[float, float, float, float],
        center: tuple[float, float],
        bridged: bool,
    ) -> TrackState:
        cx, cy = center
        w, h = image_size
        threshold_px = self.still_threshold_frac * min(w, h) if (w and h) else 30.0

        # stillness = duration of the suffix of the history where every entry
        # is within threshold of the CURRENT centroid. we walk backward from
        # the most recent observation and stop at the first one too far away.
        stillness_s = 0.0
        if self._history:
            for t, hx, hy in reversed(self._history):
                if ((hx - cx) ** 2 + (hy - cy) ** 2) ** 0.5 <= threshold_px:
                    stillness_s = now - t
                else:
                    break

        state = TrackState(
            active=True,
            center=(round(cx, 1), round(cy, 1)),
            bbox=(round(bbox[0], 1), round(bbox[1], 1),
                  round(bbox[2], 1), round(bbox[3], 1)),
            stillness_s=stillness_s,
            stationary=stillness_s >= self.still_duration_s,
            image_size=image_size,
            bridged=bridged,
        )
        self._last_state = state
        return state

    @property
    def state(self) -> TrackState:
        return self._last_state

    def reset(self) -> None:
        self._history.clear()
        self._last_bbox = None
        self._last_detection_ts = None
