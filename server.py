"""
taiga cv server.

pipeline:
  phone --(jpeg bytes over ws)-->  [ingest bus, latest-only]
                                        |
                                        v
                                  [detection worker]
                                        |
                                        v
                           [annotated frame + tracker state]
                                        |
                             +----------+-----------+
                             |                      |
                             v                      v
                      back to phone           out to viewers
                    (detection json)       (frames + detections)

the detection worker always works on the latest available raw frame and drops
any frames that arrived while it was busy. this gives us backpressure for free
- if detection is slower than the phone's capture rate, we just skip frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from aim import Calibration, CalibrationPoint
from detector import Detection, Detector
from tracker import StillnessTracker, TrackState
from turret import Turret

ROOT = Path(__file__).parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI(title="taiga")

# --- singletons ---
detector = Detector(model_name="yolov8n.pt", conf_threshold=0.25)
tracker = StillnessTracker(
    still_threshold_frac=0.05,
    still_duration_s=10.0,
    history_s=15.0,
    bridge_s=1.5,  # tolerate 1.5s detector flicker without resetting the track
)
turret = Turret(cooldown_s=10.0)
calibration = Calibration(path=ROOT / "calibration.json")


# --- recent-events ring buffer ---
# the cv server prints [threat] / [turret] / [aim] lines to stdout; mirror them
# into a bounded in-memory list that /view can poll. lets you debug from
# another room without ssh'ing back to the server box.
class EventLog:
    def __init__(self, capacity: int = 200) -> None:
        self.capacity = capacity
        self._events: list[dict] = []
        self._next_id: int = 0

    def add(self, kind: str, message: str) -> None:
        self._events.append({
            "id": self._next_id,
            "ts": time.time(),
            "kind": kind,
            "message": message,
        })
        self._next_id += 1
        if len(self._events) > self.capacity:
            del self._events[: len(self._events) - self.capacity]

    def since(self, last_id: int = -1, limit: int = 50) -> list[dict]:
        return [e for e in self._events if e["id"] > last_id][-limit:]


events = EventLog()


def log_event(kind: str, message: str) -> None:
    """append to the ring buffer AND echo to stdout (for terminal debugging)."""
    events.add(kind, message)
    print(f"[{kind}] {message}")


@dataclass
class TurretPolicy:
    """server-level fire policy, separate from the firmware's own `armed` flag.

    both have to agree for a shot to happen: the firmware's armed flag is
    sticky state on the arduino side (survives a usb reconnect), the server's
    `enabled` flag controls *whether the server will ever send a FIRE*.
    """
    # auto-fire sends a FIRE when the tracker reports stationary. off = manual only.
    auto_fire: bool = False
    # darts per auto-fire event (1 or 2). manual fire always sends 1.
    darts_per_event: int = 1


turret_policy = TurretPolicy()


@dataclass
class AimPolicy:
    """server-level aim-then-fire timing."""
    # how long to wait after a YAW move before sending FIRE. lets the servos
    # physically settle so the dart goes where we aimed, not where we *were*.
    settle_s: float = 0.3


aim_policy = AimPolicy()


@dataclass
class StreamConfig:
    """phone-side capture+upload knobs. the phone polls /config and applies
    these on the next frame. changing them is cheap; the canvas is recreated
    if max_edge changes resolution."""
    target_fps: int = 10        # frames per second uploaded
    jpeg_quality: float = 0.6   # 0.0-1.0 jpeg quality
    max_edge: int = 640         # max long-edge in pixels of the upload


stream_config = StreamConfig()


@dataclass
class AnnotatedFrame:
    """the output of one detection pass. bundles everything a viewer needs."""

    jpeg: bytes
    image_size: tuple[int, int]
    detections: list[Detection]
    track: TrackState
    ts: float
    # increments for every frame the worker produces, so clients can tell them apart
    seq: int

    def detections_payload(self) -> dict:
        return {
            "type": "detections",
            "seq": self.seq,
            "ts": self.ts,
            "image_size": list(self.image_size),
            "detections": [d.as_dict() for d in self.detections],
            "track": self.track.as_dict(),
        }


class LatestSlot:
    """single-slot async mailbox that always holds the newest value.

    producers overwrite; consumers await the next "new" value. if a new value
    arrives while the consumer is still handling the previous one, the older
    one is overwritten and lost - that is the point.
    """

    def __init__(self) -> None:
        self._value = None
        self._seq = 0
        self._event = asyncio.Event()

    def publish(self, value) -> None:
        self._value = value
        self._seq += 1
        self._event.set()

    async def next_after(self, last_seq: int) -> tuple[int, object]:
        while self._seq <= last_seq:
            self._event.clear()
            await self._event.wait()
        return self._seq, self._value

    @property
    def value(self):
        return self._value

    @property
    def seq(self) -> int:
        return self._seq


@dataclass
class BusState:
    raw: LatestSlot
    annotated: LatestSlot
    # tracks the currently-connected phone so the detection worker can push
    # detections back on its websocket. only one phone at a time.
    ingest_ws: Optional[WebSocket]
    frames_received: int
    frames_processed: int
    last_inference_ms: float


state = BusState(
    raw=LatestSlot(),
    annotated=LatestSlot(),
    ingest_ws=None,
    frames_received=0,
    frames_processed=0,
    last_inference_ms=0.0,
)


async def detection_worker() -> None:
    """consume raw frames, run detection + tracker, publish annotated results."""
    last_seq = 0
    while True:
        seq, jpeg = await state.raw.next_after(last_seq)
        last_seq = seq
        assert isinstance(jpeg, (bytes, bytearray))

        t0 = time.perf_counter()
        try:
            image_size, detections = await detector.detect_jpeg(jpeg)
        except Exception as e:
            print(f"[detector] error: {e!r}")
            continue
        track = tracker.update(detections, image_size)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        annotated = AnnotatedFrame(
            jpeg=bytes(jpeg),
            image_size=image_size,
            detections=detections,
            track=track,
            ts=time.time(),
            seq=seq,
        )
        state.annotated.publish(annotated)
        state.frames_processed += 1
        state.last_inference_ms = inference_ms

        # echo detections back to the source phone (best-effort; fire-and-forget)
        phone_ws = state.ingest_ws
        if phone_ws is not None:
            try:
                await phone_ws.send_text(json.dumps(annotated.detections_payload()))
            except Exception:
                # if the phone's gone, the ingest handler will notice and clean up
                pass

        if track.stationary:
            log_event("threat",
                f"target stationary for {track.stillness_s:.1f}s at {track.center} (frame {seq})")
            if turret_policy.auto_fire and turret.state.connected and turret.state.armed:
                # aim first (if we have calibration) then fire. if calibration
                # is empty we shoot from the current pose - user is expected
                # to have manually aimed the turret in that case.
                if track.center is not None and calibration.points:
                    est = calibration.estimate(track.center[0], track.center[1])
                    if est is not None and turret.cooldown_remaining() == 0:
                        aim_reply = await turret.aim_to(est.pitch, est.yaw_ms, settle_s=aim_policy.settle_s)
                        if not aim_reply.ok:
                            log_event("turret", f"aim failed: {aim_reply.body}")
                            continue
                        log_event("turret",
                            f"aimed at px={track.center} -> pitch={est.pitch} "
                            f"eff_yaw_ms={est.yaw_ms} (used {est.n_used} pts, farthest {est.max_distance_px}px)")
                reply = await turret.fire(count=turret_policy.darts_per_event)
                if reply.ok:
                    log_event("turret", f"auto-fired: {reply.body}")
                else:
                    # cooldown / empty mag / etc. not an error per se.
                    log_event("turret", f"fire refused: {reply.body}")


async def turret_connect_loop() -> None:
    """keep the turret connected. tries once on boot, then retries every 5s
    if the port disappears (e.g. cable unplugged and replugged)."""
    while True:
        if not turret.state.connected:
            connected = await turret.connect()
            if not connected:
                await asyncio.sleep(5.0)
                continue
        # while connected, periodically verify by reading pending bytes. a
        # disconnected port will start throwing on i/o.
        await asyncio.sleep(2.0)
        try:
            await turret._drain_into_state()
        except Exception as e:
            print(f"[turret] connection lost: {e}")
            with contextlib.suppress(Exception):
                await turret.disconnect()


@app.on_event("startup")
async def _startup() -> None:
    app.state.worker_task = asyncio.create_task(detection_worker())
    app.state.turret_task = asyncio.create_task(turret_connect_loop())
    print("[startup] detection worker + turret loop launched")


@app.on_event("shutdown")
async def _shutdown() -> None:
    for key in ("worker_task", "turret_task"):
        task: asyncio.Task = getattr(app.state, key, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    with contextlib.suppress(Exception):
        await turret.disconnect()


# --- http routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/phone", response_class=HTMLResponse)
async def phone_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "phone.html")


@app.get("/view", response_class=HTMLResponse)
async def view_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "view.html")


@app.get("/snapshot.jpg")
async def snapshot() -> Response:
    af: AnnotatedFrame | None = state.annotated.value
    if af is None:
        # fall back to the raw latest if we have it
        if state.raw.value is None:
            return Response(status_code=503, content=b"no frame yet")
        return Response(content=state.raw.value, media_type="image/jpeg")
    return Response(content=af.jpeg, media_type="image/jpeg")


# --- tunable knobs: live-read/write via /view ---
#
# each knob is (getter, setter, min, max). the setter is trusted to clamp
# into the declared range. returning a mini dsl rather than splatting the
# logic across endpoints keeps this terse.
CONFIG_KNOBS = {
    "conf_threshold": (
        lambda: detector.conf_threshold,
        lambda v: setattr(detector, "conf_threshold", v),
        0.05, 0.95,
    ),
    "still_threshold_frac": (
        lambda: tracker.still_threshold_frac,
        lambda v: setattr(tracker, "still_threshold_frac", v),
        0.005, 0.5,
    ),
    "still_duration_s": (
        lambda: tracker.still_duration_s,
        lambda v: setattr(tracker, "still_duration_s", v),
        0.5, 60.0,
    ),
    "bridge_s": (
        lambda: tracker.bridge_s,
        lambda v: setattr(tracker, "bridge_s", v),
        0.0, 10.0,
    ),
    "history_s": (
        lambda: tracker.history_s,
        lambda v: setattr(tracker, "history_s", v),
        1.0, 60.0,
    ),
    "yaw_deadband_l_ms": (
        lambda: turret.yaw_deadband_l_ms,
        lambda v: setattr(turret, "yaw_deadband_l_ms", int(v)),
        0.0, 400.0,
    ),
    "yaw_deadband_r_ms": (
        lambda: turret.yaw_deadband_r_ms,
        lambda v: setattr(turret, "yaw_deadband_r_ms", int(v)),
        0.0, 400.0,
    ),
    # aim + fire policy (numeric)
    "aim_settle_s": (
        lambda: aim_policy.settle_s,
        lambda v: setattr(aim_policy, "settle_s", float(v)),
        0.0, 2.0,
    ),
    "cooldown_s": (
        lambda: turret.cooldown_s,
        lambda v: setattr(turret, "cooldown_s", float(v)),
        0.0, 300.0,
    ),
    "darts_per_event": (
        lambda: turret_policy.darts_per_event,
        lambda v: setattr(turret_policy, "darts_per_event", max(1, min(2, int(v)))),
        1.0, 2.0,
    ),
    # stream / phone capture (the phone polls /config and applies these)
    "stream_fps": (
        lambda: stream_config.target_fps,
        lambda v: setattr(stream_config, "target_fps", max(1, min(30, int(v)))),
        1.0, 30.0,
    ),
    "stream_jpeg_quality": (
        lambda: stream_config.jpeg_quality,
        lambda v: setattr(stream_config, "jpeg_quality", float(v)),
        0.2, 0.95,
    ),
    "stream_max_edge": (
        lambda: stream_config.max_edge,
        lambda v: setattr(stream_config, "max_edge", max(160, min(1920, int(v)))),
        160.0, 1920.0,
    ),
}


def _read_config() -> dict:
    return {k: {"value": g(), "min": lo, "max": hi} for k, (g, _, lo, hi) in CONFIG_KNOBS.items()}


@app.get("/config")
async def get_config() -> dict:
    return _read_config()


@app.post("/config")
async def set_config(payload: dict = Body(...)) -> dict:
    """partial update: only keys present in the payload are applied."""
    errors: dict[str, str] = {}
    for key, raw in payload.items():
        if key not in CONFIG_KNOBS:
            errors[key] = "unknown key"
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[key] = "not a number"
            continue
        _, setter, lo, hi = CONFIG_KNOBS[key]
        if not (lo <= v <= hi):
            errors[key] = f"out of range [{lo}, {hi}]"
            continue
        setter(v)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors, "config": _read_config()})
    return _read_config()


# --- target classes endpoints ---

@app.get("/classes")
async def get_classes() -> dict:
    """list every class the model can detect + which are currently selected."""
    try:
        available = await detector.available_classes()
    except Exception as e:
        raise HTTPException(500, f"model load failed: {e!r}")
    return {
        "available": [{"id": cid, "name": name} for cid, name in sorted(available.items())],
        "selected": sorted(detector.target_class_ids),
    }


@app.post("/classes")
async def set_classes(payload: dict = Body(...)) -> dict:
    """replace the selected class set. payload = {"ids": [15, 16, ...]}."""
    ids = payload.get("ids")
    if not isinstance(ids, list):
        raise HTTPException(400, "ids must be a list of integers")
    try:
        clean = {int(i) for i in ids}
    except (TypeError, ValueError):
        raise HTTPException(400, "ids must be integers")
    available = await detector.available_classes()
    unknown = [i for i in clean if i not in available]
    if unknown:
        raise HTTPException(400, f"unknown class ids: {unknown}")
    detector.target_class_ids = clean
    # tracker is class-agnostic but if the user swaps the target entirely,
    # the old track is about a different kind of object and should be dropped.
    tracker.reset()
    return {"selected": sorted(detector.target_class_ids)}


# --- events feed ---

@app.get("/events")
async def get_events(since: int = -1, limit: int = 50) -> dict:
    return {"events": events.since(last_id=since, limit=limit)}


# --- calibration endpoints ---

def _calibration_snapshot() -> dict:
    return {
        "points": [
            {"index": i, **p.as_dict()} for i, p in enumerate(calibration.points)
        ],
    }


@app.get("/calibration")
async def get_calibration() -> dict:
    return _calibration_snapshot()


@app.post("/calibration/capture")
async def capture_calibration(payload: dict = Body(...)) -> dict:
    """record a calibration point. the caller provides the pixel position the
    user just clicked; we snapshot the turret's CURRENT pitch + yaw_ms."""
    try:
        px = int(payload["px"])
        py = int(payload["py"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "payload must be {'px': int, 'py': int}")
    if not turret.state.connected:
        raise HTTPException(409, "turret not connected")
    # store EFFECTIVE yaw (deadband-compensated) so the linearity math is
    # clean. at runtime, aim_to adds deadband back to produce the actual pulse.
    point = CalibrationPoint(
        px=px,
        py=py,
        pitch=int(turret.state.pitch),
        yaw_ms=int(turret.state.eff_yaw_ms),
        note=str(payload.get("note", "")),
    )
    calibration.add(point)
    return _calibration_snapshot()


@app.delete("/calibration/points/{index}")
async def delete_calibration_point(index: int) -> dict:
    if not (0 <= index < len(calibration.points)):
        raise HTTPException(404, "index out of range")
    calibration.remove(index)
    return _calibration_snapshot()


@app.post("/calibration/clear")
async def clear_calibration() -> dict:
    calibration.clear()
    return _calibration_snapshot()


@app.post("/calibration/estimate")
async def estimate_aim(payload: dict = Body(...)) -> dict:
    """estimate turret pose for a given pixel (for previewing / 'test aim')."""
    try:
        px = float(payload["px"])
        py = float(payload["py"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "payload must be {'px': number, 'py': number}")
    est = calibration.estimate(px, py)
    return {"estimate": est.as_dict() if est else None}


# --- turret control endpoints ---

def _turret_snapshot() -> dict:
    return {
        "state": turret.state.as_dict(),
        "policy": {
            "auto_fire": turret_policy.auto_fire,
            "darts_per_event": turret_policy.darts_per_event,
            "cooldown_s": turret.cooldown_s,
            "cooldown_remaining_s": round(turret.cooldown_remaining(), 2),
        },
    }


@app.get("/turret")
async def turret_state() -> dict:
    return _turret_snapshot()


@app.post("/turret/command")
async def turret_command(payload: dict = Body(...)) -> dict:
    """dispatch a single command. payload = {"action": "...", "args": {...}}."""
    action = str(payload.get("action", "")).lower()
    args = payload.get("args") or {}

    if action in ("arm", "disarm", "home", "reload", "status"):
        reply = await getattr(turret, action)()
    elif action == "fire":
        count = int(args.get("count", 1))
        override = bool(args.get("override_cooldown", False))
        reply = await turret.fire(count=count, override_cooldown=override)
    elif action == "pitch":
        angle = int(args["angle"])
        reply = await turret.pitch_abs(angle)
    elif action == "pitch_rel":
        delta = int(args["delta"])
        reply = await turret.pitch_rel(delta)
    elif action == "yaw":
        direction = str(args["direction"])
        ms = int(args.get("ms", 150))
        reply = await turret.yaw(direction, ms=ms)
    elif action == "aim_pixel":
        # aim at a camera-frame pixel using the current calibration; does NOT fire.
        px = float(args["px"])
        py = float(args["py"])
        est = calibration.estimate(px, py)
        if est is None:
            raise HTTPException(409, "no calibration points yet")
        reply = await turret.aim_to(est.pitch, est.yaw_ms, settle_s=0.1)
        snap = _turret_snapshot()
        snap["reply"] = {"ok": reply.ok, "kind": reply.kind, "body": reply.body}
        snap["aim"] = est.as_dict()
        return snap
    elif action == "aim_abs":
        pitch = int(args["pitch"])
        yaw_ms = int(args["yaw_ms"])
        reply = await turret.aim_to(pitch, yaw_ms, settle_s=0.1)
    elif action == "linearity_test":
        # send 3 pulses with a gap between them so the user can eyeball whether
        # physical rotation per pulse is proportional to requested duration.
        # direction alternates so the turret doesn't walk off the wall.
        widths = args.get("widths") or [100, 300, 500]
        direction = str(args.get("direction", "L")).upper()
        replies: list = []
        for w in widths:
            r = await turret.yaw(direction, ms=int(w))
            replies.append({"ms": int(w), "ok": r.ok, "body": r.body})
            await asyncio.sleep(0.6)
            # flip direction each iteration so we don't drift in one direction
            direction = "R" if direction == "L" else "L"
        snap = _turret_snapshot()
        snap["linearity_test"] = {"pulses": replies}
        return snap
    elif action == "set_timing":
        yaw_ms = args.get("yaw_ms")
        roll_ms = args.get("roll_ms")
        replies = await turret.set_timing(
            yaw_ms=int(yaw_ms) if yaw_ms is not None else None,
            roll_ms=int(roll_ms) if roll_ms is not None else None,
        )
        snap = _turret_snapshot()
        if replies:
            last = replies[-1]
            snap["reply"] = {"ok": last.ok, "kind": last.kind, "body": last.body}
        return snap
    elif action == "set_policy":
        if "auto_fire" in args:
            turret_policy.auto_fire = bool(args["auto_fire"])
        if "darts_per_event" in args:
            n = int(args["darts_per_event"])
            turret_policy.darts_per_event = max(1, min(2, n))
        if "cooldown_s" in args:
            turret.cooldown_s = max(0.0, min(300.0, float(args["cooldown_s"])))
        return _turret_snapshot()
    else:
        raise HTTPException(status_code=400, detail=f"unknown action: {action!r}")

    snap = _turret_snapshot()
    snap["reply"] = {"ok": reply.ok, "kind": reply.kind, "body": reply.body}
    return snap


@app.get("/stats")
async def stats() -> dict:
    af: AnnotatedFrame | None = state.annotated.value
    return {
        "frames_received": state.frames_received,
        "frames_processed": state.frames_processed,
        "last_inference_ms": round(state.last_inference_ms, 1),
        "ingest_connected": state.ingest_ws is not None,
        "latest": af.detections_payload() if af is not None else None,
    }


# --- websocket routes ---

@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket) -> None:
    """the phone connects here and pushes jpeg frames. we echo detections back."""
    await ws.accept()
    if state.ingest_ws is not None:
        # one source at a time - politely kick out the previous one
        with contextlib.suppress(Exception):
            await state.ingest_ws.close(code=1000, reason="superseded")
    state.ingest_ws = ws
    print(f"[ingest] phone connected from {ws.client}")
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                state.frames_received += 1
                state.raw.publish(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ingest] error: {e!r}")
    finally:
        if state.ingest_ws is ws:
            state.ingest_ws = None
        print("[ingest] phone disconnected")


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket) -> None:
    """viewers get (a) binary jpeg frame, then (b) a json detection payload,
    paired together with the same seq number. clients draw the image then the
    overlay once both arrive."""
    await ws.accept()
    print(f"[view] viewer connected from {ws.client}")
    last_seq = state.annotated.seq  # skip whatever was already there
    try:
        while True:
            try:
                seq, af = await asyncio.wait_for(
                    state.annotated.next_after(last_seq), timeout=10.0
                )
            except asyncio.TimeoutError:
                await ws.send_text(
                    json.dumps({"type": "heartbeat", "ingest_connected": state.ingest_ws is not None})
                )
                continue
            last_seq = seq
            assert isinstance(af, AnnotatedFrame)
            await ws.send_bytes(af.jpeg)
            await ws.send_text(json.dumps(af.detections_payload()))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[view] error: {e!r}")
    finally:
        print("[view] viewer disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
