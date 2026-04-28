"""async serial client for the taiga turret firmware.

responsibilities:
  - find the arduino's tty on startup and keep a connection open
  - serialize commands so two tasks can't talk to the turret at once
  - expose a simple async api: fire(), arm(), disarm(), yaw(), pitch(), home(), status()
  - hold the "fire cooldown" invariant in software so two rapid-fire triggers
    can't mag-dump the cat

the firmware prints "OK <echo>" or "ERR <reason>" in reply to every command,
plus unsolicited "LOG ..." and "STATE ..." lines on reset or STATUS. we read
everything until we see an OK/ERR line and return that as the result.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import serial
from serial.tools import list_ports

logger = logging.getLogger("turret")

# --- port detection ---

def find_turret_ports() -> list[str]:
    """heuristic search for an arduino-shaped serial device.

    the crunchlabs board shows up as a usb cdc device (/dev/ttyACM*) on linux.
    generic USB-serial chips like CH340/FTDI show up as /dev/ttyUSB*. we return
    all candidates in preference order; the first that successfully opens wins.
    """
    candidates: list[str] = []
    # pyserial's own enumerator - prefer devices with arduino-ish descriptors
    for port in list_ports.comports():
        desc = (port.description or "").lower()
        manufacturer = (port.manufacturer or "").lower()
        if any(k in desc for k in ("arduino", "nano", "ch340", "cp210", "ft232"))         \
           or any(k in manufacturer for k in ("arduino", "wch", "ftdi", "silicon labs")):
            candidates.append(port.device)
    # fall back to globbing /dev for anything that looks like a serial tty
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        for p in sorted(glob.glob(pattern)):
            if p not in candidates:
                candidates.append(p)
    return candidates


# --- protocol parsing ---

# "OK FIRE 1 darts_left=5" -> ("OK", "FIRE 1 darts_left=5")
RESPONSE_RE = re.compile(r"^(OK|ERR|STATE|LOG)\s*(.*)$")


@dataclass
class TurretReply:
    kind: str           # "OK", "ERR", "STATE"
    body: str
    extras: list[str]   # any STATE/LOG lines seen before the OK/ERR

    @property
    def ok(self) -> bool:
        return self.kind == "OK"


@dataclass
class TurretState:
    connected: bool = False
    port: Optional[str] = None
    armed: bool = True
    darts_estimate: int = 6
    pitch: int = 100
    yaw_ms: int = 0           # raw cumulative *commanded* ms (matches firmware's count)
    eff_yaw_ms: int = 0       # *effective* ms after subtracting per-pulse deadband
    yaw_pulse_ms: int = 350   # default duration of a single yaw pulse
    roll_pulse_ms: int = 250  # duration of the roll pulse per dart
    last_fire_ts: float = 0.0
    last_error: Optional[str] = None
    firmware_boot_seen: bool = False

    def as_dict(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.port,
            "armed": self.armed,
            "darts_estimate": self.darts_estimate,
            "pitch": self.pitch,
            "yaw_ms": self.yaw_ms,
            "eff_yaw_ms": self.eff_yaw_ms,
            "yaw_pulse_ms": self.yaw_pulse_ms,
            "roll_pulse_ms": self.roll_pulse_ms,
            "last_fire_age_s": (
                round(time.time() - self.last_fire_ts, 2) if self.last_fire_ts else None
            ),
            "last_error": self.last_error,
        }


STATE_KV_RE = re.compile(r"(\w+)=(-?\d+)")


def _apply_state_line(state: TurretState, body: str) -> None:
    """parse 'pitch=100 darts=6 armed=1 yaw_ms=0 yaw_pulse=350 roll_pulse=250' into state fields."""
    kvs = dict(STATE_KV_RE.findall(body))
    if "pitch" in kvs:
        state.pitch = int(kvs["pitch"])
    if "darts" in kvs:
        state.darts_estimate = int(kvs["darts"])
    if "armed" in kvs:
        state.armed = bool(int(kvs["armed"]))
    if "yaw_ms" in kvs:
        state.yaw_ms = int(kvs["yaw_ms"])
    if "yaw_pulse" in kvs:
        state.yaw_pulse_ms = int(kvs["yaw_pulse"])
    if "roll_pulse" in kvs:
        state.roll_pulse_ms = int(kvs["roll_pulse"])


# --- the client ---

class Turret:
    """async wrapper over a pyserial connection.

    usage:
        t = Turret(cooldown_s=10)
        await t.connect()
        await t.fire(1)
        await t.disconnect()

    threading note: pyserial's read/write are blocking, so we push them to a
    thread pool via asyncio.to_thread. the `_io_lock` serializes commands so
    we never have two overlapping send/reply pairs on the same port.
    """

    # boards often reset when the serial port is opened; give the bootloader
    # time to jump into the sketch and print the boot banner before we talk.
    BOOT_GRACE_S = 2.0
    # max time to wait for OK/ERR after a command. fire() can delay ~160ms per dart.
    COMMAND_TIMEOUT_S = 4.0

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 9600,
        cooldown_s: float = 10.0,
        yaw_deadband_l_ms: int = 60,
        yaw_deadband_r_ms: int = 60,
    ) -> None:
        self._explicit_port = port
        self._baud = baud
        self.cooldown_s = cooldown_s
        # continuous-rotation servos lose some ms at the start of every pulse
        # to startup lag, AND that loss is usually asymmetric between L/R
        # (manufacturing slop, gear backlash, weight bias from a cable).
        # we keep a separate deadband per direction. "effective" yaw motion
        # subtracts the relevant direction's deadband from each pulse.
        self.yaw_deadband_l_ms = yaw_deadband_l_ms
        self.yaw_deadband_r_ms = yaw_deadband_r_ms
        self.state = TurretState()
        self._ser: Optional[serial.Serial] = None
        self._io_lock = asyncio.Lock()

    def _deadband_for(self, direction: str) -> int:
        return self.yaw_deadband_l_ms if direction.upper() == "L" else self.yaw_deadband_r_ms

    # --- connect / disconnect ---

    async def connect(self) -> bool:
        ports: Iterable[str] = [self._explicit_port] if self._explicit_port else find_turret_ports()
        last_err: Optional[Exception] = None
        for port in ports:
            try:
                ser = await asyncio.to_thread(
                    serial.Serial, port, self._baud, timeout=0.2, write_timeout=1.0,
                )
            except Exception as e:
                last_err = e
                continue
            self._ser = ser
            self.state.port = port
            self.state.connected = True
            logger.info("turret: connected on %s", port)
            # drain boot banner + STATE line
            await asyncio.sleep(self.BOOT_GRACE_S)
            await self._drain_into_state()
            # ask for fresh status so we know for sure what state the firmware is in
            try:
                reply = await self._send_raw("STATUS")
                if reply.ok:
                    for extra in reply.extras:
                        if extra.startswith("STATE "):
                            _apply_state_line(self.state, extra[6:])
            except Exception as e:
                logger.warning("turret: STATUS probe failed: %s", e)
            return True
        self.state.last_error = f"no serial port openable ({last_err})"
        logger.error("turret: connect failed: %s", self.state.last_error)
        return False

    async def disconnect(self) -> None:
        ser = self._ser
        self._ser = None
        self.state.connected = False
        if ser is not None:
            await asyncio.to_thread(ser.close)

    # --- raw protocol ---

    async def _drain_into_state(self) -> list[str]:
        """read whatever is in the buffer (non-blocking-ish). returns extra lines."""
        if self._ser is None:
            return []
        def _read_all() -> list[str]:
            assert self._ser is not None
            lines: list[str] = []
            while self._ser.in_waiting:
                line = self._ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    lines.append(line)
            return lines
        lines = await asyncio.to_thread(_read_all)
        for line in lines:
            m = RESPONSE_RE.match(line)
            if not m:
                continue
            kind, body = m.group(1), m.group(2)
            if kind == "STATE":
                _apply_state_line(self.state, body)
            elif kind == "LOG":
                if "ready" in body:
                    self.state.firmware_boot_seen = True
        return lines

    async def _send_raw(self, command: str) -> TurretReply:
        if self._ser is None:
            raise RuntimeError("not connected")
        async with self._io_lock:
            def _write_and_read() -> tuple[str, list[str]]:
                assert self._ser is not None
                # flush any stale bytes first (unsolicited LOG/STATE while idle)
                self._ser.reset_input_buffer()
                self._ser.write((command + "\n").encode("utf-8"))
                self._ser.flush()
                extras: list[str] = []
                deadline = time.monotonic() + self.COMMAND_TIMEOUT_S
                while time.monotonic() < deadline:
                    raw = self._ser.readline().decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    m = RESPONSE_RE.match(raw)
                    if not m:
                        extras.append(raw)
                        continue
                    kind = m.group(1)
                    if kind in ("OK", "ERR"):
                        return f"{kind} {m.group(2)}", extras
                    extras.append(raw)
                raise asyncio.TimeoutError(f"timeout waiting for OK/ERR to {command!r}")
            raw_reply, extras = await asyncio.to_thread(_write_and_read)

        m = RESPONSE_RE.match(raw_reply)
        assert m is not None
        kind, body = m.group(1), m.group(2)
        reply = TurretReply(kind=kind, body=body, extras=extras)

        # fold any STATE lines into our cached state
        for line in extras + ([f"{kind} {body}"] if kind == "STATE" else []):
            em = RESPONSE_RE.match(line)
            if em and em.group(1) == "STATE":
                _apply_state_line(self.state, em.group(2))

        if not reply.ok:
            self.state.last_error = f"{command!r}: {body}"
            logger.warning("turret: %s", self.state.last_error)
        else:
            self.state.last_error = None
        return reply

    # --- high-level api ---

    async def status(self) -> TurretReply:
        return await self._send_raw("STATUS")

    async def home(self) -> TurretReply:
        r = await self._send_raw("HOME")
        if r.ok:
            self.state.pitch = 100
            self.state.yaw_ms = 0
            self.state.eff_yaw_ms = 0
        return r

    async def arm(self) -> TurretReply:
        r = await self._send_raw("ARM")
        if r.ok:
            self.state.armed = True
        return r

    async def disarm(self) -> TurretReply:
        r = await self._send_raw("DISARM")
        if r.ok:
            self.state.armed = False
        return r

    async def reload(self) -> TurretReply:
        r = await self._send_raw("RELOAD")
        if r.ok:
            self.state.darts_estimate = 6
        return r

    async def pitch_abs(self, angle: int) -> TurretReply:
        reply = await self._send_raw(f"PITCH {int(angle)}")
        if reply.ok:
            m = re.search(r"PITCH\s+(\d+)", reply.body)
            if m:
                self.state.pitch = int(m.group(1))
        return reply

    async def pitch_rel(self, delta: int) -> TurretReply:
        reply = await self._send_raw(f"PITCH_REL {int(delta)}")
        if reply.ok:
            m = re.search(r"PITCH_REL\s+(\d+)", reply.body)
            if m:
                self.state.pitch = int(m.group(1))
        return reply

    async def yaw(self, direction: str, ms: int = 150) -> TurretReply:
        direction = direction.upper()
        if direction not in ("L", "R"):
            raise ValueError("direction must be 'L' or 'R'")
        reply = await self._send_raw(f"YAW {direction} {int(ms)}")
        if reply.ok:
            sign = 1 if direction == "L" else -1
            # commanded ms, matches the firmware's own counter
            self.state.yaw_ms += sign * ms
            # effective ms: subtract the per-direction startup deadband.
            # never negative per pulse.
            effective = max(0, int(ms) - self._deadband_for(direction))
            self.state.eff_yaw_ms += sign * effective
        return reply

    # minimum *effective* yaw delta worth acting on. below this the servo
    # wouldn't move meaningfully anyway (after adding deadband, pulse becomes
    # too close to stuff that barely overcomes static friction).
    MIN_EFF_YAW_DELTA_MS = 15

    async def aim_to(
        self,
        pitch: int,
        eff_yaw_ms: int,
        *,
        settle_s: float = 0.3,
    ) -> TurretReply:
        """move to an absolute pose (pitch angle, cumulative EFFECTIVE yaw ms).

        `eff_yaw_ms` is deadband-compensated: 1 unit of effective ms
        corresponds to 1 unit of actual rotation, roughly. we compute the
        effective delta from the current dead-reckoned position and send a
        single YAW pulse of `|delta| + yaw_deadband_ms` ms so that the
        motor's startup lag doesn't eat the motion.
        """
        reply = await self.pitch_abs(pitch)
        if not reply.ok:
            return reply
        delta = eff_yaw_ms - self.state.eff_yaw_ms
        if abs(delta) >= self.MIN_EFF_YAW_DELTA_MS:
            direction = "L" if delta > 0 else "R"
            # to move `|delta|` of real rotation, pay the direction-specific startup tax once.
            pulse_ms = abs(delta) + self._deadband_for(direction)
            reply = await self.yaw(direction, ms=pulse_ms)
            if not reply.ok:
                return reply
        if settle_s > 0:
            await asyncio.sleep(settle_s)
        return reply

    async def set_timing(self, *, yaw_ms: int | None = None, roll_ms: int | None = None) -> list[TurretReply]:
        """push one or both of the runtime-tunable pulse widths to firmware."""
        replies: list[TurretReply] = []
        if yaw_ms is not None:
            r = await self._send_raw(f"SET YAW_MS {int(yaw_ms)}")
            if r.ok:
                m = re.search(r"YAW_MS\s+(\d+)", r.body)
                if m:
                    self.state.yaw_pulse_ms = int(m.group(1))
            replies.append(r)
        if roll_ms is not None:
            r = await self._send_raw(f"SET ROLL_MS {int(roll_ms)}")
            if r.ok:
                m = re.search(r"ROLL_MS\s+(\d+)", r.body)
                if m:
                    self.state.roll_pulse_ms = int(m.group(1))
            replies.append(r)
        return replies

    def cooldown_remaining(self, now: Optional[float] = None) -> float:
        """seconds remaining before the next fire() is allowed. 0 = ready."""
        now = now if now is not None else time.time()
        due = self.state.last_fire_ts + self.cooldown_s
        return max(0.0, due - now)

    async def fire(self, count: int = 1, *, override_cooldown: bool = False) -> TurretReply:
        """fire `count` darts if armed, connected, and cooldown has elapsed.

        returns an ERR reply (without hitting hardware) if cooldown blocks the
        shot, so callers get structured feedback without having to check
        cooldown themselves.
        """
        if not self.state.connected:
            return TurretReply(kind="ERR", body="NOT_CONNECTED", extras=[])
        if not self.state.armed:
            return TurretReply(kind="ERR", body="DISARMED", extras=[])
        if self.state.darts_estimate <= 0:
            return TurretReply(kind="ERR", body="EMPTY_MAG", extras=[])
        remaining = self.cooldown_remaining()
        if remaining > 0 and not override_cooldown:
            return TurretReply(
                kind="ERR",
                body=f"COOLDOWN {remaining:.1f}s",
                extras=[],
            )
        count = max(1, min(2, int(count)))
        reply = await self._send_raw(f"FIRE {count}")
        if reply.ok:
            self.state.last_fire_ts = time.time()
            # firmware echoes "FIRE <fired> darts_left=N" - pull the remaining count
            m = re.search(r"darts_left=(\d+)", reply.body)
            if m:
                self.state.darts_estimate = int(m.group(1))
            else:
                self.state.darts_estimate = max(0, self.state.darts_estimate - count)
        return reply
