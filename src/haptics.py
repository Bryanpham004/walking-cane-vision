"""Vibration-motor output over the existing Arduino serial link.

The cane already drives its motors from an Arduino, so this module speaks to
that board rather than replacing it. Wire protocol, newline-terminated ASCII:

    V<zone>,<intensity>,<duration_ms>\\n   -> pulse one motor
    S\\n                                   -> stop all motors
    P\\n                                   -> ping, board replies "OK"

Zones map to motors by position: 0 = left grip, 1 = centre, 2 = right grip.
Keeping the protocol printable means you can drive it from the Arduino serial
monitor while debugging, with no host software running.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .alerts import Alert
from .config import HapticConfig
from .geometry import Zone

ZONE_MOTOR: dict[Zone, int] = {Zone.LEFT: 0, Zone.CENTER: 1, Zone.RIGHT: 2}


@runtime_checkable
class HapticSink(Protocol):
    def pulse(self, alert: Alert) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def encode_pulse(alert: Alert, config: HapticConfig) -> str:
    """Build the wire command for an alert. Pure, so it is easy to test."""
    motor = ZONE_MOTOR[alert.zone]
    intensity = config.intensity.get(alert.urgency.value, 128)
    duration = config.pulse_ms.get(alert.urgency.value, 150)
    return f"V{motor},{intensity},{duration}\n"


class SerialHaptics:
    """Real Arduino over USB/UART."""

    def __init__(self, config: HapticConfig) -> None:
        self._config = config
        self._port = None

    def _ensure_port(self):
        if self._port is None:
            try:
                import serial  # pyserial
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "pyserial is required for SerialHaptics; "
                    "pip install pyserial, or use NullHaptics for a dry run"
                ) from exc
            self._port = serial.Serial(
                self._config.port, self._config.baud, timeout=self._config.timeout_s
            )
        return self._port

    def pulse(self, alert: Alert) -> None:
        self._ensure_port().write(encode_pulse(alert, self._config).encode("ascii"))

    def stop(self) -> None:
        self._ensure_port().write(b"S\n")

    def ping(self) -> bool:
        """Confirm the board is alive and speaking the expected protocol."""
        port = self._ensure_port()
        port.write(b"P\n")
        return port.readline().strip() == b"OK"

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.write(b"S\n")
            finally:
                self._port.close()
                self._port = None


class NullHaptics:
    """No-op sink that records commands, for dry runs and tests."""

    def __init__(self, config: Optional[HapticConfig] = None) -> None:
        self._config = config or HapticConfig()
        self.commands: list[str] = []
        self.closed = False

    def pulse(self, alert: Alert) -> None:
        self.commands.append(encode_pulse(alert, self._config).strip())

    def stop(self) -> None:
        self.commands.append("S")

    def close(self) -> None:
        self.closed = True
