"""Spoken announcements.

Speech is the high-bandwidth channel ("chair to your left, 1.2 metres") but it
is slow, so the pipeline only speaks when haptics alone would be ambiguous.
Announcements are dropped rather than queued when one is already playing: a
backlog of stale distances is worse than silence while walking.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .alerts import Alert


@runtime_checkable
class AudioSink(Protocol):
    def announce(self, alert: Alert) -> None: ...
    def close(self) -> None: ...


class TextToSpeechAudio:
    """Offline TTS via pyttsx3 - no network, which matters on a mobility aid."""

    def __init__(self, rate_wpm: int = 190, volume: float = 1.0) -> None:
        self._rate = rate_wpm
        self._volume = volume
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "pyttsx3 is required for TextToSpeechAudio; "
                    "pip install pyttsx3, or use ConsoleAudio for a dry run"
                ) from exc
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            self._engine = engine
        return self._engine

    def announce(self, alert: Alert) -> None:
        engine = self._ensure_engine()
        if engine.isBusy():
            # Drop rather than queue: by the time a backlogged phrase plays,
            # the distance it names is wrong.
            return
        engine.say(alert.message)
        engine.runAndWait()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None


class ConsoleAudio:
    """Prints instead of speaking - the default for development."""

    def __init__(self, stream=None) -> None:
        self._stream = stream
        self.spoken: list[str] = []
        self.closed = False

    def announce(self, alert: Alert) -> None:
        caveat = "" if alert.distance_is_calibrated else "  (uncalibrated)"
        line = f"[{alert.urgency.value:>8}] {alert.message}{caveat}"
        self.spoken.append(alert.message)
        if self._stream is not None:
            print(line, file=self._stream)
        else:
            print(line)

    def close(self) -> None:
        self.closed = True


def build_audio(speak: bool) -> AudioSink:
    """Real speech when asked for, console output otherwise."""
    return TextToSpeechAudio() if speak else ConsoleAudio()
