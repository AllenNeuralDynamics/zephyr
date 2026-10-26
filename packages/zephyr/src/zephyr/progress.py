"""A progress line for long steps (decoding a long video, predicting it).

On a terminal it is one line rewritten in place; elsewhere (a log, the app's job
output) a line at every tenth of the way, so nothing fills with carriage returns.
"""

import sys
import threading
import time

REDRAW_S = 0.5
"""Seconds between redraws of a terminal's line."""


class Progress:
    """Counts *total* *unit* of *label*'s work, from any thread, and shows how far."""

    def __init__(self, label: str, total: int, unit: str, stream=None) -> None:
        self.label, self.total, self.unit = label, max(int(total), 1), unit
        self.stream = stream or sys.stdout
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.done, self._shown, self._tenth = 0, 0.0, 0
        self.started = time.perf_counter()
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            now = time.perf_counter()
            tenth = min(10, self.done * 10 // self.total)
            if self.live and now - self._shown >= REDRAW_S:
                self._shown = now
                self._write("\r" + self.line(now))
            elif not self.live and tenth > self._tenth:
                self._tenth = tenth
                self._write(self.line(now) + "\n")

    def close(self) -> None:
        """The final line, once the work is done."""
        with self._lock:
            if self.live:
                self._write("\r" + self.line(time.perf_counter()) + "\n")

    def line(self, now: float) -> str:
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        left = (self.total - self.done) / rate if rate > 0 else None
        eta = f"{left // 60:.0f}:{left % 60:02.0f} left" if left is not None else ""
        return (
            f"  {self.label}: {self.done}/{self.total} {self.unit} "
            f"{100 * self.done / self.total:3.0f}% | {rate:,.0f} {self.unit}/s | "
            f"{elapsed // 60:.0f}:{elapsed % 60:02.0f} {eta}"
        )

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
