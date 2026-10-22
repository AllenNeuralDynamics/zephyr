"""Running a ``zephyr`` command from the app: in a local child process, never remotely.

A :class:`Job` starts :mod:`.worker` with ``python -m``, so a crash, a CUDA error or
Cancel never takes the window down, and a stopped run resumes with ``--resume``. Its
output is read on a thread and drained with :meth:`Job.poll`; no Qt here, so jobs are
tested headless. :meth:`Job.command_line` is the same command for a terminal.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

from .worker import COMMANDS, MARK


class Job:
    """``zephyr <command> <argv...>`` run in *cwd* (snippets: ``run`` only)."""

    def __init__(
        self,
        command: str,
        argv: list[str],
        cwd: Path,
        *,
        snippets: dict | None = None,
    ) -> None:
        if command not in COMMANDS:
            raise ValueError(f"the app does not run `zephyr {command}`")
        self.command = command
        self.argv = [str(a) for a in argv]
        self.cwd = Path(cwd)
        self.snippets = snippets
        self.process: subprocess.Popen | None = None
        self.returncode: int | None = None
        self.cancelled = False
        self._queue: Queue = Queue()
        self._reader: threading.Thread | None = None
        self._reported = False

    def command_line(self) -> str:
        """The same command, to run in a terminal from :attr:`cwd`."""
        return shlex.join(["zephyr", self.command, *self.argv])

    def start(self) -> "Job":
        spec = {"command": self.command, "argv": self.argv, "snippets": self.snippets}
        env = os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        options: dict = {}
        if sys.platform == "win32":
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            options["start_new_session"] = True
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "zephyr.gui.worker", json.dumps(spec)],
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **options,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith(MARK):
                try:
                    self._queue.put(("snippets", json.loads(line[len(MARK) :])))
                except ValueError:
                    pass
            else:
                self._queue.put(("log", line.rstrip("\r\n")))
        self.returncode = self.process.wait()

    @property
    def running(self) -> bool:
        return self.process is not None and self.returncode is None

    def poll(self) -> list[tuple[str, object]]:
        """Messages since the last poll: ``("log", line)``, ``("snippets", dict)``
        and, once, ``("exit", returncode)`` after the last of them."""
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        finished = self._reader is not None and not self._reader.is_alive()
        if finished and self._queue.empty() and not self._reported:
            self._reported = True
            out.append(("exit", self.returncode))
        return out

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the job ends (tests and scripts); its return code."""
        if self._reader is not None:
            self._reader.join(timeout)
        return self.returncode

    def cancel(self) -> None:
        """Stop the job and every process it started (data loaders, ffmpeg)."""
        if not self.running:
            return
        self.cancelled = True
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
