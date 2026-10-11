"""Run evidence: a structured, redacted log plus richer artifacts on failure.

    runs/<run_id>/
      log.jsonl            one event per line: what happened, why, where
      NN-<label>.png       masked screenshots (every step in discovery, failures in replay)
      NN-<label>.aria.txt  redacted accessibility snapshot of every frame
      result.json          the RunResult, outputs redacted

Every record passes through the Redactor before it touches disk. The caller
gets real output values in the returned RunResult; the evidence never has them.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy import Redactor

RUNS = Path(__file__).resolve().parent.parent / "runs"


def new_run_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{kind}-{uuid.uuid4().hex[:6]}"


class RunLog:
    def __init__(self, run_id: str, redactor: Redactor, root: Path | str = RUNS,
                 echo: Any = None) -> None:
        self.run_id = run_id
        self.redactor = redactor
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.dir / "log.jsonl").open("a")
        self._start = time.monotonic()
        self._n = 0
        self.echo = echo        # optional callable(str) for live progress

    def event(self, kind: str, /, **fields: Any) -> None:
        record = {
            "t": round(time.monotonic() - self._start, 3),
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": kind,
            **self.redactor.obj(fields),
        }
        self._log.write(json.dumps(record, default=str) + "\n")
        self._log.flush()
        if self.echo:
            summary = " ".join(f"{k}={v}" for k, v in record.items()
                               if k not in ("t", "at", "event") and v not in (None, "", [], {}))
            self.echo(f"{kind:<14} {summary}"[:220])

    def _name(self, label: str, suffix: str) -> Path:
        self._n += 1
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
        return self.dir / f"{self._n:02d}-{safe}{suffix}"

    def capture(self, surface: Any, label: str, aria: bool = False) -> list[str]:
        """Screenshot (and optionally ARIA snapshot) of the live surface."""
        paths: list[str] = []
        try:
            paths.append(str(surface.screenshot(self._name(label, ".png"))))
        except Exception as e:
            self.event("evidence_error", what="screenshot", error=str(e))
        if aria:
            try:
                path = self._name(label, ".aria.txt")
                path.write_text(surface.accessibility_snapshot())
                paths.append(str(path))
            except Exception as e:
                self.event("evidence_error", what="aria", error=str(e))
        return [self.relative(p) for p in paths]

    def relative(self, path: str | Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.dir.resolve().parent.parent))
        except ValueError:
            return str(path)

    def write_json(self, name: str, data: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(self.redactor.obj(data), indent=2, default=str))
        return path

    def close(self) -> None:
        self._log.close()
