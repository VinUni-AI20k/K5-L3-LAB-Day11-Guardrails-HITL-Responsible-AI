"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request; caller may supply a unique ID for concurrent users."""
        from guardrails.output_guardrails import content_filter
        key = request_id or user_id
        if key in self._open:
            raise ValueError("Request already open; use a unique request_id")
        self._open[key] = {
            "request_id": request_id or uuid4().hex,
            "user_id": user_id,
            "input": content_filter(text)["redacted"],
            "started_at": utc_now_iso(),
            "started_monotonic": time.monotonic(),
        }
        return self._open[key]["request_id"]

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a request, redacting logs as well as the customer response."""
        from guardrails.output_guardrails import content_filter
        key = request_id or user_id
        pending = self._open.get(key)
        if pending is None or pending["user_id"] != user_id:
            raise ValueError("No matching request for this user")
        row = self._open.pop(key)
        start = row.pop("started_monotonic")
        row.update(output=content_filter(text)["redacted"], blocked=blocked,
                   layer=layer, finished_at=utc_now_iso(),
                   latency_ms=round((time.monotonic() - start) * 1000, 3))
        self.logs.append(row)
        return row

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
