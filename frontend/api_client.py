from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BackendApiError(Exception):
    status: int
    message: str
    body: str | None = None

    def __str__(self) -> str:
        base = f"Backend API error {self.status}: {self.message}"
        return f"{base} ({self.body})" if self.body else base


def backend_url() -> str:
    return (os.getenv("BACKEND_URL") or "http://localhost:8000").rstrip("/")


@dataclass(frozen=True)
class BackendApiClient:
    base_url: str
    timeout_seconds: float = 10.0

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, default=str).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(url, data=data, headers=headers, method=method.upper())
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)

        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except HTTPError as e:
            body = None
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = None
            raise BackendApiError(status=int(e.code), message=str(e.reason), body=body) from e
        except URLError as e:
            raise BackendApiError(status=0, message=str(e.reason)) from e

    def enqueue_optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.request_json("POST", "/v1/optimize", payload=payload)
        if not isinstance(resp, dict) or "job" not in resp:
            raise BackendApiError(status=0, message="Invalid optimize response")
        return resp["job"]

    def enqueue_discover(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.request_json("POST", "/v1/discover", payload=payload)
        if not isinstance(resp, dict) or "job" not in resp:
            raise BackendApiError(status=0, message="Invalid discover response")
        return resp["job"]

    def enqueue_leaderboard_refresh(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = self.request_json("POST", "/v1/leaderboard/refresh", payload=(payload or {}))
        if not isinstance(resp, dict) or "job" not in resp:
            raise BackendApiError(status=0, message="Invalid leaderboard refresh response")
        return resp["job"]

    def enqueue_patterns_refresh(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = self.request_json("POST", "/v1/patterns/refresh", payload=(payload or {}))
        if not isinstance(resp, dict) or "job" not in resp:
            raise BackendApiError(status=0, message="Invalid patterns refresh response")
        return resp["job"]

    def run_backtest(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.request_json("POST", "/v1/backtest", payload={"params": params})
        if not isinstance(resp, dict) or "stats" not in resp:
            raise BackendApiError(status=0, message="Invalid backtest response")
        return resp

    def get_job(self, job_id: int) -> dict[str, Any]:
        resp = self.request_json("GET", f"/v1/jobs/{int(job_id)}")
        if not isinstance(resp, dict) or "id" not in resp:
            raise BackendApiError(status=0, message="Invalid job response")
        return resp

    def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        qs = [f"limit={int(limit)}", f"offset={int(offset)}"]
        if status:
            qs.append(f"status={status}")
        if job_type:
            qs.append(f"job_type={job_type}")
        path = f"/v1/jobs?{'&'.join(qs)}"
        resp = self.request_json("GET", path)
        if not isinstance(resp, dict) or "jobs" not in resp:
            raise BackendApiError(status=0, message="Invalid jobs list response")
        jobs = resp["jobs"]
        return list(jobs) if isinstance(jobs, list) else []

    def get_job_events(self, job_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
        resp = self.request_json("GET", f"/v1/jobs/{int(job_id)}/events?limit={int(limit)}")
        if not isinstance(resp, dict) or "events" not in resp:
            raise BackendApiError(status=0, message="Invalid job events response")
        events = resp["events"]
        return list(events) if isinstance(events, list) else []

    def cancel_job(self, job_id: int) -> dict[str, Any]:
        resp = self.request_json("POST", f"/v1/jobs/{int(job_id)}/cancel", payload={})
        if not isinstance(resp, dict) or "id" not in resp:
            raise BackendApiError(status=0, message="Invalid cancel job response")
        return resp

    def get_discovery_stats(self) -> dict[str, Any]:
        resp = self.request_json("GET", "/v1/discovery/stats")
        if not isinstance(resp, dict) or "stats" not in resp:
            raise BackendApiError(status=0, message="Invalid discovery stats response")
        return resp["stats"]

    def get_patterns(self, *, min_confidence: float = 0.3) -> list[dict[str, Any]]:
        resp = self.request_json("GET", f"/v1/patterns?min_confidence={float(min_confidence)}")
        if not isinstance(resp, dict) or "rules" not in resp:
            raise BackendApiError(status=0, message="Invalid patterns response")
        rules = resp["rules"]
        return list(rules) if isinstance(rules, list) else []

    def get_leaderboard_latest(self) -> dict[str, Any]:
        resp = self.request_json("GET", "/v1/leaderboard")
        if not isinstance(resp, dict) or "payload" not in resp:
            raise BackendApiError(status=0, message="Invalid leaderboard response")
        return resp


def _default_client() -> BackendApiClient:
    return BackendApiClient(backend_url())


def request_json(
    method: str,
    path: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    timeout_seconds: float = 10.0,
) -> Any:
    return _default_client().request_json(
        method,
        path,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )


def enqueue_optimize(payload: dict[str, Any]) -> dict[str, Any]:
    return _default_client().enqueue_optimize(payload)


def enqueue_discover(payload: dict[str, Any]) -> dict[str, Any]:
    return _default_client().enqueue_discover(payload)


def enqueue_leaderboard_refresh(payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _default_client().enqueue_leaderboard_refresh(payload)


def enqueue_patterns_refresh(payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _default_client().enqueue_patterns_refresh(payload)


def run_backtest(params: dict[str, Any]) -> dict[str, Any]:
    return _default_client().run_backtest(params)


def get_job(job_id: int) -> dict[str, Any]:
    return _default_client().get_job(job_id)


def list_jobs(
    *,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return _default_client().list_jobs(status=status, job_type=job_type, limit=limit, offset=offset)


def get_job_events(job_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
    return _default_client().get_job_events(job_id, limit=limit)


def cancel_job(job_id: int) -> dict[str, Any]:
    return _default_client().cancel_job(job_id)


def get_discovery_stats() -> dict[str, Any]:
    return _default_client().get_discovery_stats()


def get_patterns(*, min_confidence: float = 0.3) -> list[dict[str, Any]]:
    return _default_client().get_patterns(min_confidence=min_confidence)


def get_leaderboard_latest() -> dict[str, Any]:
    return _default_client().get_leaderboard_latest()
