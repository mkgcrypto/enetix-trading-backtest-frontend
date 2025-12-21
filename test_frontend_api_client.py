#!/usr/bin/env python3
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from frontend.api_client import BackendApiError, cancel_job, list_jobs, request_json, run_backtest


class DummyResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        os.environ["BACKEND_URL"] = "http://example.test"

    def tearDown(self):
        os.environ.pop("BACKEND_URL", None)

    def test_request_json_success(self):
        def fake_urlopen(req, timeout):
            return DummyResponse(b'{"ok": true}')

        with patch("frontend.api_client.urlopen", fake_urlopen):
            resp = request_json("GET", "/health")
            self.assertEqual(resp, {"ok": True})

    def test_request_json_posts_json(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["method"] = req.get_method()
            captured["content_type"] = req.headers.get("Content-Type") or req.headers.get("Content-type")
            return DummyResponse(b'{"job": {"id": 1}}')

        with patch("frontend.api_client.urlopen", fake_urlopen):
            resp = request_json("POST", "/v1/optimize", payload={"a": 1})
            self.assertEqual(captured["method"], "POST")
            self.assertEqual(captured["content_type"], "application/json")
            self.assertIn("job", resp)

    def test_request_json_http_error(self):
        def fake_urlopen(req, timeout):
            fp = io.BytesIO(b'{"detail":"bad"}')
            raise HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=fp)

        with patch("frontend.api_client.urlopen", fake_urlopen):
            with self.assertRaises(BackendApiError) as ctx:
                request_json("GET", "/v1/jobs/999")
            self.assertEqual(ctx.exception.status, 400)
            self.assertIn("Bad Request", str(ctx.exception))

    def test_request_json_url_error(self):
        def fake_urlopen(req, timeout):
            raise URLError("no route")

        with patch("frontend.api_client.urlopen", fake_urlopen):
            with self.assertRaises(BackendApiError) as ctx:
                request_json("GET", "/health")
            self.assertEqual(ctx.exception.status, 0)

    def test_list_jobs_query_params(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            return DummyResponse(b'{"jobs": []}')

        with patch("frontend.api_client.urlopen", fake_urlopen):
            jobs = list_jobs(status="running", job_type="optimize", limit=25, offset=50)
            self.assertEqual(jobs, [])
            self.assertIn("/v1/jobs?", captured["url"])
            self.assertIn("status=running", captured["url"])
            self.assertIn("job_type=optimize", captured["url"])
            self.assertIn("limit=25", captured["url"])
            self.assertIn("offset=50", captured["url"])

    def test_cancel_job(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            return DummyResponse(b'{"id": 123, "status": "canceled"}')

        with patch("frontend.api_client.urlopen", fake_urlopen):
            job = cancel_job(123)
            self.assertEqual(captured["method"], "POST")
            self.assertIn("/v1/jobs/123/cancel", captured["url"])
            self.assertEqual(job.get("status"), "canceled")

    def test_run_backtest(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            payload = json.loads(req.data.decode("utf-8"))
            captured["payload"] = payload
            return DummyResponse(b'{"stats": {"ok": true}, "df": null, "ds": null, "trades": null, "equity_curve": [], "params": {}}')

        with patch("frontend.api_client.urlopen", fake_urlopen):
            resp = run_backtest({"exchange": "bitstamp"})
            self.assertEqual(captured["method"], "POST")
            self.assertIn("/v1/backtest", captured["url"])
            self.assertIn("params", captured["payload"])
            self.assertIn("stats", resp)


if __name__ == "__main__":
    unittest.main()
