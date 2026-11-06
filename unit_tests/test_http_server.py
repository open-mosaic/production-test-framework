# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for http_server module."""

from http.server import BaseHTTPRequestHandler

import pytest
import requests

from production_test_framework.http_server import BackgroundHTTPServer, InMemoryFileServer


class _HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"hello")

    def log_message(self, format, *args):  # noqa: A002
        pass


class TestBackgroundHTTPServer:
    """Tests for BackgroundHTTPServer."""

    def test_serves_until_stopped(self):
        with BackgroundHTTPServer(_HelloHandler, "127.0.0.1", bind="127.0.0.1") as server:
            assert server.base_url == f"http://127.0.0.1:{server.port}"
            assert requests.get(server.base_url, timeout=5).text == "hello"
        with pytest.raises(requests.ConnectionError):
            requests.get(server.base_url, timeout=2)


class TestInMemoryFileServer:
    """Tests for InMemoryFileServer."""

    @pytest.fixture
    def server(self):
        with InMemoryFileServer("127.0.0.1", bind="127.0.0.1", slow_chunk=1024, slow_delay_s=0.001) as server:
            yield server

    def test_serves_file_with_length(self, server):
        url = server.add_file("an image.img", b"x" * 5000)

        resp = requests.get(url, timeout=5)

        assert url.endswith("/files/an%20image.img")
        assert resp.status_code == 200
        assert resp.content == b"x" * 5000
        assert resp.headers["Content-Length"] == "5000"

    def test_head_sends_no_body(self, server):
        resp = requests.head(server.add_file("a.img", b"abc"), timeout=5)
        assert resp.status_code == 200
        assert resp.headers["Content-Length"] == "3"
        assert resp.content == b""

    def test_slow_route_serves_whole_file(self, server):
        server.add_file("a.img", bytes(range(256)) * 20)
        assert requests.get(server.slow_url("a.img"), timeout=5).content == bytes(range(256)) * 20

    @pytest.mark.parametrize("path", ["/files/unknown.img", "/other/a.img", "/"])
    def test_unknown_paths_are_404(self, server, path):
        server.add_file("a.img", b"abc")
        assert requests.get(f"{server.base_url}{path}", timeout=5).status_code == 404

    def test_missing_url_and_removed_file_are_404(self, server):
        server.add_file("a.img", b"abc")
        assert requests.get(server.missing_url("a.img"), timeout=5).status_code == 404
        server.remove_file("a.img")
        assert requests.get(server.url("a.img"), timeout=5).status_code == 404
