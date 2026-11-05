# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
HTTP servers run from a background thread, for tests that need a system under
test to fetch something from the test host.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from urllib.parse import quote, unquote

_log = logging.getLogger(__name__)


class BackgroundHTTPServer:
    """
    A ThreadingHTTPServer on an OS-assigned port, served from a daemon thread.

    Args:
        handler: Request handler class, or a factory such as functools.partial over one.
        host: Address the system under test reaches this host on; used to build base_url.
        bind: Local address to listen on.
        name: Name of the serving thread, also used in log messages.
    """

    def __init__(
        self,
        handler: Callable[..., BaseHTTPRequestHandler],
        host: str,
        *,
        bind: str = "0.0.0.0",
        name: str = "http-server",
    ) -> None:
        self.host = host
        self.name = name
        self._httpd = ThreadingHTTPServer((bind, 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name=name, daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> Self:
        self._thread.start()
        _log.info(f"{self.name} serving on {self.base_url}")
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


class _InMemoryFileHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - signature defined by the base class
        _log.debug("in-memory file server: " + format, *args)

    def do_HEAD(self) -> None:  # noqa: N802 - name defined by the base class
        self._serve(send_body=False)

    def do_GET(self) -> None:  # noqa: N802 - name defined by the base class
        self._serve(send_body=True)

    def _serve(self, *, send_body: bool) -> None:
        owner: InMemoryFileServer = self.server.file_server  # type: ignore[attr-defined]
        route, _, name = self.path.lstrip("/").partition("/")
        data = owner.files.get(unquote(name))
        if route not in ("files", "slow") or data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not send_body:
            return
        try:
            if route == "files":
                self.wfile.write(data)
                return
            for offset in range(0, len(data), owner.slow_chunk):
                if owner.stopping.is_set():
                    return
                self.wfile.write(data[offset : offset + owner.slow_chunk])
                self.wfile.flush()
                time.sleep(owner.slow_delay_s)
        except BrokenPipeError, ConnectionResetError:
            _log.debug(f"in-memory file server: client closed {self.path} early")


class InMemoryFileServer(BackgroundHTTPServer):
    """
    Serves byte strings held in memory.

    /files/<name> serves a file at full speed, /slow/<name> trickles it out in
    slow_chunk pieces every slow_delay_s so a download stays in progress, and any
    other path is a 404.
    """

    def __init__(
        self,
        host: str,
        *,
        bind: str = "0.0.0.0",
        slow_chunk: int = 64 * 1024,
        slow_delay_s: float = 0.25,
        name: str = "in-memory-file-server",
    ) -> None:
        super().__init__(_InMemoryFileHandler, host, bind=bind, name=name)
        self.files: dict[str, bytes] = {}
        self.slow_chunk = slow_chunk
        self.slow_delay_s = slow_delay_s
        self.stopping = threading.Event()
        self._httpd.file_server = self  # type: ignore[attr-defined]

    def stop(self) -> None:
        self.stopping.set()
        super().stop()

    def add_file(self, name: str, data: bytes) -> str:
        """Serve data as name and return its full-speed URL."""
        self.files[name] = data
        return self.url(name)

    def remove_file(self, name: str) -> None:
        self.files.pop(name, None)

    def url(self, name: str) -> str:
        return f"{self.base_url}/files/{quote(name)}"

    def slow_url(self, name: str) -> str:
        return f"{self.base_url}/slow/{quote(name)}"

    def missing_url(self, name: str) -> str:
        return f"{self.base_url}/files/missing-{quote(name)}"
