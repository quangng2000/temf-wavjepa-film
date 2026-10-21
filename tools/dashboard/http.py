"""Loopback-only HTTP surface with no training or resource controls."""
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

WEB = Path(__file__).with_name("web")
STATIC = {"/": ("index.html", "text/html"), "/styles.css": ("styles.css", "text/css")}
STATIC.update({"/" + name: (name, "text/javascript") for name in
               ("app.js", "render.js", "charts.js", "format.js")})


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin not in {"http://" + h for h in allowed}):
            self.send_error(403)
            return
        path = urlparse(self.path).path
        if path == "/api/status":
            self.respond(json.dumps(self.server.reader.fetch(), allow_nan=False).encode(),
                         "application/json")
        elif path == "/health":
            self.respond(b'{"service":"temf-live-dashboard","read_only":true}', "application/json")
        elif path in STATIC:
            name, mime = STATIC[path]
            self.respond((WEB / name).read_bytes(), mime)
        else:
            self.send_error(404)

    def respond(self, body, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass
