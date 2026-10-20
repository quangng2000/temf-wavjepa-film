"""Serve the live dashboard locally; remote access is read-only."""
import argparse
import json
from http.server import ThreadingHTTPServer

from tools.dashboard.http import Handler
from tools.dashboard.reader import Reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.reader = Reader(args.config)
    print(json.dumps(dict(url=f"http://127.0.0.1:{server.server_port}", read_only=True)),
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
