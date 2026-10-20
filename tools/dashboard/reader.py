"""Cache read-only snapshots so multiple tabs never duplicate SSH polling."""
import subprocess
import threading
import time

from tools.dashboard.config import load_config
from tools.dashboard.pace import Pace
from tools.dashboard.summary import summarize
from tools.dashboard.transport import read_snapshot


class Reader:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.pace = Pace(self.config["observations_file"])
        self.lock, self.cached, self.last_attempt = threading.Lock(), None, 0

    def fetch(self):
        with self.lock:
            if self.cached and time.monotonic() - self.last_attempt < 8:
                return self.cached
            self.last_attempt = time.monotonic()
            try:
                raw = read_snapshot(self.config)
                if raw.get("metrics_error"):
                    raise ValueError("Metrics could not be read")
                self.cached = summarize(raw, self.config, self.pace)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                self.cached = dict(self.cached or {}, connection_error=
                                   "Could not refresh the pod. "
                                   "Showing the last successful reading.",
                                   attempted_at=time.time())
            return self.cached
