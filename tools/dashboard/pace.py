"""Observed wall-clock pace; never invent individual optimizer timings."""
import json
from pathlib import Path


class Pace:
    def __init__(self, path):
        self.path, self.state = Path(path), {}
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            pass

    def observe(self, identity, now, step):
        observations = self.state.get("observations", [])
        previous = observations[-1] if observations else None
        reset = self.state.get("identity") != identity or (previous and
                (now <= previous[0] or now - previous[0] > 90 or step < previous[1]))
        if reset:
            self.state = dict(identity=identity, observations=[], history=[], anchor=[now, step])
        self.state.setdefault("observations", []).append([now, step])
        self.state["observations"] = [row for row in self.state["observations"]
                                      if now - row[0] <= 300]
        self.record_interval(now, step)
        self.persist()
        return self.summary()

    def record_interval(self, now, step):
        anchor = self.state.setdefault("anchor", [now, step])
        span, updates = now - anchor[0], step - anchor[1]
        if span >= 30 and updates > 0:
            point = dict(step=step, seconds=span / updates, span_seconds=span, updates=updates)
            self.state.setdefault("history", []).append(point)
            self.state["history"] = self.state["history"][-120:]
            self.state["anchor"] = [now, step]

    def persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state))
        temporary.replace(self.path)

    def summary(self):
        first, last = self.state["observations"][0], self.state["observations"][-1]
        span, updates = last[0] - first[0], last[1] - first[1]
        seconds = span / updates if span >= 60 and updates > 0 else None
        return dict(seconds_per_update=seconds, observation_seconds=span,
                    rate_sample_count=updates, rate_history=self.state.get("history", []))
