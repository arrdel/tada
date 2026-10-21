"""Per-run metric accumulation."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Metrics:
    correct: list = field(default_factory=list)
    latency: list = field(default_factory=list)
    probes: list = field(default_factory=list)

    def record(self, correct: bool, latency: float, probes: int):
        self.correct.append(bool(correct))
        self.latency.append(float(latency))
        self.probes.append(int(probes))

    def summary(self) -> dict:
        c = np.array(self.correct, dtype=float)
        lat = np.array(self.latency, dtype=float)
        pr = np.array(self.probes, dtype=float)
        return {
            "accuracy": float(c.mean()) if len(c) else 0.0,
            "latency_mean_ms": float(lat.mean() * 1e3) if len(lat) else 0.0,
            "latency_p95_ms": float(np.percentile(lat, 95) * 1e3) if len(lat) else 0.0,
            "probes_per_query": float(pr.mean()) if len(pr) else 0.0,
            "n_queries": int(len(c)),
        }
