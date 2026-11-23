"""Per-run metric accumulation."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Metrics:
    correct: list = field(default_factory=list)
    latency: list = field(default_factory=list)
    probes: list = field(default_factory=list)
    audits: list = field(default_factory=list)
    exposure: list = field(default_factory=list)

    def record(self, correct, latency, probes, audits, exposure):
        self.correct.append(bool(correct))
        self.latency.append(float(latency))
        self.probes.append(int(probes))
        self.audits.append(int(audits))
        self.exposure.append(float(exposure))

    def summary(self, n_bins: int = 10) -> dict:
        c = np.array(self.correct, dtype=float)
        lat = np.array(self.latency) * 1e3
        half = len(c) // 2
        return {
            "accuracy": float(c.mean()),
            "accuracy_steady": float(c[half:].mean()),
            "latency_mean_ms": float(lat.mean()),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "probes_per_query": float(np.mean(self.probes)),
            "audits_per_query": float(np.mean(self.audits)),
            "adversary_exposure": float(np.mean(self.exposure)),
            "acc_curve": [float(b.mean()) for b in np.array_split(c, n_bins)],
            "n_queries": int(len(c)),
        }
