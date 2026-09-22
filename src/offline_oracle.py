from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


class ReplicateOracle:

    def __init__(self, raw_map: Dict[str, List[Tuple[float, float]]], max_reps_per_cond: int = 4):
        self.raw_map_base = raw_map
        self.max_reps_per_cond = int(max_reps_per_cond)
        self.streams: Dict[str, List[Tuple[float, float]]] = {}
        self.cap_per_cid: Dict[str, int] = {}

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self.streams = {}
        self.cap_per_cid = {}

        for cid, pairs in self.raw_map_base.items():
            if not pairs:
                self.cap_per_cid[cid] = 0
                self.streams[cid] = []
                continue

            cap = min(len(pairs), self.max_reps_per_cond)  # allow 4 if present and max>=4
            self.cap_per_cid[cid] = cap

            order = rng.permutation(len(pairs))
            shuf = [pairs[i] for i in order]
            self.streams[cid] = shuf[:cap]

    def remaining(self, cid: str) -> int:
        return len(self.streams.get(cid, []))

    def cap(self, cid: str) -> int:
        return int(self.cap_per_cid.get(cid, 0))

    def draw(self, cid: str) -> Optional[Tuple[float, float]]:
        s = self.streams.get(cid, None)
        if not s:
            return None
        y_m, y_d = s.pop(0)
        return float(y_m), float(y_d)


@dataclass
class ObsSummary:
    n_total: int
    n_unique: int
    n_repeats: int
    max_per_cond: int


class ObservationStore:
    def __init__(self):
        self.obs: Dict[str, List[Tuple[float, float]]] = {}

    def clear(self):
        self.obs.clear()

    def add(self, cid: str, y_mono: float, y_di: float):
        self.obs.setdefault(cid, []).append((float(y_mono), float(y_di)))

    def n_total(self) -> int:
        return sum(len(v) for v in self.obs.values())

    def n_unique(self) -> int:
        return len(self.obs)

    def n_repeats(self) -> int:
        # repeats beyond the first measurement per condition
        return sum(max(0, len(v) - 1) for v in self.obs.values())

    def max_per_cond(self) -> int:
        return max((len(v) for v in self.obs.values()), default=0)

    def summary(self) -> ObsSummary:
        n_total = self.n_total()
        n_rep = self.n_repeats()
        return ObsSummary(
            n_total=n_total,
            n_unique=self.n_unique(),
            n_repeats=n_rep,
            max_per_cond=self.max_per_cond(),
        )