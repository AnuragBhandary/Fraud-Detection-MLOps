"""Online feature store: the history features of :mod:`fraudops.features`, one event at a time.

Serving cannot run a window function over the whole table, so each entity keeps a small running
state (count, sum, sum of squares, last timestamp and the timestamps of the last 24 hours). Scoring
reads the state *before* the transaction is added, which is exactly the "strictly earlier"
definition the offline code uses. ``tests/test_parity.py`` replays a stream through this class and
checks every value against the offline builder.
"""

from __future__ import annotations

import math
import pickle
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fraudops.config import AMOUNT, DAY_SECONDS, TIME
from fraudops.features import HISTORY_KEYS


@dataclass
class EntityState:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    last_t: int | None = None
    recent: deque[int] = field(default_factory=deque)


def _f32(x: float | None) -> float | None:
    """Round through float32 the way the offline Parquet columns are stored."""
    if x is None:
        return None
    import numpy as np

    return float(np.float32(x))


def row_values(txn: Mapping[str, Any]) -> dict[str, Any]:
    """Row features for one transaction (mirrors :func:`fraudops.features.row_features`)."""
    t = int(txn[TIME])
    amt = float(txn[AMOUNT])
    day = t // DAY_SECONDS
    d1 = txn.get("D1")
    account_start = None if d1 is None else float(day - d1)
    addr1, card1 = txn.get("addr1"), txn.get("card1")
    uid = None
    if addr1 is not None and account_start is not None and card1 is not None:
        uid = f"{int(card1)}_{int(addr1)}_{int(account_start)}"
    return {
        "hour": (t // 3600) % 24,
        "dow": day % 7,
        "amt_log": _f32(math.log1p(amt)),
        "amt_cents": _f32(round(amt * 100) % 100),
        "account_start": _f32(account_start),
        "uid": uid,
    }


class OnlineFeatureStore:
    def __init__(self, keys: tuple[str, ...] = HISTORY_KEYS) -> None:
        self.keys = keys
        self.state: dict[str, dict[Any, EntityState]] = {k: {} for k in keys}
        self._lock = threading.Lock()

    def _features_for(self, key: str, value: Any, t: int, amt: float) -> dict[str, float | None]:
        names = [f"{key}_{s}" for s in ("n_prior", "amt_mean_prior", "amt_std_prior")]
        names += [f"{key}_{s}" for s in ("amt_ratio", "secs_since_prev", "n_24h")]
        if value is None:
            return dict.fromkeys(names)
        st = self.state[key].get(value)
        if st is None:
            st = EntityState()
        while st.recent and st.recent[0] <= t - DAY_SECONDS:
            st.recent.popleft()
        mean = st.s / st.n if st.n > 0 else None
        std = None
        if st.n > 1 and mean is not None:
            std = math.sqrt(max(st.ss / st.n - mean * mean, 0.0))
        vals = [
            float(st.n),
            mean,
            std,
            amt / mean if mean is not None else None,
            float(t - st.last_t) if st.last_t is not None else None,
            float(len(st.recent)),
        ]
        return {n: _f32(v) for n, v in zip(names, vals, strict=True)}

    def _update(self, key: str, value: Any, t: int, amt: float) -> None:
        if value is None:
            return
        st = self.state[key].setdefault(value, EntityState())
        st.n += 1
        st.s += amt
        st.ss += amt * amt
        st.last_t = t
        st.recent.append(t)

    def process(self, txn: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        """Features for ``txn`` from the state before it; then (optionally) add it to the state."""
        row = row_values(txn)
        t = int(txn[TIME])
        amt = float(txn[AMOUNT])
        values = {"card1": txn.get("card1"), "uid": row["uid"]}
        out = dict(row)
        with self._lock:
            for k in self.keys:
                out.update(self._features_for(k, values[k], t, amt))
            if update:
                for k in self.keys:
                    self._update(k, values[k], t, amt)
        return out

    def warm(self, rows: list[dict[str, Any]]) -> None:
        """Replay history (already in stream order) without scoring it."""
        for r in rows:
            self.process(r, update=True)

    def entities(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.state.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> OnlineFeatureStore:
        store = cls()
        with path.open("rb") as f:
            store.state = pickle.load(f)
        return store
