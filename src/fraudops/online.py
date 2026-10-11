"""Online feature store: the history features of :mod:`fraudops.features`, one event at a time.

Serving cannot run a window function over the whole table, so each entity keeps a small running
state (count, sum, sum of squares, last timestamp and the timestamps of the last 24 hours). Scoring
reads the state *before* the transaction is added, which is exactly the "strictly earlier"
definition the offline code uses. ``tests/test_parity.py`` replays a stream through this class and
checks every value against the offline builder.

Known-fraud features need labels, which in production arrive as chargebacks weeks later. Each
entity keeps its transactions of the last ``LABEL_DELAY_DAYS`` in a pending queue; when one gets
older than the delay it *matures*: fraud if a chargeback was reported for it
(:meth:`OnlineFeatureStore.report_chargeback`), legitimate otherwise.
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

from fraudops.config import AMOUNT, DAY_SECONDS, ID, LABEL_DELAY_DAYS, TARGET, TIME
from fraudops.features import HISTORY_KEYS


@dataclass
class EntityState:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    last_t: int | None = None
    recent: deque[int] = field(default_factory=deque)
    pending: deque[tuple[int, int]] = field(default_factory=deque)  # (time, transaction id)
    matured: int = 0
    fraud_known: int = 0


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
    def __init__(
        self, keys: tuple[str, ...] = HISTORY_KEYS, label_delay_days: int = LABEL_DELAY_DAYS
    ) -> None:
        self.keys = keys
        self.delay = label_delay_days * DAY_SECONDS
        self.state: dict[str, dict[Any, EntityState]] = {k: {} for k in keys}
        self.chargebacks: set[int] = set()  # reported fraud whose label has not matured everywhere
        self._lock = threading.Lock()

    def report_chargeback(self, transaction_id: int) -> None:
        with self._lock:
            self.chargebacks.add(int(transaction_id))

    def _mature(self, st: EntityState, t: int) -> None:
        while st.pending and st.pending[0][0] <= t - self.delay:
            _, txn_id = st.pending.popleft()
            st.matured += 1
            st.fraud_known += txn_id in self.chargebacks

    def _features_for(self, key: str, value: Any, t: int, amt: float) -> dict[str, float | None]:
        names = [f"{key}_{s}" for s in ("n_prior", "amt_mean_prior", "amt_std_prior")]
        names += [f"{key}_{s}" for s in ("amt_ratio", "secs_since_prev", "n_24h")]
        names += [f"{key}_{s}" for s in ("labels_matured", "fraud_known", "fraud_rate_known")]
        if value is None:
            return dict.fromkeys(names)
        st = self.state[key].get(value)
        if st is None:
            st = EntityState()
        while st.recent and st.recent[0] <= t - DAY_SECONDS:
            st.recent.popleft()
        self._mature(st, t)
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
            float(st.matured),
            float(st.fraud_known),
            st.fraud_known / st.matured if st.matured > 0 else None,
        ]
        return {n: _f32(v) for n, v in zip(names, vals, strict=True)}

    def _update(self, key: str, value: Any, t: int, amt: float, txn_id: int) -> None:
        if value is None:
            return
        st = self.state[key].setdefault(value, EntityState())
        st.n += 1
        st.s += amt
        st.ss += amt * amt
        st.last_t = t
        st.recent.append(t)
        st.pending.append((t, txn_id))

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
                    self._update(k, values[k], t, amt, int(txn[ID]))
        return out

    def warm(self, rows: list[dict[str, Any]]) -> None:
        """Replay labelled history (in stream order); fraud rows are reported as chargebacks."""
        for r in rows:
            self.process(r, update=True)
            if r.get(TARGET):
                self.report_chargeback(r[ID])

    def entities(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.state.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            snapshot = {"delay": self.delay, "state": self.state, "chargebacks": self.chargebacks}
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> OnlineFeatureStore:
        store = cls()
        with path.open("rb") as f:
            snapshot = pickle.load(f)
        store.delay = snapshot["delay"]
        store.state = snapshot["state"]
        store.chargebacks = snapshot["chargebacks"]
        return store
