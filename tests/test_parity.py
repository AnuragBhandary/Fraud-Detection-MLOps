"""The online store must reproduce the offline features exactly, row by row."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fraudops.config import ID, TIME
from fraudops.features import history_columns
from fraudops.online import OnlineFeatureStore

COLUMNS = [*history_columns(), "hour", "dow", "amt_log", "amt_cents", "account_start", "uid"]


def test_replay_matches_offline(features: pl.DataFrame) -> None:
    store = OnlineFeatureStore()
    mismatches = []
    for row in features.sort([TIME, ID]).iter_rows(named=True):
        online = store.process(row)
        for c in COLUMNS:
            if row[c] != online[c]:
                mismatches.append((row[ID], c, row[c], online[c]))
    assert mismatches == []


def test_snapshot_round_trip(features: pl.DataFrame, tmp_path: Path) -> None:
    rows = features.sort([TIME, ID]).iter_rows(named=True)
    history = [next(rows) for _ in range(2000)]
    store = OnlineFeatureStore()
    store.warm(history)
    store.save(tmp_path / "s.pkl")
    loaded = OnlineFeatureStore.load(tmp_path / "s.pkl")
    assert loaded.entities() == store.entities()
    nxt = next(rows)
    assert loaded.process(nxt, update=False) == store.process(nxt, update=False)


def test_scoring_without_update_leaves_state_alone(features: pl.DataFrame) -> None:
    row = features.sort([TIME, ID]).row(0, named=True)
    store = OnlineFeatureStore()
    store.process(row, update=False)
    assert store.entities() == {"card1": 0, "uid": 0}
