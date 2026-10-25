import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_ambulance.db")
    database.init_db()
    yield


def test_insert_and_retrieve():
    did = database.save_dispatch(
        origin="A", destination="B", origin_source="manual",
        standard_distance_m=1000, optimized_distance_m=900,
        standard_time_min=10.0, optimized_time_min=8.0,
        congestion_level="HIGH", predicted_traffic=35.0,
    )
    rows = database.get_all_dispatches()
    assert len(rows) == 1
    assert rows[0]["id"] == did
    assert rows[0]["time_saved_min"] == 2.0


def test_increment_reroute_count():
    did = database.save_dispatch(
        origin="A", destination="B", origin_source="gps",
        standard_distance_m=1000, optimized_distance_m=900,
        standard_time_min=10.0, optimized_time_min=8.0,
        congestion_level="LOW", predicted_traffic=45.0,
    )
    database.increment_reroute_count(did)
    database.increment_reroute_count(did)
    rows = database.get_all_dispatches()
    assert rows[0]["reroute_count"] == 2


def test_summary_stats():
    database.save_dispatch(
        origin="A", destination="B", origin_source="manual",
        standard_distance_m=1000, optimized_distance_m=800,
        standard_time_min=10.0, optimized_time_min=6.0,
        congestion_level="SEVERE", predicted_traffic=20.0,
    )
    database.save_dispatch(
        origin="C", destination="D", origin_source="gps",
        standard_distance_m=2000, optimized_distance_m=2000,
        standard_time_min=15.0, optimized_time_min=15.0,
        congestion_level="LOW", predicted_traffic=50.0,
    )
    stats = database.get_summary_stats()
    assert stats["total_dispatches"] == 2
    assert stats["avg_time_saved_min"] == 2.0  # (4 + 0) / 2
    assert stats["congestion_breakdown"] == {"SEVERE": 1, "LOW": 1}
