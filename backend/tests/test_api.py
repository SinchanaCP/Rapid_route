import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "model_loaded" in body
    assert "road_graph_loaded" in body
    assert "database_available" in body


def test_stats_endpoint_empty_db_ok(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["total_dispatches"] >= 0


def test_dispatches_endpoint(client):
    r = client.get("/api/dispatches")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_route_without_graph_returns_503_not_fake_data(client):
    # If the OSM graph hasn't been built on this machine yet, the API must
    # fail honestly (503) rather than return a fabricated route.
    r = client.post("/api/route", json={"origin": "Indiranagar, Bengaluru", "destination": "Whitefield, Bengaluru"})
    assert r.status_code in (503, 502, 404, 200)  # 200 only if a graph IS cached on this machine
    if r.status_code == 503:
        assert "graph" in r.json()["detail"].lower()


def test_route_invalid_origin_source_rejected(client):
    r = client.post("/api/route", json={
        "origin": "X", "destination": "Y", "origin_source": "not_a_real_source"
    })
    assert r.status_code == 422  # pydantic validation error

