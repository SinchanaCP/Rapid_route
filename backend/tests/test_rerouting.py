"""
Tests for rerouting.py's decision logic.

These tests build a tiny SYNTHETIC road graph by hand (a diamond: A -> B -> D
and A -> C -> D) instead of downloading the real Bengaluru OSM graph, because
this sandbox cannot reach OpenStreetMap's Overpass/Nominatim services (see
routing.py's module docstring). This still fully exercises rerouting.py's
actual decision logic, since that logic only depends on the networkx graph
API, not on OSMnx specifically.
"""
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import routing
import rerouting
from predict import TrafficPredictor


def make_diamond_graph():
    """
    A (0) --- B (1) --- D (3)   [short but will become "congested" road]
      \\                    /
       -- C (2) -----------    [longer road, always free-flow]
    """
    G = nx.MultiDiGraph()
    G.add_node(0, y=12.90, x=77.50)
    G.add_node(1, y=12.91, x=77.51)
    G.add_node(2, y=12.89, x=77.52)
    G.add_node(3, y=12.92, x=77.53)

    # A -> B -> D: short route on a "monitored" road
    G.add_edge(0, 1, key=0, length=1000, name="Monitored Road")
    G.add_edge(1, 3, key=0, length=1000, name="Monitored Road")

    # A -> C -> D: longer route on an unmonitored road (always free-flow default)
    G.add_edge(0, 2, key=0, length=1800, name="Other Road")
    G.add_edge(2, 3, key=0, length=1800, name="Other Road")

    return G


def test_standard_route_picks_shortest_by_distance():
    G = make_diamond_graph()
    result = routing.standard_route(G, 0, 3)
    assert result["path_nodes"] == [0, 1, 3]
    assert result["distance_m"] == 2000


def test_predictive_route_prefers_faster_predicted_road():
    G = make_diamond_graph()
    # Monitored Road predicted VERY slow (should push route onto Other Road)
    road_predictions = {"Monitored Road": 5.0}  # km/h, very congested
    G_w = routing.apply_predicted_weights(G, road_predictions)
    result = routing.predictive_route(G_w, 0, 3)
    assert result["path_nodes"] == [0, 2, 3]


def test_predictive_route_keeps_fast_monitored_road():
    G = make_diamond_graph()
    road_predictions = {"Monitored Road": 60.0}  # fast, should stay on shorter road
    G_w = routing.apply_predicted_weights(G, road_predictions)
    result = routing.predictive_route(G_w, 0, 3)
    assert result["path_nodes"] == [0, 1, 3]


def test_route_average_speed_uses_only_predicted_route_segments():
    G = make_diamond_graph()
    G_w = routing.apply_predicted_weights(G, {"Monitored Road": 40.0})

    assert routing.route_average_predicted_speed_kmh(G_w, [0, 1, 3]) == pytest.approx(40.0)
    assert routing.route_average_predicted_speed_kmh(G_w, [0, 2, 3]) is None


def test_route_congestion_classifies_each_predicted_speed_level():
    predictor = TrafficPredictor()
    predictor._speed_quartiles = (30.0, 40.0, 50.0)
    expected_levels = {
        25.0: "SEVERE",
        35.0: "HIGH",
        45.0: "MODERATE",
        55.0: "LOW",
    }

    for speed, expected_level in expected_levels.items():
        G_w = routing.apply_predicted_weights(make_diamond_graph(), {"Monitored Road": speed})
        route_speed = routing.route_average_predicted_speed_kmh(G_w, [0, 1, 3])
        assert route_speed == pytest.approx(speed)
        assert predictor.congestion_for_speed(route_speed) == expected_level


def test_no_route_found_raises():
    G = nx.MultiDiGraph()
    G.add_node(0, y=0, x=0)
    G.add_node(1, y=0, x=0)
    with pytest.raises(routing.NoRouteFoundError):
        routing.standard_route(G, 0, 1)


def test_reroute_triggered_by_significant_improvement():
    G = make_diamond_graph()
    # Current route is A->B->D (on now-congested Monitored Road), but we're
    # told its ETA is very high (as if measured earlier before congestion appeared).
    road_predictions = {"Monitored Road": 5.0}
    decision = rerouting.evaluate_reroute(
        G=G,
        road_predictions=road_predictions,
        current_position_node=0,
        current_route_remaining_nodes=[0, 1, 3],
        destination_node=3,
        current_route_remaining_eta_minutes=30.0,  # artificially high vs candidate
    )
    assert decision.should_reroute is True
    assert decision.new_route["path_nodes"] == [0, 2, 3]


def test_no_reroute_when_improvement_below_threshold():
    G = make_diamond_graph()
    road_predictions = {"Monitored Road": 60.0}
    # Compute the actual candidate ETA first, then say current ETA is only
    # marginally worse (below REROUTE_THRESHOLD_MINUTES).
    G_w = routing.apply_predicted_weights(G, road_predictions)
    candidate = routing.predictive_route(G_w, 0, 3)
    tiny_worse_eta = candidate["eta_minutes"] + 0.1

    decision = rerouting.evaluate_reroute(
        G=G,
        road_predictions=road_predictions,
        current_position_node=0,
        current_route_remaining_nodes=[0, 1, 3],
        destination_node=3,
        current_route_remaining_eta_minutes=tiny_worse_eta,
    )
    assert decision.should_reroute is False


def test_reroute_triggered_by_route_obstruction():
    G = make_diamond_graph()
    G.remove_edge(1, 3, key=0)  # simulate the road becoming unusable

    decision = rerouting.evaluate_reroute(
        G=G,
        road_predictions={},
        current_position_node=0,
        current_route_remaining_nodes=[0, 1, 3],
        destination_node=3,
        current_route_remaining_eta_minutes=5.0,
    )
    assert decision.should_reroute is True
    assert "obstructed" in decision.reason.lower()
