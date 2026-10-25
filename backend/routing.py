"""
routing.py

Builds/caches the Bengaluru road network (OSMnx + NetworkX) and computes:
    1. the STANDARD route (shortest by physical distance)
    2. the PREDICTIVE route (shortest by predicted travel time, where edge
       weight = length / predicted_speed for the roads RapidRoute has traffic
       predictions for, and a reasonable default free-flow speed for edges
       the dataset doesn't cover)

IMPORTANT / HONEST LIMITATION (documented per project Rule 4):
    Downloading the Bengaluru graph requires reaching OpenStreetMap's
    Overpass API (and geocoding requires Nominatim). In the sandbox this
    code was developed and tested in, outbound network access is restricted
    to a fixed allow-list of domains that does NOT include
    overpass-api.de / nominatim.openstreetmap.org (verified: both return
    HTTP 403 from this sandbox). This means the OSM-dependent code paths in
    this file could NOT be executed/tested inside that sandbox. They are
    written against the standard, documented OSMnx/NetworkX API and follow
    the same patterns used throughout OSMnx's own examples, but you should
    run `build_graph()` once on a machine with normal internet access (e.g.
    your own laptop, or a server without the sandbox's domain allow-list) to
    confirm it downloads and caches correctly before relying on it.

This module never fabricates a road graph, a route, or a distance. If the
graph cannot be downloaded/loaded, functions here raise a clear
GraphUnavailableError rather than returning fake data.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    import osmnx as ox
except ImportError:  # pragma: no cover
    ox = None

BASE_DIR = Path(__file__).resolve().parent
GRAPH_CACHE_PATH = BASE_DIR / "data" / "bengaluru_graph.pkl"

# Bengaluru bounding place name used for the OSM query. A named place query
# (rather than a raw bounding box) is used so the graph stays focused on the
# city rather than pulling in a large surrounding region.
BENGALURU_PLACE = "Bengaluru, Karnataka, India"

# Default free-flow speed (km/h) used ONLY for edges that fall outside the
# 16 roads/intersections covered by the RapidRoute traffic dataset. This is
# a conservative city-arterial assumption, not a measured value, and is
# labeled as such everywhere it's used.
DEFAULT_FREE_FLOW_SPEED_KMH = 30.0


class GraphUnavailableError(RuntimeError):
    pass


class NoRouteFoundError(RuntimeError):
    pass


def build_graph(force_refresh: bool = False) -> nx.MultiDiGraph:
    """Download (or load from local cache) the Bengaluru drivable road graph.

    Caches to disk so the graph is not re-downloaded on every request/process
    start, per Phase 10's requirement to cache rather than re-download.
    """
    if ox is None:
        raise GraphUnavailableError(
            "osmnx is not installed. Install it with `pip install osmnx`."
        )

    if GRAPH_CACHE_PATH.exists() and not force_refresh:
        with open(GRAPH_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    try:
        G = ox.graph_from_place(BENGALURU_PLACE, network_type="drive")
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
    except Exception as exc:  # network failure, Overpass timeout, etc.
        raise GraphUnavailableError(
            f"Could not download the Bengaluru road graph from OpenStreetMap: {exc}"
        ) from exc

    GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_CACHE_PATH, "wb") as f:
        pickle.dump(G, f)

    return G


def load_cached_graph() -> Optional[nx.MultiDiGraph]:
    if GRAPH_CACHE_PATH.exists():
        with open(GRAPH_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return None


def nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    if ox is None:
        raise GraphUnavailableError("osmnx is not installed.")
    return ox.distance.nearest_nodes(G, X=lng, Y=lat)


def standard_route(G: nx.MultiDiGraph, origin_node: int, dest_node: int) -> dict:
    """Baseline route: shortest path by physical distance (edge 'length')."""
    try:
        path = nx.shortest_path(G, origin_node, dest_node, weight="length")
    except nx.NetworkXNoPath as exc:
        raise NoRouteFoundError(f"No path exists between the given nodes: {exc}") from exc

    distance_m = nx.path_weight(G, path, weight="length")
    # Fallback ETA using free-flow speed only (no traffic awareness) -
    # this is the "reactive/current-state" baseline RapidRoute is compared against.
    eta_minutes = (distance_m / 1000.0) / DEFAULT_FREE_FLOW_SPEED_KMH * 60.0

    return {
        "path_nodes": path,
        "distance_m": distance_m,
        "eta_minutes": round(eta_minutes, 2),
        "route_type": "standard",
    }


def apply_predicted_weights(G: nx.MultiDiGraph, road_predictions: dict[str, float]) -> nx.MultiDiGraph:
    """
    Returns a COPY of G with a 'predicted_travel_time' edge attribute set.

    road_predictions: {road_name (matches OSM 'name' tag): predicted_avg_speed_kmh}

    For edges whose OSM 'name' matches one of RapidRoute's 16 monitored
    roads, edge weight = length / predicted_speed. All other edges fall back
    to DEFAULT_FREE_FLOW_SPEED_KMH (explicitly NOT a prediction - documented
    via the 'speed_source' edge attribute so this is inspectable, not
    silently mixed with real predictions).
    """
    H = G.copy()
    for u, v, k, data in H.edges(keys=True, data=True):
        name = data.get("name")
        names = name if isinstance(name, list) else [name]
        matched_speed = None
        for n in names:
            if n in road_predictions:
                matched_speed = road_predictions[n]
                break

        speed_kmh = matched_speed if matched_speed is not None else DEFAULT_FREE_FLOW_SPEED_KMH
        length_m = data.get("length", 0.0)
        travel_time_s = (length_m / 1000.0) / max(speed_kmh, 1.0) * 3600.0

        data["predicted_travel_time"] = travel_time_s
        data["speed_source"] = "lstm_prediction" if matched_speed is not None else "default_free_flow"

    return H


def predictive_route(G_weighted: nx.MultiDiGraph, origin_node: int, dest_node: int) -> dict:
    """Optimized route: shortest path by predicted travel time.

    Uses Dijkstra's algorithm via networkx.shortest_path(weight=...), which
    is the standard choice for non-negative edge weights (all travel times
    here are non-negative), and is what NetworkX uses internally for
    weighted shortest_path on non-negative weights.
    """
    try:
        path = nx.shortest_path(G_weighted, origin_node, dest_node, weight="predicted_travel_time")
    except nx.NetworkXNoPath as exc:
        raise NoRouteFoundError(f"No path exists between the given nodes: {exc}") from exc

    distance_m = nx.path_weight(G_weighted, path, weight="length")
    travel_time_s = nx.path_weight(G_weighted, path, weight="predicted_travel_time")

    return {
        "path_nodes": path,
        "distance_m": distance_m,
        "eta_minutes": round(travel_time_s / 60.0, 2),
        "route_type": "predictive",
    }


def route_average_predicted_speed_kmh(
    G_weighted: nx.MultiDiGraph, path_nodes: list[int]
) -> Optional[float]:
    """Return the effective speed across predicted segments of a route.

    Unmonitored edges use an assumed fallback speed and are intentionally
    excluded so they cannot be presented as traffic predictions. Parallel
    edges are handled using the same minimum-travel-time choice as routing.
    """
    predicted_distance_m = 0.0
    predicted_travel_time_s = 0.0

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = G_weighted.get_edge_data(u, v)
        if not edge_data:
            continue

        edge = min(edge_data.values(), key=lambda data: data["predicted_travel_time"])
        if edge.get("speed_source") != "lstm_prediction":
            continue

        length_m = edge.get("length", 0.0)
        travel_time_s = edge.get("predicted_travel_time", 0.0)
        if length_m > 0 and travel_time_s > 0:
            predicted_distance_m += length_m
            predicted_travel_time_s += travel_time_s

    if predicted_distance_m == 0 or predicted_travel_time_s == 0:
        return None

    return (predicted_distance_m / 1000.0) / (predicted_travel_time_s / 3600.0)


def route_to_coords(G: nx.MultiDiGraph, path_nodes: list[int]) -> list[tuple[float, float]]:
    """Convert a list of node ids into [(lat, lng), ...] for the frontend map."""
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path_nodes]
