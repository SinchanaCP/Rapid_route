"""
rerouting.py

Decides WHETHER the ambulance's current route should be replaced, given an
updated position and updated traffic predictions. This module does NOT
compute routes itself (that's routing.py's job) - it only compares a
candidate route against the current route and applies a threshold-based
decision policy, to avoid route-A/route-B oscillation (Phase 20).

Rerouting triggers implemented:
    1. Significant improvement: candidate ETA is at least
       REROUTE_THRESHOLD_MINUTES faster than the current route's remaining ETA.
    2. Congestion increase: the current route's remaining edges have
       developed SEVERE predicted congestion on a monitored road.
    3. Route obstruction: an edge on the current route no longer exists in
       the graph (e.g. it was removed/blocked) - candidate route is used
       unconditionally in this case.
    4. Route deviation: the ambulance's current position is farther than
       DEVIATION_THRESHOLD_M from the current planned route - triggers a
       recompute from the new actual position.

REROUTE_THRESHOLD_MINUTES and DEVIATION_THRESHOLD_M are configurable and
documented below rather than picked arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from routing import (
    standard_route,
    predictive_route,
    apply_predicted_weights,
    nearest_node,
    NoRouteFoundError,
)

# Reroute only if the new route saves at least this many minutes. Chosen as
# a middle ground: small enough to catch a genuinely faster alternative,
# large enough that routes with near-identical ETAs (a few seconds/minutes
# apart, which is noise-level given the model's measured MAE of ~8.4 km/h on
# Average Speed - see training_metadata.json) don't cause pointless
# rerouting/oscillation. Tune this in a real deployment against measured
# prediction error.
REROUTE_THRESHOLD_MINUTES = 3.0

# If the ambulance is farther than this from every point on the currently
# planned route, treat it as having deviated (e.g. manual override, road
# closure forcing a detour) and force a recompute from its real position.
DEVIATION_THRESHOLD_M = 150.0

# A remaining route edge that is on a monitored road and predicted SEVERE
# always triggers a reroute check, regardless of the ETA-only threshold.
CONGESTION_REROUTE_LEVELS = {"SEVERE"}


@dataclass
class RerouteDecision:
    should_reroute: bool
    reason: str
    new_route: Optional[dict] = None
    current_route_eta_minutes: Optional[float] = None
    candidate_route_eta_minutes: Optional[float] = None
    improvement_minutes: Optional[float] = None


def _remaining_route_congestion_levels(
    G_weighted: nx.MultiDiGraph, remaining_nodes: list[int]
) -> list[str]:
    """Return the 'speed_source'/congestion info for edges still ahead on
    the current route, by re-deriving congestion category from the weighted
    graph's predicted speed if available. Kept simple/explicit rather than
    re-importing predictor internals here, to keep this module's
    responsibility limited to reroute DECISIONS, not predictions."""
    levels = []
    for u, v in zip(remaining_nodes[:-1], remaining_nodes[1:]):
        if G_weighted.has_edge(u, v):
            edge_data = G_weighted.get_edge_data(u, v)
            # MultiDiGraph: could be multiple parallel edges; check all
            for _, data in edge_data.items():
                lvl = data.get("congestion_level")
                if lvl:
                    levels.append(lvl)
    return levels


def evaluate_reroute(
    G,
    road_predictions: dict[str, float],
    current_position_node: int,
    current_route_remaining_nodes: list[int],
    destination_node: int,
    current_route_remaining_eta_minutes: float,
) -> RerouteDecision:
    """
    Core decision function, following the architecture:
        current position -> updated predictions -> updated weights
        -> candidate route -> compare with current -> reroute Y/N

    Raises nothing on "no better route found" - that's a normal NO decision.
    Propagates NoRouteFoundError / GraphUnavailableError from routing.py if
    the graph itself is broken, since that's not a decision this module can
    make silently.
    """
    G_weighted = apply_predicted_weights(G, road_predictions)

    # Trigger 3: route obstruction - if any edge on the remaining planned
    # route no longer exists, we must reroute regardless of threshold.
    obstructed = any(
        not G_weighted.has_edge(u, v)
        for u, v in zip(current_route_remaining_nodes[:-1], current_route_remaining_nodes[1:])
    )

    candidate = predictive_route(G_weighted, current_position_node, destination_node)
    candidate_eta = candidate["eta_minutes"]
    improvement = current_route_remaining_eta_minutes - candidate_eta

    if obstructed:
        return RerouteDecision(
            should_reroute=True,
            reason="Current route is obstructed (a planned road segment is no longer usable).",
            new_route=candidate,
            current_route_eta_minutes=current_route_remaining_eta_minutes,
            candidate_route_eta_minutes=candidate_eta,
            improvement_minutes=round(improvement, 2),
        )

    # Trigger 2: congestion increase on the remaining current route.
    remaining_levels = _remaining_route_congestion_levels(G_weighted, current_route_remaining_nodes)
    if any(lvl in CONGESTION_REROUTE_LEVELS for lvl in remaining_levels):
        return RerouteDecision(
            should_reroute=True,
            reason="Severe predicted congestion detected ahead on the current route.",
            new_route=candidate,
            current_route_eta_minutes=current_route_remaining_eta_minutes,
            candidate_route_eta_minutes=candidate_eta,
            improvement_minutes=round(improvement, 2),
        )

    # Trigger 1: significant improvement.
    if improvement >= REROUTE_THRESHOLD_MINUTES:
        return RerouteDecision(
            should_reroute=True,
            reason=f"Predicted travel time improves by {improvement:.1f} min "
                   f"(>= threshold of {REROUTE_THRESHOLD_MINUTES} min).",
            new_route=candidate,
            current_route_eta_minutes=current_route_remaining_eta_minutes,
            candidate_route_eta_minutes=candidate_eta,
            improvement_minutes=round(improvement, 2),
        )

    return RerouteDecision(
        should_reroute=False,
        reason="No significant improvement detected; continuing on current route.",
        new_route=None,
        current_route_eta_minutes=current_route_remaining_eta_minutes,
        candidate_route_eta_minutes=candidate_eta,
        improvement_minutes=round(improvement, 2),
    )


def has_deviated(G, current_lat: float, current_lng: float, planned_route_nodes: list[int]) -> bool:
    """Trigger 4: has the ambulance moved meaningfully off the planned route?

    Approximated by checking whether the nearest graph node to the
    ambulance's current GPS position is actually one of the planned route's
    nodes (or immediately adjacent to one). This is a simple, explainable
    approximation appropriate for an academic prototype - a production
    system would do a proper point-to-polyline distance calculation.
    """
    nearest = nearest_node(G, current_lat, current_lng)
    if nearest in planned_route_nodes:
        return False
    neighbors = set()
    for n in planned_route_nodes:
        neighbors.update(G.neighbors(n))
    return nearest not in neighbors
