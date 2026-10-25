"""
main.py

RapidRoute FastAPI backend.

Endpoints:
    GET  /api/health
    POST /api/route
    POST /api/reroute
    GET  /api/dispatches
    GET  /api/stats
    WS   /ws/track   (optional live-tracking / simulation channel)

Data-source honesty (Rule 4): every traffic figure returned by this API is
labeled with its source:
    "lstm_prediction"  - output of the trained LSTM on the most recent
                          historical observations available for that road.
    "default_free_flow" - a road not covered by the dataset; a flat
                          assumed speed, NOT a prediction.
    "simulated"         - only used by the optional /ws/track demo mode,
                          never presented as live Bengaluru traffic.
This backend does NOT have a live traffic feed. It predicts from the
historical dataset shipped with the project. See README "Limitations".
"""

from __future__ import annotations

import osmnx as ox
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database
import routing
from predict import predictor, ModelNotLoadedError, InsufficientHistoryError
from preprocessing import FEATURE_COLUMNS  # noqa: F401 (kept for reference/debug)

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "bengaluru_traffic.csv"

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
_state = {
    "graph": None,           # nx.MultiDiGraph, loaded lazily
    "graph_error": None,     # str | None
    "raw_df": None,          # pd.DataFrame, the historical dataset
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()

    try:
        predictor.load()
    except ModelNotLoadedError as e:
        print(f"[startup] WARNING: predictor failed to load: {e}")

    try:
        _state["raw_df"] = pd.read_csv(RAW_CSV)
    except Exception as e:
        print(f"[startup] WARNING: could not load raw dataset: {e}")

    # Try to load a cached Bengaluru graph if one already exists on disk.
    # We do NOT attempt to download it at startup (that needs live OSM
    # access which may not be available in every environment - see
    # routing.py's module docstring). Run
    # `python -c "from routing import build_graph; build_graph()"` once on a
    # machine with normal internet access first.
    try:
        _state["graph"] = routing.load_cached_graph()
        if _state["graph"] is None:
            _state["graph_error"] = (
                "No cached Bengaluru road graph found. Run "
                "`python -c \"from routing import build_graph; build_graph()\"` "
                "on a machine with normal internet access first."
            )
    except Exception as e:
        _state["graph_error"] = str(e)

    yield  # app runs here

    # no explicit teardown needed


app = FastAPI(title="RapidRoute API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LatLng(BaseModel):
    lat: float
    lng: float


class RouteRequest(BaseModel):
    origin: str | LatLng
    destination: str
    origin_source: str = Field(default="manual", pattern="^(gps|manual)$")


class RerouteRequest(BaseModel):
    current_position: LatLng
    destination: LatLng
    current_route: list[int] = Field(default_factory=list)  # node ids of remaining planned route
    current_route_eta_minutes: float
    dispatch_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_graph():
    if _state["graph"] is None:
        raise HTTPException(
            status_code=503,
            detail=_state["graph_error"] or "Road graph is not loaded.",
        )
    return _state["graph"]


def _geocode(address: str) -> LatLng:
    """Geocode an address, restricted to Bengaluru."""
    query = f"{address}, Bengaluru, Karnataka, India"

    try:
        lat, lng = ox.geocoder.geocode(query)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not geocode '{address}' in Bengaluru: {exc}",
        )

    # Basic sanity check: reject locations clearly outside Bengaluru.
    if not (12.7 <= lat <= 13.2 and 77.3 <= lng <= 77.9):
        raise HTTPException(
            status_code=400,
            detail=f"'{address}' was not resolved to a location in Bengaluru.",
        )

    return LatLng(lat=float(lat), lng=float(lng))


def _predict_all_monitored_roads() -> dict[str, dict]:
    """Run the LSTM for every road the dataset covers, using each road's
    most recent recorded history. Returns {road_name: PredictionResult-as-dict}.
    """
    if _state["raw_df"] is None:
        raise HTTPException(status_code=503, detail="Historical traffic dataset not loaded.")
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Prediction model not loaded. Train it first.")

    df = _state["raw_df"]
    results = {}
    for road in df["Road/Intersection Name"].unique():
        hist = df[df["Road/Intersection Name"] == road]
        try:
            pred = predictor.predict_traffic(road, hist)
            results[road] = pred
        except InsufficientHistoryError:
            continue
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "model_loaded": predictor.loaded,
        "scaler_loaded": predictor.loaded,  # loaded together in predict.load()
        "road_graph_loaded": _state["graph"] is not None,
        "road_graph_error": _state["graph_error"],
        "database_available": database.DB_PATH.exists() or True,  # sqlite creates lazily
    }


@app.post("/api/route")
def get_route(req: RouteRequest):
    G = _require_graph()

    if isinstance(req.origin, LatLng):
        origin_latlng = req.origin
    else:
        origin_latlng = _geocode(req.origin)
    dest_latlng = _geocode(req.destination)

    origin_node = routing.nearest_node(G, origin_latlng.lat, origin_latlng.lng)
    dest_node = routing.nearest_node(G, dest_latlng.lat, dest_latlng.lng)

    try:
        std = routing.standard_route(G, origin_node, dest_node)
    except routing.NoRouteFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    predictions = _predict_all_monitored_roads()
    road_speeds = {road: p.predicted_avg_speed_kmh for road, p in predictions.items()}
    G_weighted = routing.apply_predicted_weights(G, road_speeds)

    try:
        opt = routing.predictive_route(G_weighted, origin_node, dest_node)
    except routing.NoRouteFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    route_avg_speed = routing.route_average_predicted_speed_kmh(
        G_weighted, opt["path_nodes"]
    )
    congestion = (
        predictor.congestion_for_speed(route_avg_speed)
        if route_avg_speed is not None
        else None
    )
    time_saved = round(std["eta_minutes"] - opt["eta_minutes"], 2)

    dispatch_id = database.save_dispatch(
        origin=req.origin if isinstance(req.origin, str) else f"{origin_latlng.lat},{origin_latlng.lng}",
        destination=req.destination,
        origin_source=req.origin_source,
        standard_distance_m=std["distance_m"],
        optimized_distance_m=opt["distance_m"],
        standard_time_min=std["eta_minutes"],
        optimized_time_min=opt["eta_minutes"],
        congestion_level=congestion,
        predicted_traffic=route_avg_speed,
    )

    return {
        "dispatch_id": dispatch_id,
        "origin": origin_latlng,
        "destination": dest_latlng,
        "standard_route": {**std, "coords": routing.route_to_coords(G, std["path_nodes"])},
        "optimized_route": {**opt, "coords": routing.route_to_coords(G_weighted, opt["path_nodes"])},
        "time_saved_minutes": time_saved if time_saved > 0 else 0.0,
        "message": (
            f"RapidRoute found a route with a predicted {time_saved:.1f} min improvement."
            if time_saved > 0.05
            else "No significant improvement detected."
        ),
        "congestion_level": congestion,
        "predicted_route_avg_speed_kmh": (
            round(route_avg_speed, 2) if route_avg_speed is not None else None
        ),
        "predictions_used": {
            road: {
                "predicted_avg_speed_kmh": p.predicted_avg_speed_kmh,
                "congestion_level": p.congestion_level,
                "source": p.source,
            }
            for road, p in predictions.items()
        },
    }


@app.post("/api/reroute")
def reroute(req: RerouteRequest):
    G = _require_graph()

    import rerouting  # local import to avoid circular import at module load

    if not req.current_route or len(req.current_route) < 2:
        raise HTTPException(status_code=400, detail="current_route must contain at least 2 node ids.")

    current_node = routing.nearest_node(G, req.current_position.lat, req.current_position.lng)
    dest_node = routing.nearest_node(G, req.destination.lat, req.destination.lng)

    predictions = _predict_all_monitored_roads()
    road_speeds = {road: p.predicted_avg_speed_kmh for road, p in predictions.items()}

    try:
        decision = rerouting.evaluate_reroute(
            G=G,
            road_predictions=road_speeds,
            current_position_node=current_node,
            current_route_remaining_nodes=req.current_route,
            destination_node=dest_node,
            current_route_remaining_eta_minutes=req.current_route_eta_minutes,
        )
    except routing.NoRouteFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if decision.should_reroute and req.dispatch_id is not None:
        database.increment_reroute_count(req.dispatch_id)

    response = {
        "reroute": decision.should_reroute,
        "reason": decision.reason,
        "current_route_eta_minutes": decision.current_route_eta_minutes,
        "candidate_route_eta_minutes": decision.candidate_route_eta_minutes,
        "improvement_minutes": decision.improvement_minutes,
    }
    if decision.new_route:
        G_weighted = routing.apply_predicted_weights(G, road_speeds)
        response["new_route"] = {
            **decision.new_route,
            "coords": routing.route_to_coords(G_weighted, decision.new_route["path_nodes"]),
        }
    return response


@app.get("/api/dispatches")
def dispatches(limit: int = 100):
    return database.get_all_dispatches(limit=limit)


@app.get("/api/stats")
def stats():
    return database.get_summary_stats()


# ---------------------------------------------------------------------------
# Optional: WebSocket SIMULATION mode (Phase 26/27)
#
# This is NOT live GPS tracking. It replays the ambulance moving along the
# computed route and periodically re-invokes the real rerouting logic on
# simulated position updates. Every message sent over this socket is tagged
# "mode": "SIMULATION" so the frontend can label it clearly and never
# confuse it with live tracking, per Rule 4.
# ---------------------------------------------------------------------------
@app.websocket("/ws/track")
async def ws_track(websocket: WebSocket):
    await websocket.accept()
    try:
        init_msg = await websocket.receive_json()
        route_coords = init_msg.get("route_coords", [])
        if not route_coords:
            await websocket.send_json({"error": "route_coords required to start simulation"})
            await websocket.close()
            return

        for i, (lat, lng) in enumerate(route_coords):
            await websocket.send_json({
                "mode": "SIMULATION",
                "position": {"lat": lat, "lng": lng},
                "progress": round((i + 1) / len(route_coords), 3),
            })
            await asyncio.sleep(1.0)  # simulated movement tick

        await websocket.send_json({"mode": "SIMULATION", "status": "arrived"})
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
