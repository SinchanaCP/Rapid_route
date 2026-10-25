# RapidRoute — AI-Based Predictive & Dynamic Ambulance Routing for Bengaluru

**Academic project (final-year CSE major project). This is a prototype demonstrating an
approach, not a production emergency-dispatch system.**

---

## 1. Problem

Most navigation systems are reactive: they answer *"what's the fastest route right now?"*
using current or very-recent traffic conditions. RapidRoute explores a more proactive
question: *"what traffic conditions are likely soon, and which route is likely to have
the lowest travel time given that?"*

It combines three things that are normally separate:
1. **Time-series prediction** (LSTM) of near-future road speed, from Bengaluru traffic history.
2. **Graph-based routing** (OSMnx + NetworkX) that turns those predictions into travel-time
   edge weights on the real Bengaluru road network.
3. **Dynamic rerouting** that periodically re-checks the route against updated predictions
   as the ambulance moves, and only switches routes when the improvement clears a threshold.

## 2. Solution / Architecture

```
Bengaluru traffic history (CSV)
            │
            ▼
   Preprocessing + feature engineering
            │
            ▼
        LSTM model  ───────────────►  Predicted Average Speed (km/h), per road
            │
            ▼
  Congestion category (LOW/MODERATE/HIGH/SEVERE)
  derived from the dataset's own speed quartiles
            │
            ▼
  Edge weight = road length / predicted speed
            │
            ▼
     NetworkX (Dijkstra via shortest_path)
            │
            ▼
        Predictive route            Standard route
     (traffic-aware)               (distance-only, baseline)
            │                             │
            └───────────────┬─────────────┘
                             ▼
                     FastAPI /api/route
                             │
                             ▼
                    React + Leaflet map
                             │
                    Ambulance starts moving
                             │
                             ▼
              Periodic position update (GPS or simulation)
                             │
                             ▼
                 FastAPI /api/reroute → rerouting.py
        (re-predicts, re-weighs graph, compares candidate vs
         current route, applies threshold-based reroute decision)
                             │
                    reroute? ─┴─ no  → continue
                       │yes
                       ▼
                Updated route pushed to the map
```

The critical separation (also the interview-ready explanation of this project):

> The LSTM predicts future traffic. It does not calculate a route. The prediction is
> converted into an edge travel-time cost on the Bengaluru road graph. NetworkX finds
> the shortest path using those costs. As the ambulance moves, its position and updated
> predictions are used to recompute a candidate route; the system only switches routes
> when that candidate offers a meaningful improvement.

## 3. Dataset

Source: `backend/data/bengaluru_traffic.csv` (provided with this project).

| Property | Value |
|---|---|
| Rows | 8,936 |
| Columns | 16 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Date range | 2022-01-01 → 2024-08-09 |
| Granularity | **One row per (Date, Road/Intersection)** — a daily, per-road dataset |
| Sampling | **Irregular** per road: gaps of 1–10 days between consecutive observations for the same road (no fixed frequency) |
| Roads/intersections | 16, across 8 areas (Indiranagar, Whitefield, Koramangala, Electronic City, Hebbal, Jayanagar, M.G. Road, Yeshwanthpur) |

**Target column:** `Average Speed` (km/h), predicted one step ahead per road.

**Why not `Congestion Level` / `Travel Time Index` / `Road Capacity Utilization`?**
Direct inspection showed these are heavily saturated at their maximum value — 43.5%,
58.5%, and 73.7% of all rows respectively sit exactly at the column's max. That makes
them poor continuous regression targets (most of the "signal" is just the ceiling).
`Average Speed` is continuous across its full range (20.0–89.8 km/h, 0% of rows at the
max) and converts directly into a travel-time edge weight for routing
(`time = distance / speed`).

**Feature columns used:** Traffic Volume, Average Speed, Congestion Level, Road Capacity
Utilization, Incident Reports, Public Transport Usage, Traffic Signal Compliance, Parking
Usage, Pedestrian and Cyclist Count, `is_weekend`, `Roadwork_flag`, and one-hot encoded
Weather Conditions. See `backend/preprocessing.py` for the exact, documented feature
engineering (including an explicit note on which "obvious" temporal features — like
hour-of-day — were deliberately **not** invented, because this dataset has no
intra-day timestamp).

**Sequence length:** 10 previous *recorded* observations per road (not 10 fixed time
units, since sampling is irregular — see `training_metadata.json` for the full reasoning).

**Split:** chronological, 70/15/15, by the target row's date, globally across all roads.
No shuffling. No temporal leakage.

## 4. Machine Learning

Architecture (`backend/model/train_model.py`):

```
Input (10 timesteps × 17 features)
        │
     LSTM(64, return_sequences=True)
        │
     Dropout(0.2)
        │
     LSTM(32)
        │
     Dropout(0.2)
        │
     Dense(16, relu)
        │
     Dense(1)   →  predicted Average Speed (km/h)
```

Adam optimizer, MSE loss, EarlyStopping + ModelCheckpoint on validation loss.

### Actual measured results (test set, chronologically held out)

| Metric | LSTM | Naive persistence baseline (predict = last observed speed) |
|---|---|---|
| MAE | **8.39 km/h** | 11.91 km/h |
| RMSE | **10.48 km/h** | 14.90 km/h |
| R² | **0.02** | — |

**Honest interpretation:** the LSTM beats a naive "assume tomorrow = today" baseline,
so it has learned *something*. But an R² of ~0.02 means it explains almost none of the
variance in Average Speed — this dataset does not appear to contain a strong, learnable
temporal traffic pattern (each row looks close to independently generated rather than
part of a real, evolving traffic process). **This is reported as-is, not adjusted or
hidden** — see `backend/model/training_metadata.json` for the exact numbers and
`results/actual_vs_predicted.png` / `results/training_val_loss.png` for the plots. Any
claim about "time saved" downstream should be read with this limitation in mind — it
reflects what the *routing algorithm* would do if the predictions were accurate, not
proof that the predictions are highly accurate.

## 5. Routing

- `backend/routing.py` downloads (once) and caches the Bengaluru **drivable** road graph
  via OSMnx (`ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive")`),
  storing it at `backend/data/bengaluru_graph.pkl` so it is never re-downloaded per request.
- **Standard route:** `networkx.shortest_path(..., weight="length")` — physical-distance
  baseline, no traffic awareness.
- **Predictive route:** edge weight = `length / predicted_speed` for the ~16 monitored
  roads (matched by OSM `name` tag), and a documented default free-flow speed
  (30 km/h) for all other edges the dataset doesn't cover. `networkx.shortest_path`
  (Dijkstra, appropriate here since all edge weights are non-negative) finds the route
  minimizing predicted travel time.
- Every edge is tagged `speed_source: "lstm_prediction"` or `"default_free_flow"` so it's
  always inspectable which edges are actually traffic-aware.

### ⚠️ Important limitation you should know before running this

Downloading the OSM graph requires reaching **Overpass API** and geocoding requires
**Nominatim**. This project was built and tested inside a sandboxed environment whose
outbound network access is restricted to a package-registry allow-list — `overpass-api.de`
and `nominatim.openstreetmap.org` both returned **HTTP 403** from that sandbox. That means:

- `backend/routing.py`'s graph download / geocoding code is written against the standard,
  documented OSMnx API but **was not able to be executed there**.
- The **rerouting decision logic** (`backend/rerouting.py`) and the **routing algorithm
  logic** (shortest-path selection, weight application) **were fully tested** against a
  small hand-built synthetic road graph (`backend/tests/test_rerouting.py`) — that part of
  the system is verified to work correctly, just not against the real Bengaluru graph yet.

**Before running this for real:** on a machine with normal internet access, run:
```bash
python -c "from routing import build_graph; build_graph()"
```
from `backend/`, once. This downloads and caches the graph. After that, `main.py`'s
startup will find and load it automatically.

## 6. Dynamic Rerouting

`backend/rerouting.py` implements four triggers, all threshold-based to avoid
route-A/route-B oscillation:

1. **Significant improvement** — candidate route saves ≥ `REROUTE_THRESHOLD_MINUTES`
   (default **3.0 min**, chosen conservatively relative to the model's measured ~8.4 km/h
   MAE so routes aren't swapped over prediction noise).
2. **Congestion increase** — a `SEVERE`-predicted road appears on the remaining current route.
3. **Route obstruction** — an edge on the current route no longer exists in the graph.
4. **Route deviation** — the ambulance's nearest graph node is no longer on (or adjacent
   to) the planned route (`DEVIATION_THRESHOLD_M`, default 150 m).

Routing (*calculates* routes) and rerouting (*decides whether to replace* the current
route) are deliberately kept in separate modules.

## 7. Backend (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Reports model/scaler/graph/database load status |
| `POST /api/route` | Geocodes origin/destination, predicts traffic, returns standard + predictive routes, saves a dispatch |
| `POST /api/reroute` | Given current position + current route, returns a reroute decision (see rerouting.py) |
| `GET /api/dispatches` | Recent dispatch history from SQLite |
| `GET /api/stats` | Aggregate statistics from SQLite |
| `WS /ws/track` | Optional **SIMULATION MODE** channel — replays the ambulance along the computed route and periodically invokes the real rerouting logic. Every message is explicitly tagged `"mode": "SIMULATION"`; this is never presented as live GPS tracking. |

Every prediction returned by the API is labeled with its `source`
(`lstm_prediction` / `default_free_flow` / `simulated`), per the project's no-fake-data rule.

## 8. Database

SQLite (`ambulance.db`, created automatically). Table `dispatches` stores origin,
destination, origin source (gps/manual), standard vs optimized distance/time, time saved,
predicted congestion, and reroute count. All queries in `backend/database.py` are
parameterized.

## 9. Frontend

React + Vite + Tailwind + React-Leaflet + Recharts (`frontend/`). Header, origin/destination
inputs with "use current location", a Leaflet map showing both routes and an ambulance
marker, a route-comparison panel, a rerouting notification banner, a per-road predicted-speed
chart, and a statistics panel reading from `/api/stats`.

This frontend was `npm run build`-tested successfully (compiles cleanly against the API
contracts above). It has **not** been tested against a live backend + real Bengaluru graph
in this environment, for the same network-access reason described in Section 5.

## 10. Installation

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 11. Running

```bash
# 1. Train the model (only needs to be done once, or after changing the dataset/features)
cd backend/model
python train_model.py

# 2. Build the Bengaluru road graph (only needs to be done once; requires normal internet access)
cd ../
python -c "from routing import build_graph; build_graph()"

# 3. Start the backend
uvicorn main:app --reload --port 8000

# 4. In a separate terminal, start the frontend
cd ../frontend
npm run dev
```

Then open the frontend (default `http://localhost:5173`).

## 12. Testing

```bash
cd backend
pytest tests/ -v
```

23 tests covering preprocessing/sequence-building/chronological-splitting, the prediction
module (including error handling for missing model / insufficient history), the SQLite
layer, the rerouting decision logic (all four triggers, via a synthetic graph), and the
FastAPI endpoints (including verifying `/api/route` fails honestly with 503 rather than
returning fake data when the road graph isn't loaded). All 23 currently pass.

## 13. Limitations (stated honestly)

- **Model predictive power is weak (R² ≈ 0.02).** The LSTM outperforms a naive baseline
  but does not strongly predict Average Speed from this dataset. This is the single most
  important limitation of the project and should be the first thing mentioned in any
  interview or demo.
- **The dataset is daily and irregularly sampled**, not the high-frequency feed a
  production traffic system would use. Sequence length and "future" prediction are
  measured in "next recorded observation," not a fixed real-world time horizon.
- **No live Bengaluru traffic feed exists in this project.** All predictions come from the
  historical CSV. The `/ws/track` endpoint's movement is a clearly-labeled simulation, not
  real GPS.
- **OSM graph download/geocoding untested in the development sandbox** due to a network
  allow-list restriction (see Section 5) — must be verified on a normally-connected machine.
- **Congestion thresholds** (LOW/MODERATE/HIGH/SEVERE) are derived from this dataset's own
  Average Speed quartiles, not validated against real-world Bengaluru congestion definitions.
- This is **not** a production emergency-dispatch system: no integration with actual
  ambulance fleets, hospitals, traffic signal control, or emergency services exists or is
  claimed.

## 14. Future Work

- Integrate an actual real-time Bengaluru traffic feed (e.g. a commercial traffic API) in
  place of / alongside the historical dataset.
- Collect higher-frequency, locally-gathered traffic data to give the LSTM a real temporal
  signal to learn from.
- Incident detection (accidents, road closures) feeding directly into `rerouting.py`'s
  obstruction trigger.
- Weather-integrated prediction (the dataset's weather column is currently a static
  per-row feature, not a live forecast).
- Ambulance priority traffic-signal integration (not implemented or claimed here).
- Production-grade deployment (auth, HTTPS, proper GPS hardware integration, multi-ambulance
  fleet coordination).

## 15. Individual Role (ML + Backend Integration)

This project is structured so its primary contributor can describe their role as:

- **Machine Learning:** dataset analysis, preprocessing/feature engineering, LSTM training
  and evaluation, the prediction module (`predict.py`).
- **Backend:** FastAPI app, integrating the ML prediction module with the routing and
  rerouting modules, request validation, error handling.
- **Database:** SQLite dispatch logging and statistics (`database.py`).
- **Git/GitHub:** repository organization, branch management, integration of components.

Routing/rerouting graph logic (`routing.py`, `rerouting.py`) and the frontend
(`frontend/`) are structured as separate, independently-testable modules — if a teammate
owns those, that division of work is preserved by this project's structure.
