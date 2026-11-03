"""
app.py - FastAPI web server: a thin JSON API on top of src/dispatch.py + the web front end.

Start it:
    uvicorn app:app --reload
then open http://127.0.0.1:8000

Environment variables (all optional):
    DATA_PATH   path to the Kaggle csv        (default: data/train.csv)
    USE_SYNTHETIC=1   ignore the csv and use fake data in the same format (for testing the UI)

API
    GET  /api/status       data source, cities, learned time model, is the neural network available
    GET  /api/eda          numbers behind the dataset charts
    GET  /api/algorithms   small textbook examples solved live by our own solvers
    POST /api/dispatch     run one full dispatch (`body = DispatchConfig fields)
    GET  /docs             automatic interactive API documentation
"""
import importlib.util
import os
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.cost_matrix import derive_model_stats
from src.dispatch import DispatchConfig, available_cities, run_dispatch, _py
from src.eda import eda_summary
from src.hungarian import hungarian
from src.nn_cost_model import MODEL_PATH
from src.preprocess import clean_dataframe, load_and_clean
from src.simplex import simplex
from src.synthetic import make_raw_dataframe
from src.transportation import solve_transportation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE: dict[str, Any] = {}          # filled once at start-up (cleaning 45k rows takes about a second)


def _load_data():
    path = os.environ.get("DATA_PATH", os.path.join(BASE_DIR, "data", "train.csv"))
    if os.environ.get("USE_SYNTHETIC") == "1" or not os.path.exists(path):
        df = clean_dataframe(make_raw_dataframe(8000, seed=7))
        source = "synthetic"
    else:
        df = load_and_clean(path)
        source = "kaggle"
    STATE.update(df=df, stats=derive_model_stats(df), source=source, eda=eda_summary(df),
                 cities=available_cities(df), nn_bundle=None)


@asynccontextmanager
async def lifespan(app):
    _load_data()
    yield


app = FastAPI(title="Optimal Food Delivery Dispatch", lifespan=lifespan)


class DispatchRequest(BaseModel):
    """Body of POST /api/dispatch. Limits keep one request under a few seconds."""
    city: Optional[str] = None
    n_orders: int = Field(30, ge=1, le=200)
    n_riders: int = Field(30, ge=1, le=200)
    n_zones: int = Field(4, ge=1, le=10)
    seed: int = Field(42, ge=0, le=10_000_000)
    rebalance: bool = True
    use_nn: bool = False
    lp_method: Literal["two_phase", "big_m"] = "two_phase"
    revenue_per_order: float = Field(90.0, ge=0, le=10_000)
    rider_cost_per_min: float = Field(1.0, ge=0, le=1_000)
    delay_target_min: float = Field(30.0, ge=1, le=300)
    delay_penalty_per_min: float = Field(1.5, ge=0, le=1_000)
    max_delivery_min: float = Field(60.0, ge=1, le=600)
    min_service_share: float = Field(0.5, ge=0, le=1)


def _nn_available():
    """The network can be used when torch is installed and a trained model file exists."""
    return importlib.util.find_spec("torch") is not None and os.path.exists(MODEL_PATH)


@app.get("/api/status")
def status():
    df = STATE["df"]
    return _py({
        "source": STATE["source"],
        "rows_raw": df.attrs.get("rows_raw", len(df)),
        "rows_clean": len(df),
        "cities": STATE["cities"],
        "time_model": STATE["stats"],
        "nn_available": _nn_available(),
        "defaults": DispatchConfig().__dict__,
    })


@app.get("/api/eda")
def eda():
    return _py(STATE["eda"])


@app.post("/api/dispatch")
def dispatch(req: DispatchRequest):
    valid_codes = {c["code"] for c in STATE["cities"]}
    if req.city is not None and req.city not in valid_codes:
        raise HTTPException(400, f"Unknown city code '{req.city}'. Choose one of: {sorted(valid_codes)}")

    bundle = None
    if req.use_nn:
        if not _nn_available():
            raise HTTPException(400, "The neural network is not available. Install torch and run "
                                     "'python train_nn.py' first, or switch the option off.")
        if STATE["nn_bundle"] is None:
            from src.nn_cost_model import load_or_train
            STATE["nn_bundle"] = load_or_train(STATE["df"], STATE["stats"])
        bundle = STATE["nn_bundle"]

    try:
        cfg = DispatchConfig(**req.model_dump())
        return run_dispatch(STATE["df"], STATE["stats"], cfg, nn_bundle=bundle)
    except ValueError as err:                    # for example: not enough orders in that city
        raise HTTPException(400, str(err))


@app.get("/api/algorithms")
def algorithms():
    """Textbook examples solved LIVE by our own code, shown on the 'How it works' page."""
    cost = np.array([[4, 2, 8], [2, 3, 7], [3, 1, 6]], dtype=float)
    rows, cols, total = hungarian(cost)

    supply, demand = [10, 20], [15, 15]
    tcost = [[2, 3], [5, 4]]
    transport = solve_transportation(supply, demand, tcost)

    lp_max = simplex([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
    lp_min = {m: simplex([2, 3], A_ge=[[1, 1], [1, 3]], b_ge=[4, 6], method=m) for m in ("two_phase", "big_m")}
    return _py({
        "hungarian": {"cost": cost, "pairs": list(zip(rows, cols)), "total": total},
        "transportation": {"supply": supply, "demand": demand, "cost": tcost,
                           "flow": transport["flow"], "initial_cost": transport["initial_cost"],
                           "total_cost": transport["total_cost"], "iterations": transport["iterations"]},
        "simplex": {
            "maximise": {"text": "max 3x + 5y  subject to  x <= 4,  2y <= 12,  3x + 2y <= 18",
                         "x": lp_max["x"], "objective": lp_max["objective"], "iterations": lp_max["iterations"]},
            "minimise": {"text": "min 2x + 3y  subject to  x + y >= 4,  x + 3y >= 6",
                         "two_phase": {"x": lp_min["two_phase"]["x"], "objective": lp_min["two_phase"]["objective"],
                                       "iterations": lp_min["two_phase"]["iterations"]},
                         "big_m": {"x": lp_min["big_m"]["x"], "objective": lp_min["big_m"]["objective"],
                                   "iterations": lp_min["big_m"]["iterations"]}},
        },
    })


# The web front end (index.html, app.js, style.css). Mounted LAST so /api/... routes win.
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "web"), html=True), name="web")
