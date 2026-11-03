"""Tests for cleaning, cost matrix, profit LP and the full dispatch run (uses fake data)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from src.baseline import nearest_rider_first, random_assignment
from src.cost_matrix import build_cost_matrix, derive_model_stats
from src.dispatch import DispatchConfig, available_cities, make_batch, run_dispatch
from src.nn_cost_model import build_nn_cost_matrix, encode_features
from src.preprocess import clean_dataframe, haversine_km
from src.profit_lp import build_profit_lp
from src.simplex import simplex
from src.synthetic import make_raw_dataframe
from typing import Any


@pytest.fixture(scope="module")
def clean():
    return clean_dataframe(make_raw_dataframe(3000, seed=1))


def test_haversine_known_distance():
    # Bengaluru to Mysuru is roughly 125 km in a straight line
    d = haversine_km(12.9716, 77.5946, 12.2958, 76.6394)
    assert 120 < float(d) < 135


def test_cleaning_fixes_the_messy_formats():
    raw = pd.DataFrame({
        "ID": ["a", "b", "c"], "Delivery_person_ID": ["BANGRES01DEL01", "RANCHIRES02DEL01", "BANGRES03DEL01"],
        "Delivery_person_Age": ["30", "NaN ", "25"], "Delivery_person_Ratings": ["4.5", "6.0", "4.1"],
        "Restaurant_latitude": [12.97, 23.35, 0.0], "Restaurant_longitude": [77.59, 85.33, 0.0],
        "Delivery_location_latitude": [13.0, 23.4, 0.05], "Delivery_location_longitude": [77.65, 85.4, 0.05],
        "Weatherconditions": ["conditions Sunny", "conditions Fog", "conditions NaN"],
        "Road_traffic_density": ["High ", "Jam ", "NaN "], "Type_of_order": ["Snack ", "Meal ", "Meal "],
        "Type_of_vehicle": ["motorcycle ", "scooter ", "bicycle "], "City": ["Urban ", "Metropolitian ", "Urban "],
        "Time_taken(min)": ["(min) 24", "(min) 31", "(min) 18"],
    })
    out = clean_dataframe(raw)
    assert len(out) == 2                                   # the all-zero-coordinate row is removed
    assert list(out["time_taken_min"]) == [24, 31]
    assert list(out["weather"]) == ["Sunny", "Fog"]
    assert list(out["traffic"]) == ["High", "Jam"]
    assert list(out["city_code"]) == ["BANG", "RANCHI"]
    assert out["rating"].max() <= 5.0                      # the typo rating 6.0 is clipped
    assert out["age"].notna().all()                        # the missing age was filled


def test_clean_data_is_valid(clean):
    assert clean["distance_km"].gt(0.1).all()
    assert clean[["traffic", "weather", "vehicle", "time_taken_min"]].notna().all().all()


def test_model_stats_recover_the_synthetic_pattern(clean):
    stats = derive_model_stats(clean)
    assert stats["traffic_mult"]["Jam"] > stats["traffic_mult"]["Low"]     # jams are slower
    assert stats["vehicle_mult"]["bicycle"] > stats["vehicle_mult"]["motorcycle"]
    assert 0.8 <= stats["min_per_km"] <= 6.0


def test_cost_matrix_shape_and_traffic_effect(clean):
    stats = derive_model_stats(clean)
    cfg = DispatchConfig(city="BANG", n_orders=12, n_riders=9)
    orders, riders = make_batch(clean, cfg, np.random.default_rng(0))
    cost, dist = build_cost_matrix(orders, riders, stats)
    assert cost.shape == (12, 9) and dist.shape == (12, 9)
    assert (cost > 0).all()
    jam = orders.copy(); jam["traffic"] = "Jam"
    low = orders.copy(); low["traffic"] = "Low"
    assert build_cost_matrix(jam, riders, stats)[0].sum() > build_cost_matrix(low, riders, stats)[0].sum()


def test_pickup_leg_uses_the_assumed_pace_not_the_fitted_one(clean):
    """Moving a rider 1 degree of latitude farther must add about distance x pickup pace."""
    stats = derive_model_stats(clean)
    assert stats["pickup_min_per_km"] == 3.0
    orders, riders = make_batch(clean, DispatchConfig(city="BANG", n_orders=3, n_riders=3), np.random.default_rng(0))
    near = build_cost_matrix(orders, riders, stats)[0]
    far_riders = riders.copy(); far_riders["rider_lat"] += 0.1      # about 11 km further north
    far, d = build_cost_matrix(orders, far_riders, stats)
    mult = (np.array([stats["traffic_mult"][t] for t in orders["traffic"]])
            * np.array([stats["weather_mult"][w] for w in orders["weather"]]))[:, None]
    veh = np.array([stats["vehicle_mult"][v] for v in riders["vehicle"]])[None, :]
    extra_km = d - build_cost_matrix(orders, riders, stats)[1]
    assert np.allclose(far - near, extra_km * 3.0 * mult * veh, rtol=1e-6)


def test_baselines_return_valid_pairs():
    d = np.random.default_rng(0).random((6, 6))
    rows, cols = nearest_rider_first(d)
    assert sorted(cols) == list(range(6))                  # every rider used exactly once
    rows, cols = random_assignment(8, 5, np.random.default_rng(0))
    assert len(rows) == len(cols) == 5 and len(set(cols)) == 5


def test_profit_lp_matches_linprog(clean):
    stats = derive_model_stats(clean)
    cfg = DispatchConfig()
    n_orders = np.array([8.0, 5.0, 3.0]); n_riders = np.array([4.0, 6.0, 6.0])
    zone_km = np.array([[0, 3, 6], [3, 0, 4], [6, 4, 0]], dtype=float)
    lp = build_profit_lp(n_orders, n_riders, zone_km, np.array([4.0, 5.0, 6.0]), np.ones(3), stats, cfg, 0.5)
    ours = simplex(lp["c"], lp["A_ub"], lp["b_ub"], lp["A_ge"], lp["b_ge"], maximize=True)
    A = np.vstack([lp["A_ub"], -lp["A_ge"]]); b = np.concatenate([lp["b_ub"], -lp["b_ge"]])
    ref = linprog(-lp["c"], A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert ours["status"] == "optimal"
    assert ours["objective"] == pytest.approx(-ref.fun, rel=1e-6)


@pytest.mark.parametrize("method", ["two_phase", "big_m"])
def test_full_dispatch_run(clean, method):
    stats = derive_model_stats(clean)
    cfg = DispatchConfig(city="BANG", n_orders=25, n_riders=25, n_zones=4, lp_method=method, seed=3)
    result: Any = run_dispatch(clean, stats, cfg)
    json.dumps(result)                                     # must be JSON friendly
    by_name = {s["name"]: s for s in result["strategies"]}
    # the optimal assignment can never be worse than the greedy or the random rule
    assert by_name["Hungarian (optimal)"]["total_minutes"] <= by_name["Nearest rider first"]["total_minutes"] + 1e-9
    assert by_name["Hungarian (optimal)"]["total_minutes"] <= by_name["Random assignment"]["total_minutes"]
    assert len(result["assignments"]) == 25
    assert len({a["rider_id"] for a in result["assignments"]}) == 25
    assert result["profit_plan"]["status"] == "optimal"


def test_dispatch_unbalanced_and_single_zone(clean):
    stats = derive_model_stats(clean)
    result: Any = run_dispatch(clean, stats, DispatchConfig(n_orders=20, n_riders=12, n_zones=1, seed=5))
    assert result["strategies"][2]["served"] == 12         # only 12 riders -> 12 orders served
    result = run_dispatch(clean, stats, DispatchConfig(n_orders=10, n_riders=15, n_zones=3, seed=6))
    assert result["strategies"][2]["served"] == 10         # spare riders stay idle


def test_available_cities_lists_codes(clean):
    codes = {c["code"] for c in available_cities(clean, min_orders=100)}
    assert {"BANG", "HYD"} <= codes


def test_neural_network_path_with_stand_in_model(clean):
    """torch is optional, so we test the plumbing with a fake 'model' that is plain NumPy."""
    X = encode_features([2.0, 5.0], ["Low", "Jam"], ["Sunny", "Fog"], [4.5, 4.0])
    assert X.shape == (2, 3 + 6) and X[0, 1] == 0 and X[1, 1] == 3
    stats = derive_model_stats(clean)
    orders, riders = make_batch(clean, DispatchConfig(city="BANG", n_orders=6, n_riders=6), np.random.default_rng(0))
    bundle = {"predict": lambda X: 10 + 3 * X[:, 0], "mean": np.zeros(3), "std": np.ones(3),
              "metrics": {"nn_mae_min": 0.0, "formula_mae_min": 0.0}}
    cost, _ = build_nn_cost_matrix(orders, riders, stats, bundle)
    assert cost.shape == (6, 6) and (cost > 0).all()
    result: Any = run_dispatch(clean, stats, DispatchConfig(city="BANG", n_orders=10, n_riders=10, use_nn=True), nn_bundle=bundle)
    assert result["neural_network"] is not None
    assert any("neural-network" in s["name"] for s in result["strategies"])
