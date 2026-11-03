"""Tests for the FastAPI server (uses fake data, so no Kaggle file is needed)."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["USE_SYNTHETIC"] = "1"
from app import app  # noqa: E402  (must come after the environment variable)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:        # the 'with' runs the start-up code that loads the data
        yield c


def test_status(client):
    body = client.get("/api/status").json()
    assert body["source"] == "synthetic"
    assert len(body["cities"]) >= 3
    assert body["time_model"]["min_per_km"] > 0
    assert body["nn_available"] in (True, False)


def test_eda_has_chart_data(client):
    body = client.get("/api/eda").json()
    for key in ("distance_vs_time", "traffic_vs_time", "weather_vs_time", "hour_vs_time", "cities"):
        assert key in body
    assert len(body["traffic_vs_time"]["labels"]) == len(body["traffic_vs_time"]["mean_minutes"])


def test_algorithm_examples_match_hand_solutions(client):
    body = client.get("/api/algorithms").json()
    assert body["hungarian"]["total"] == 10
    assert body["transportation"]["total_cost"] == 105
    assert body["simplex"]["maximise"]["objective"] == 36
    assert body["simplex"]["minimise"]["two_phase"]["objective"] == 9
    assert body["simplex"]["minimise"]["big_m"]["objective"] == 9


def test_dispatch_returns_everything_the_ui_needs(client):
    city = client.get("/api/status").json()["cities"][0]["code"]
    res = client.post("/api/dispatch", json={"city": city, "n_orders": 20, "n_riders": 20, "n_zones": 3})
    assert res.status_code == 200
    body = res.json()
    for key in ("strategies", "assignments", "zones", "profit_plan", "final_strategy", "runtime_ms"):
        assert key in body
    assert len(body["assignments"]) == 20
    assert {"order_id", "rider_id", "rider", "restaurant", "customer", "est_minutes"} <= set(body["assignments"][0])


def test_dispatch_rejects_bad_input_with_clear_messages(client):
    assert "Unknown city" in client.post("/api/dispatch", json={"city": "NOPE"}).json()["detail"]
    assert client.post("/api/dispatch", json={"n_orders": 5000}).status_code == 422
    assert client.post("/api/dispatch", json={"lp_method": "magic"}).status_code == 422


def test_neural_network_switch_explains_when_unavailable(client):
    if client.get("/api/status").json()["nn_available"]:
        pytest.skip("a trained model exists on this machine")
    res = client.post("/api/dispatch", json={"use_nn": True})
    assert res.status_code == 400 and "not available" in res.json()["detail"]


def test_front_end_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Delivery dispatch optimiser" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
