"""
dispatch.py - the service layer. ONE function, run_dispatch(), runs the whole project:

    1. make a batch        pick N real orders from the dataset and place M riders in the city
    2. cost matrix         estimated delivery time for every (order, rider) pair
    3. baselines           random and nearest-rider-first
    4. Hungarian           optimal one-to-one assignment (minimum total time)
    5. Transportation      zone level: how many riders should move from surplus zones
                           to zones with more orders (VAM + MODI), then Hungarian again
    6. Profit LP           zone level plan that maximises profit (Simplex)
    7. (optional) NN cost  assignment with delivery times predicted by a neural network

The result is a plain dict (numbers, lists, strings) so ANY front end can use it:
a web API, a notebook, or a command line script.

Honest note: the dataset only has the time of the rider who really delivered each order.
There is no ground truth for "what if another rider had delivered it", so every time in the
comparison table is a MODEL ESTIMATE (see cost_matrix.py), not a measurement.
"""
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.baseline import nearest_rider_first, random_assignment
from src.cost_matrix import _lookup, build_cost_matrix
from src.hungarian import hungarian
from src.preprocess import haversine_km
from src.profit_lp import zone_distance_matrix, zone_profit_plan
from src.transportation import solve_transportation

N_RANDOM_TRIALS = 200


@dataclass
class DispatchConfig:
    """All the knobs of one dispatch run (the web UI sends these)."""
    city: str | None = None             # city code such as "BANG"; None = the city with most orders
    n_orders: int = 30
    n_riders: int = 30
    n_zones: int = 4
    seed: int = 42
    rebalance: bool = True       # run the Transportation step and move riders between zones
    use_nn: bool = False         # optional neural-network cost matrix
    lp_method: str = "two_phase"  # or "big_m"
    # ---- business assumptions (NOT in the dataset, state them in the report) -----------
    revenue_per_order: float = 90.0      # delivery fee + commission, in rupees
    rider_cost_per_min: float = 1.0      # rupees paid per minute of rider time
    delay_target_min: float = 30.0       # promised delivery time
    delay_penalty_per_min: float = 1.5   # rupees lost per minute beyond the promise
    max_delivery_min: float = 60.0       # an order is never planned above this time
    min_service_share: float = 0.5       # every zone must get at least this share of its orders served


# ----------------------------------------------------------------------------- helpers
def _py(obj):
    """Convert numpy types to plain Python so the result can be turned into JSON."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def available_cities(df, min_orders=100):
    """City codes with enough orders to build a batch (for the city selector)."""
    counts = df["city_code"].value_counts()
    return [{"code": str(code), "orders": int(n)} for code, n in counts.items() if n >= min_orders]


def make_batch(df, cfg, rng):
    """
    Build one dispatch batch.
      * orders: real rows from the dataset (restaurant, customer, traffic, weather)
      * riders: real rider attributes (rating, vehicle). The dataset has no live GPS of riders,
        so each rider waits near a randomly chosen real restaurant of the city (~1 km spread).
    """
    pool = df if cfg.city is None else df[df["city_code"] == cfg.city]
    if len(pool) < cfg.n_orders:
        raise ValueError(f"Only {len(pool)} orders available for city {cfg.city!r}.")

    sampled = pool.sample(cfg.n_orders, random_state=int(rng.integers(1_000_000_000)))
    orders = pd.DataFrame({
        "order_id": sampled["order_id"].to_numpy(),
        "rest_lat": sampled["rest_lat"].to_numpy(), "rest_lon": sampled["rest_lon"].to_numpy(),
        "cust_lat": sampled["cust_lat"].to_numpy(), "cust_lon": sampled["cust_lon"].to_numpy(),
        "traffic": sampled["traffic"].to_numpy(), "weather": sampled["weather"].to_numpy(),
        "recorded_minutes": sampled["time_taken_min"].to_numpy(),
    })

    rider_table = pool.drop_duplicates("rider_id")
    replace = len(rider_table) < cfg.n_riders
    chosen = rider_table.sample(cfg.n_riders, replace=replace,
                                random_state=int(rng.integers(1_000_000_000)))
    spots = pool.sample(cfg.n_riders, replace=True, random_state=int(rng.integers(1_000_000_000)))
    riders = pd.DataFrame({
        "rider_id": [f"{rid}#{k}" if replace else rid for k, rid in enumerate(chosen["rider_id"])],
        "rating": chosen["rating"].to_numpy(),
        "vehicle": chosen["vehicle"].to_numpy(),
        "rider_lat": spots["rest_lat"].to_numpy() + rng.normal(0, 0.008, cfg.n_riders),
        "rider_lon": spots["rest_lon"].to_numpy() + rng.normal(0, 0.008, cfg.n_riders),
    })
    return orders.reset_index(drop=True), riders.reset_index(drop=True)


def build_zones(orders, riders, n_zones, seed):
    """
    Cluster all restaurant and customer points into zones with KMeans.
    Returns centroids (k x 2), the zone of every order (by its restaurant) and of every rider.
    """
    pts = np.vstack([orders[["rest_lat", "rest_lon"]].to_numpy(),
                     orders[["cust_lat", "cust_lon"]].to_numpy()])
    k = int(max(1, min(n_zones, len(pts))))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pts)
    order_zone = km.predict(orders[["rest_lat", "rest_lon"]].to_numpy())
    rider_zone = km.predict(riders[["rider_lat", "rider_lon"]].to_numpy())
    return km.cluster_centers_, order_zone, rider_zone


def reposition_riders(riders, rider_zone, flow, centroids, rng):
    """
    Apply the Transportation plan: flow[i, j] riders leave zone i and go to zone j.
    We move the riders of zone i that are closest to the centre of zone j.
    """
    riders = riders.copy()
    rider_zone = rider_zone.copy()
    moved = 0
    k = len(centroids)
    for i in range(k):
        for j in range(k):
            count = int(round(flow[i, j]))
            if i == j or count <= 0:
                continue
            members = np.flatnonzero(rider_zone == i)
            dist = haversine_km(riders.loc[members, "rider_lat"].to_numpy(),
                                riders.loc[members, "rider_lon"].to_numpy(),
                                centroids[j, 0], centroids[j, 1])
            for idx in members[np.argsort(dist)[:count]]:
                riders.loc[idx, "rider_lat"] = centroids[j, 0] + rng.normal(0, 0.004)
                riders.loc[idx, "rider_lon"] = centroids[j, 1] + rng.normal(0, 0.004)
                rider_zone[idx] = j
                moved += 1
    return riders, rider_zone, moved


def evaluate(cost, rows, cols, cfg, n_orders):
    """Total/average time, delayed orders and profit for a set of (order, rider) pairs."""
    times = cost[rows, cols]
    served = len(rows)
    delay = np.maximum(0.0, times - cfg.delay_target_min)
    profit = (served * cfg.revenue_per_order - cfg.rider_cost_per_min * times.sum()
              - cfg.delay_penalty_per_min * delay.sum())
    return {"served": served, "unserved": n_orders - served,
            "total_minutes": float(times.sum()),
            "avg_minutes": float(times.mean()) if served else 0.0,
            "delayed_orders": int((times > cfg.delay_target_min).sum()),
            "profit": float(profit)}


def _with_improvements(rows):
    """Add '% faster than random / nearest-rider-first' to every strategy row."""
    base = {r["name"]: r["avg_minutes"] for r in rows}
    for r in rows:
        for key, ref in (("improvement_vs_random_pct", "Random assignment"),
                         ("improvement_vs_nearest_pct", "Nearest rider first")):
            r[key] = round(100.0 * (base[ref] - r["avg_minutes"]) / base[ref], 2) if base[ref] else 0.0
    return rows


# -------------------------------------------------------------------------- main entry
def run_dispatch(df, stats, cfg, nn_bundle=None):
    """Run the full pipeline once and return a JSON-friendly dict."""
    t0 = time.perf_counter()
    if cfg.city is None:
        # a batch only makes sense inside ONE city (riders cannot ride between cities),
        # so without a choice we use the city with the most orders
        cfg = replace(cfg, city=str(df["city_code"].value_counts().idxmax()))
    rng = np.random.default_rng(cfg.seed)
    orders, riders = make_batch(df, cfg, rng)
    n_orders, n_riders = len(orders), len(riders)

    # 2. cost matrix and baselines (rider positions BEFORE any repositioning)
    cost0, d_rr0 = build_cost_matrix(orders, riders, stats)
    strategies = []

    rand_metrics = []
    for _ in range(N_RANDOM_TRIALS):
        r, c = random_assignment(n_orders, n_riders, rng)
        rand_metrics.append(evaluate(cost0, r, c, cfg, n_orders))
    mean_rand = {k: float(np.mean([m[k] for m in rand_metrics])) for k in rand_metrics[0]}
    strategies.append({"name": "Random assignment", **mean_rand})

    r_near, c_near = nearest_rider_first(d_rr0)
    strategies.append({"name": "Nearest rider first", **evaluate(cost0, r_near, c_near, cfg, n_orders)})

    # 4. Hungarian (optimal)
    r_h, c_h, _ = hungarian(cost0)
    strategies.append({"name": "Hungarian (optimal)", **evaluate(cost0, r_h, c_h, cfg, n_orders)})
    final = {"rows": r_h, "cols": c_h, "cost": cost0, "riders": riders, "label": "Hungarian (optimal)"}

    # 5. Zones + Transportation problem
    centroids, order_zone, rider_zone = build_zones(orders, riders, cfg.n_zones, cfg.seed)
    k = len(centroids)
    supply = np.bincount(rider_zone, minlength=k).astype(float)    # riders per zone
    demand = np.bincount(order_zone, minlength=k).astype(float)    # orders per zone
    zone_km = zone_distance_matrix(centroids)
    transport = solve_transportation(supply, demand, zone_km)
    zones_info = {
        "centroids": centroids, "riders_per_zone": supply, "orders_per_zone": demand,
        "flow": transport["flow"], "transport_cost_km": transport["total_cost"],
        "initial_vam_cost_km": transport["initial_cost"], "modi_iterations": transport["iterations"],
        "dummy": transport["dummy"], "riders_moved": 0,
    }

    if cfg.rebalance:
        riders2, rider_zone2, moved = reposition_riders(riders, rider_zone, transport["flow"], centroids, rng)
        cost1, _ = build_cost_matrix(orders, riders2, stats)
        r1, c1, _ = hungarian(cost1)
        name = "Hungarian + rider rebalancing"
        strategies.append({"name": name, **evaluate(cost1, r1, c1, cfg, n_orders)})
        zones_info["riders_moved"] = moved
        final = {"rows": r1, "cols": c1, "cost": cost1, "riders": riders2, "label": name}

    # 6. Profit LP at zone level
    leg_km = haversine_km(orders["rest_lat"].to_numpy(), orders["rest_lon"].to_numpy(),
                          orders["cust_lat"].to_numpy(), orders["cust_lon"].to_numpy())
    mult = (_lookup(stats["traffic_mult"], orders["traffic"]) * _lookup(stats["weather_mult"], orders["weather"]))
    lp = zone_profit_plan(order_zone, rider_zone, leg_km, mult, centroids, stats, cfg, cfg.lp_method)

    # 7. Optional neural-network cost matrix
    nn_info = None
    if cfg.use_nn and nn_bundle is not None:
        from src.nn_cost_model import build_nn_cost_matrix
        cost_nn, _ = build_nn_cost_matrix(orders, riders, stats, nn_bundle)
        r_n, c_n, _ = hungarian(cost_nn)
        strategies.append({"name": "Hungarian + neural-network cost",
                           **evaluate(cost0, r_n, c_n, cfg, n_orders)})
        nn_info = {"metrics": nn_bundle["metrics"],
                   "total_minutes_under_nn_costs": float(cost_nn[r_n, c_n].sum()),
                   "total_minutes_hungarian_under_nn_costs": float(cost_nn[r_h, c_h].sum())}

    strategies = _with_improvements(strategies)

    # assignments of the final (best) strategy, ready to draw on a map
    pairs = []
    for o, r in zip(final["rows"], final["cols"]):
        rider = final["riders"].iloc[r]
        order = orders.iloc[o]
        pairs.append({
            "order_id": order["order_id"], "rider_id": rider["rider_id"],
            "rider": [rider["rider_lat"], rider["rider_lon"]],
            "restaurant": [order["rest_lat"], order["rest_lon"]],
            "customer": [order["cust_lat"], order["cust_lon"]],
            "est_minutes": final["cost"][o, r], "traffic": order["traffic"], "weather": order["weather"],
        })

    return _py({
        "config": asdict(cfg),
        "model_stats": stats,
        "n_orders": n_orders, "n_riders": n_riders,
        "strategies": strategies,
        "final_strategy": final["label"],
        "assignments": pairs,
        "zones": zones_info,
        "profit_plan": lp,
        "neural_network": nn_info,
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 1),
        "note": "All delivery times are model estimates, not measurements.",
    })
