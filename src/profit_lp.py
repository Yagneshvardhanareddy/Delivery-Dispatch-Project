"""
profit_lp.py - Linear Program that plans, at ZONE level, which orders to serve to maximise profit.

Decision variable   x[i, j] = number of orders picked up in zone i that are served by riders
                              currently waiting in zone j.

Objective (maximise)
    profit = revenue - rider cost - delay penalty         summed over all served orders
    profit[i, j] = revenue_per_order
                   - rider_cost_per_min * est_minutes[i, j]
                   - delay_penalty_per_min * max(0, est_minutes[i, j] - delay_target)

Constraints
    sum_j x[i, j] <= orders[i]            cannot serve more orders than exist in zone i
    sum_i x[i, j] <= riders[j]            rider capacity (1 order per rider in a batch)
    sum_j x[i, j] >= share * orders[i]    minimum service level in every zone (a ">=" row,
                                          so the solver needs Two-Phase or Big-M)
    x[i, j] is only allowed if est_minutes[i, j] <= max_delivery_min (maximum delivery time)

This LP is solved with our own Simplex (src/simplex.py).
"""
import numpy as np

from src.preprocess import haversine_km
from src.simplex import simplex


def zone_distance_matrix(centroids):
    """Distance in km between every pair of zone centres."""
    lat = centroids[:, 0]
    lon = centroids[:, 1]
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def build_profit_lp(n_orders, n_riders, zone_km, avg_leg_km, avg_mult, stats, cfg, share):
    """
    Build the matrices of the profit LP.

    n_orders[i]  orders per pickup zone        n_riders[j]  riders per zone
    zone_km[i,j] distance between zone centres avg_leg_km[i] mean restaurant->customer km
    avg_mult[i]  mean traffic*weather multiplier of the orders in zone i
    """
    k = len(n_orders)
    pickup_rate = stats.get("pickup_min_per_km", stats["min_per_km"])
    est = (stats["fixed_min"]
           + (zone_km * pickup_rate + avg_leg_km[:, None] * stats["min_per_km"]) * avg_mult[:, None])
    profit = (cfg.revenue_per_order
              - cfg.rider_cost_per_min * est
              - cfg.delay_penalty_per_min * np.maximum(0.0, est - cfg.delay_target_min))
    allowed = (est <= cfg.max_delivery_min) & (n_orders[:, None] > 0) & (n_riders[None, :] > 0)
    var_idx = [(int(i), int(j)) for i, j in zip(*np.nonzero(allowed))]

    c = np.array([profit[i, j] for i, j in var_idx])
    A_ub, b_ub, A_ge, b_ge = [], [], [], []
    for i in range(k):
        row = np.array([1.0 if vi == i else 0.0 for vi, _ in var_idx])
        if row.any():
            A_ub.append(row)
            b_ub.append(n_orders[i])
            if share > 0:
                A_ge.append(row)
                b_ge.append(share * n_orders[i])
    for j in range(k):
        row = np.array([1.0 if vj == j else 0.0 for _, vj in var_idx])
        if row.any():
            A_ub.append(row)
            b_ub.append(n_riders[j])
    return {"c": c, "A_ub": np.array(A_ub), "b_ub": np.array(b_ub),
            "A_ge": np.array(A_ge) if A_ge else None,
            "b_ge": np.array(b_ge) if b_ge else None,
            "var_idx": var_idx, "est_minutes": est, "profit_matrix": profit}


def zone_profit_plan(order_zone, rider_zone, order_leg_km, order_mult, centroids, stats, cfg,
                     method="two_phase"):
    """
    Solve the zone-level profit LP. If the minimum-service rows make the LP infeasible
    (not enough riders) we relax the service share step by step: share, share/2, 0.
    """
    k = len(centroids)
    n_orders = np.bincount(order_zone, minlength=k).astype(float)
    n_riders = np.bincount(rider_zone, minlength=k).astype(float)
    zone_km = zone_distance_matrix(centroids)

    avg_leg = np.zeros(k)
    avg_mult = np.ones(k)
    for z in range(k):
        mask = order_zone == z
        if mask.any():
            avg_leg[z] = order_leg_km[mask].mean()
            avg_mult[z] = order_mult[mask].mean()

    for share in (cfg.min_service_share, cfg.min_service_share / 2.0, 0.0):
        lp = build_profit_lp(n_orders, n_riders, zone_km, avg_leg, avg_mult, stats, cfg, share)
        if len(lp["var_idx"]) == 0:
            return {"status": "no_variables", "objective": 0.0, "plan": np.zeros((k, k)),
                    "share_used": share, "method": method, "iterations": 0}
        res = simplex(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_ge=lp["A_ge"], b_ge=lp["b_ge"],
                      maximize=True, method=method)
        if res["status"] == "optimal":
            plan = np.zeros((k, k))
            for (i, j), value in zip(lp["var_idx"], res["x"]):
                plan[i, j] = value
            return {"status": "optimal", "objective": res["objective"], "plan": plan,
                    "share_used": share, "method": method, "iterations": res["iterations"],
                    "est_minutes": lp["est_minutes"]}
    return {"status": "infeasible", "objective": 0.0, "plan": np.zeros((k, k)),
            "share_used": 0.0, "method": method, "iterations": 0}
