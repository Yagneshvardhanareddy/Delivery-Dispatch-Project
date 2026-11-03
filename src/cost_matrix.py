"""
cost_matrix.py - turn orders + riders into a matrix of estimated delivery times (minutes).

Idea
----
For order i and rider j the estimated time is

    time(i, j) = fixed_min
                 + ( dist(rider_j -> restaurant_i) * pickup_min_per_km
                   + dist(restaurant_i -> customer_i) * min_per_km )
                   * traffic_mult(i) * weather_mult(i) * vehicle_mult(j)

LEARNED FROM THE DATA by derive_model_stats():  fixed_min, min_per_km and the three multipliers.
AN ASSUMPTION:  pickup_min_per_km. The dataset only records the restaurant -> customer distance,
and it never records where riders are, so nothing in the data tells us how fast a rider reaches
a restaurant. We assume normal city riding speed (3 min per km = 20 km/h). In this dataset
distance explains little of the delivery time (correlation about 0.3), so its fitted slope is
NOT a riding speed and must not be used for the rider's trip to the restaurant.
"""
import numpy as np

from src.preprocess import haversine_km

PICKUP_MIN_PER_KM = 3.0   # assumed riding speed to the restaurant: 20 km/h


def derive_model_stats(df):
    """
    Learn the time model from the cleaned historical data.

    Step 1: straight-line fit  time = intercept + slope * distance   (least squares)
            slope     -> min_per_km  (minutes needed per km)
            intercept -> fixed_min   (preparation + handover time that does not depend on distance)
    Step 2: for every traffic level / weather / vehicle, the multiplier is

            multiplier = mean(actual time) / mean(time predicted by the straight line)

            e.g. "Jam" = 1.4 means deliveries in a traffic jam take 40% longer than the
            distance alone would suggest.
    """
    ok = df.dropna(subset=["time_taken_min"])
    if len(ok) < 50:
        raise ValueError("Need at least 50 rows with Time_taken to learn the time model.")

    slope, intercept = np.polyfit(ok["distance_km"], ok["time_taken_min"], 1)
    min_per_km = float(np.clip(slope, 0.2, 6.0))        # only guards against a zero or negative fit
    fixed_min = float(np.clip(intercept, 0.0, 30.0))

    predicted = fixed_min + min_per_km * ok["distance_km"]

    def multipliers(column):
        actual = ok.groupby(column)["time_taken_min"].mean()
        pred = predicted.groupby(ok[column]).mean()
        return (actual / pred).clip(0.6, 2.0).round(4).to_dict()

    return {
        "min_per_km": round(min_per_km, 4),
        "pickup_min_per_km": PICKUP_MIN_PER_KM,
        "fixed_min": round(fixed_min, 4),
        "traffic_mult": multipliers("traffic"),
        "weather_mult": multipliers("weather"),
        "vehicle_mult": multipliers("vehicle"),
        "n_rows_used": int(len(ok)),
    }


def _lookup(mapping, values):
    """Look up a multiplier for each value; unseen categories get 1.0 (no change)."""
    return np.array([mapping.get(v, 1.0) for v in values], dtype=float)


def formula_leg_minutes(stats, distance_km, traffic, weather):
    """
    Time for ONE leg of `distance_km` in the given traffic/weather (no rider information).
    Used as the 'no neural network' baseline when we judge the optional neural network.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    return (stats["fixed_min"]
            + distance_km * stats["min_per_km"]
            * _lookup(stats["traffic_mult"], traffic) * _lookup(stats["weather_mult"], weather))


def build_cost_matrix(orders, riders, stats):
    """
    Build the (n_orders x n_riders) matrix of estimated delivery times in minutes.

    orders needs columns: rest_lat, rest_lon, cust_lat, cust_lon, traffic, weather
    riders needs columns: rider_lat, rider_lon, vehicle

    Returns (cost, d_rider_to_rest) where d_rider_to_rest is the plain distance matrix in km
    (needed by the 'nearest rider first' baseline).
    """
    r_lat = riders["rider_lat"].to_numpy(float)[None, :]      # shape (1, n_riders)
    r_lon = riders["rider_lon"].to_numpy(float)[None, :]
    o_lat = orders["rest_lat"].to_numpy(float)[:, None]       # shape (n_orders, 1)
    o_lon = orders["rest_lon"].to_numpy(float)[:, None]

    # NumPy broadcasting gives the full (n_orders x n_riders) distance matrix in one go
    d_rider_to_rest = haversine_km(r_lat, r_lon, o_lat, o_lon)

    # restaurant -> customer distance does not depend on the rider: one value per order
    d_rest_to_cust = haversine_km(orders["rest_lat"].to_numpy(float), orders["rest_lon"].to_numpy(float),
                                  orders["cust_lat"].to_numpy(float), orders["cust_lon"].to_numpy(float))

    traffic = _lookup(stats["traffic_mult"], orders["traffic"])[:, None]
    weather = _lookup(stats["weather_mult"], orders["weather"])[:, None]
    vehicle = _lookup(stats["vehicle_mult"], riders["vehicle"])[None, :]

    pickup_rate = stats.get("pickup_min_per_km", PICKUP_MIN_PER_KM)
    travel = d_rider_to_rest * pickup_rate + d_rest_to_cust[:, None] * stats["min_per_km"]
    cost = stats["fixed_min"] + travel * traffic * weather * vehicle
    return cost, d_rider_to_rest
