"""
synthetic.py - a FAKE dataset that has exactly the same messy format as the Kaggle file.

Why: lets every teammate run and test the whole project before downloading train.csv,
and lets the unit tests run anywhere. The numbers are random - never use it for results.
"""
import numpy as np
import pandas as pd

CITY_CENTRES = {          # city code -> (latitude, longitude)
    "BANG": (12.97, 77.59), "INDO": (22.72, 75.86), "MUM": (19.07, 72.87),
    "PUNE": (18.52, 73.86), "HYD": (17.38, 78.48),
}


def make_raw_dataframe(n=4000, seed=0):
    """Return a dataframe that looks like the raw Kaggle train.csv (strings, 'NaN ', '(min) 24')."""
    rng = np.random.default_rng(seed)
    codes = rng.choice(list(CITY_CENTRES), n)
    centre = np.array([CITY_CENTRES[c] for c in codes])
    rest_lat = centre[:, 0] + rng.normal(0, 0.05, n)
    rest_lon = centre[:, 1] + rng.normal(0, 0.05, n)
    cust_lat = rest_lat + rng.uniform(0.01, 0.12, n)
    cust_lon = rest_lon + rng.uniform(0.01, 0.12, n)

    traffic = rng.choice(["Low ", "Medium ", "High ", "Jam "], n, p=[0.3, 0.35, 0.25, 0.1])
    weather = rng.choice(["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"], n)
    vehicle = rng.choice(["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], n,
                         p=[0.5, 0.3, 0.15, 0.05])
    t_mult = pd.Series(traffic, dtype=str).str.strip().map({"Low": 0.9, "Medium": 1.0, "High": 1.2, "Jam": 1.5}).to_numpy()
    w_mult = pd.Series(weather).map({"Sunny": 0.95, "Cloudy": 1.0, "Fog": 1.15, "Windy": 1.0,
                                     "Stormy": 1.2, "Sandstorms": 1.25}).to_numpy()
    v_mult = pd.Series(vehicle, dtype=str).str.strip().map({"motorcycle": 1.0, "scooter": 1.05, "electric_scooter": 1.1, "bicycle": 1.5}).to_numpy()
    dist_km = np.hypot((cust_lat - rest_lat) * 111, (cust_lon - rest_lon) * 108)
    minutes = np.round(8 + 1.6 * dist_km * t_mult * w_mult * v_mult + rng.normal(0, 2.5, n)).clip(10, 60)

    rider_no = rng.integers(1, 40, n)
    df = pd.DataFrame({
        "ID": [f"0x{i:04x}" for i in range(n)],
        "Delivery_person_ID": [f"{c}RES{r:02d}DEL{rng.integers(1, 4):02d}" for c, r in zip(codes, rider_no)],
        "Delivery_person_Age": rng.integers(20, 40, n).astype(str),
        "Delivery_person_Ratings": np.round(rng.uniform(3.5, 5.0, n), 1).astype(str),
        "Restaurant_latitude": rest_lat, "Restaurant_longitude": rest_lon,
        "Delivery_location_latitude": cust_lat, "Delivery_location_longitude": cust_lon,
        "Order_Date": "19-03-2022",
        "Time_Orderd": [f"{h:02d}:{m:02d}:00" for h, m in zip(rng.integers(8, 23, n), rng.integers(0, 60, n))],
        "Weatherconditions": ["conditions " + w for w in weather],
        "Road_traffic_density": traffic,
        "Type_of_order": rng.choice(["Snack ", "Meal ", "Drinks ", "Buffet "], n),
        "Type_of_vehicle": vehicle,
        "City": rng.choice(["Urban ", "Metropolitian ", "Semi-Urban "], n),
        "Time_taken(min)": ["(min) " + str(int(m)) for m in minutes],
    })

    # Add the same kinds of dirt as the real file
    dirty = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[dirty[: len(dirty) // 3], "Delivery_person_Age"] = "NaN "
    df.loc[dirty[len(dirty) // 3: 2 * len(dirty) // 3], "Road_traffic_density"] = "NaN "
    df.loc[dirty[2 * len(dirty) // 3:], "Weatherconditions"] = "conditions NaN"
    zero_rows = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[zero_rows, ["Restaurant_latitude", "Restaurant_longitude"]] = 0.0
    neg_rows = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[neg_rows, "Restaurant_latitude"] = -df.loc[neg_rows, "Restaurant_latitude"]
    return df
