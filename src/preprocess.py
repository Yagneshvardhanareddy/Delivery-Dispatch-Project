"""
preprocess.py - load and clean the Kaggle "Food Delivery Dataset" (Gaurav Malik).

The raw CSV is messy:
  * Time_taken(min) looks like "(min) 24"      -> we want the integer 24
  * Weatherconditions looks like "conditions Sunny" -> we want "Sunny"
  * Many text cells have trailing spaces ("High ", "motorcycle ")
  * Missing values are written as the TEXT "NaN " (not a real NaN)
  * Some coordinates are 0 or negative (data-entry errors)
  * The "City" column is only the city TYPE (Urban / Metropolitian / Semi-Urban).
    The real city is hidden in Delivery_person_ID, e.g. "BANGRES15DEL02" -> BANG.

Main entry point:  load_and_clean("data/train.csv")
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
INDIA_LAT = (6.0, 38.0)    # rough bounding box of India, used to reject bad coordinates
INDIA_LON = (68.0, 98.0)

# Columns the raw file must contain (Time_taken(min) is optional: test.csv does not have it)
REQUIRED_COLUMNS = [
    "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Weatherconditions", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "City",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in km between two points on Earth.

    Maths (Haversine formula):
        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        d = 2 * R * arcsin( sqrt(a) )
    Works with scalars, numpy arrays and pandas Series (NumPy broadcasting rules apply).
    """
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)  # guard against tiny floating point overshoot
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _is_text(series):
    """True if a column holds text (works for both old 'object' and new 'string' dtypes)."""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _strip_and_fix_nan(df):
    """Strip spaces from every text cell and turn the TEXT 'NaN' into a real missing value."""
    for col in df.columns:
        if _is_text(df[col]):
            df[col] = df[col].str.strip()
    return df.replace({"NaN": np.nan, "nan": np.nan, "": np.nan})


def _drop_far_restaurants(df, max_km=60.0):
    """
    Some restaurants have wrong coordinates and land hundreds of km from their city.
    We drop restaurants farther than max_km from the median restaurant position of their city.
    """
    grp = df.groupby("city_code")[["rest_lat", "rest_lon"]]
    med = grp.transform("median")
    dist = haversine_km(df["rest_lat"], df["rest_lon"], med["rest_lat"], med["rest_lon"])
    keep = dist.isna() | (dist <= max_km)   # rows without a city code are kept
    return df[keep]


def clean_dataframe(raw, require_target=True):
    """
    Clean a raw dataframe read from the Kaggle CSV.

    Returns a new dataframe with tidy snake_case columns:
        order_id, rider_id, city_code, city_type, age, rating, vehicle, order_type,
        rest_lat, rest_lon, cust_lat, cust_lon, traffic, weather,
        distance_km, hour, time_taken_min
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"Dataset is missing expected columns: {missing}")

    n_raw = len(raw)
    df = _strip_and_fix_nan(raw.copy())

    out = pd.DataFrame()
    out["order_id"] = df["ID"]
    out["rider_id"] = df["Delivery_person_ID"]
    # "INDORES13DEL02" -> "INDO" ; "RANCHIRES13DEL01" -> "RANCHI"
    out["city_code"] = df["Delivery_person_ID"].str.extract(r"^([A-Z]+?)RES\d+")[0]
    out["city_type"] = df["City"]

    out["age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    out["rating"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    out["vehicle"] = df["Type_of_vehicle"].str.lower()
    out["order_type"] = df["Type_of_order"]

    # Coordinates: some are negative or 0 by mistake. We take abs() and then reject
    # everything outside India (this also removes the 0.0 rows).
    for new, old in [("rest_lat", "Restaurant_latitude"), ("rest_lon", "Restaurant_longitude"),
                     ("cust_lat", "Delivery_location_latitude"),
                     ("cust_lon", "Delivery_location_longitude")]:
        out[new] = pd.to_numeric(df[old], errors="coerce").abs()

    out["traffic"] = df["Road_traffic_density"].str.title()
    out["weather"] = (df["Weatherconditions"].str.replace("conditions", "", regex=False)
                      .str.strip().replace({"NaN": np.nan}))

    # Target: "(min) 24" -> 24
    if "Time_taken(min)" in df.columns:
        out["time_taken_min"] = pd.to_numeric(
            df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    else:
        out["time_taken_min"] = np.nan

    # Hour of the day the order was placed (optional column, filled with the median if absent)
    if "Time_Orderd" in df.columns:
        hour = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.hour
    else:
        hour = pd.Series(np.nan, index=df.index)
    out["hour"] = hour.fillna(hour.median() if hour.notna().any() else 12).astype(int)

    # ---- validity filters --------------------------------------------------
    in_india = (out["rest_lat"].between(*INDIA_LAT) & out["rest_lon"].between(*INDIA_LON) &
                out["cust_lat"].between(*INDIA_LAT) & out["cust_lon"].between(*INDIA_LON))
    out = out[in_india]
    out = _drop_far_restaurants(out)

    # Distance between restaurant and customer (km)
    out["distance_km"] = haversine_km(out["rest_lat"], out["rest_lon"],
                                      out["cust_lat"], out["cust_lon"])
    out = out[out["distance_km"] > 0.1]

    # Missing categories -> drop the row (only a few percent of the data)
    out = out.dropna(subset=["traffic", "weather", "vehicle"])
    if require_target:
        out = out.dropna(subset=["time_taken_min"])

    # Numeric gaps -> fill with the median. Ratings in the raw file go up to 6 (a typo) -> clip.
    out["rating"] = out["rating"].fillna(out["rating"].median()).clip(1.0, 5.0)
    out["age"] = out["age"].fillna(out["age"].median())

    out = out.reset_index(drop=True)
    out.attrs["rows_raw"] = n_raw
    out.attrs["rows_clean"] = len(out)
    return out


def load_and_clean(path, require_target=True):
    """Read a Kaggle CSV file from disk and return the cleaned dataframe."""
    raw = pd.read_csv(path)
    return clean_dataframe(raw, require_target=require_target)


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    data = load_and_clean(csv)
    print(f"{data.attrs['rows_raw']} raw rows -> {data.attrs['rows_clean']} clean rows")
    print(data.head())
