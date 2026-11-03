"""
eda.py - Exploratory Data Analysis helpers.

eda_summary(df)           -> plain dict with the numbers behind every EDA chart (used by the web UI)
save_eda_charts(df, dir)  -> writes PNG charts with matplotlib (distance/traffic/weather vs time)
"""
import os

import numpy as np
import pandas as pd


def eda_summary(df):
    """Aggregate numbers for the dashboard's dataset-overview page."""
    ok = df.dropna(subset=["time_taken_min"])

    bins = [0, 2, 4, 6, 8, 10, 15, 20, 1000]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20+"]
    dist_bin = pd.cut(ok["distance_km"], bins=bins, labels=labels, right=False)
    by_dist = ok.groupby(dist_bin, observed=True)["time_taken_min"].agg(["mean", "count"])

    def group_mean(col, order=None):
        g = ok.groupby(col)["time_taken_min"].agg(["mean", "count"])
        if order:
            g = g.reindex([o for o in order if o in g.index])
        return {"labels": [str(i) for i in g.index], "mean_minutes": g["mean"].round(2).tolist(),
                "count": g["count"].astype(int).tolist()}

    return {
        "rows": int(len(df)),
        "raw_rows": int(df.attrs.get("rows_raw", len(df))),
        "cities": {str(k): int(v) for k, v in df["city_code"].value_counts().items()},
        "mean_minutes": round(float(ok["time_taken_min"].mean()), 2),
        "mean_distance_km": round(float(df["distance_km"].mean()), 2),
        "distance_vs_time": {"labels": [str(i) for i in by_dist.index],
                             "mean_minutes": by_dist["mean"].round(2).tolist(),
                             "count": by_dist["count"].astype(int).tolist()},
        "traffic_vs_time": group_mean("traffic", ["Low", "Medium", "High", "Jam"]),
        "weather_vs_time": group_mean("weather"),
        "vehicle_vs_time": group_mean("vehicle"),
        "hour_vs_time": group_mean("hour"),
    }


def save_eda_charts(df, out_dir="outputs/eda"):
    """Save four PNG charts. Returns the list of file paths."""
    import matplotlib
    matplotlib.use("Agg")                      # no window needed, just write files
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ok = df.dropna(subset=["time_taken_min"])
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    sample = ok.sample(min(3000, len(ok)), random_state=0)
    ax.scatter(sample["distance_km"], sample["time_taken_min"], s=6, alpha=0.3)
    slope, intercept = np.polyfit(ok["distance_km"], ok["time_taken_min"], 1)
    xs = np.linspace(ok["distance_km"].min(), ok["distance_km"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="red", label=f"fit: {slope:.2f} min/km")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery time (min)")
    ax.set_title("Distance vs delivery time")
    ax.legend()
    paths.append(os.path.join(out_dir, "distance_vs_time.png"))
    fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = eda_summary(df)
    for key, title, fname in [("traffic_vs_time", "Traffic vs delivery time", "traffic_vs_time.png"),
                              ("weather_vs_time", "Weather vs delivery time", "weather_vs_time.png"),
                              ("hour_vs_time", "Hour of day vs delivery time", "hour_vs_time.png")]:
        data = summary[key]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(data["labels"], data["mean_minutes"], color="#3b82f6")
        ax.set_ylabel("Mean delivery time (min)")
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        paths.append(os.path.join(out_dir, fname))
        fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
        plt.close(fig)
    return paths
