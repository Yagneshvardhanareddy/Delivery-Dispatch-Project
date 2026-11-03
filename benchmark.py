"""
benchmark.py - robust numbers for the report: average results over many cities and random seeds.

    python benchmark.py                       # all cities, 5 seeds, 30 orders and 30 riders
    python benchmark.py --orders 50 --riders 50 --seeds 10

One run can be lucky, so quote the AVERAGE (and the spread) in your slides, not a single batch.
Saves every run to outputs/benchmark.csv.
"""
import argparse
import os

import pandas as pd

from src.cost_matrix import derive_model_stats
from src.dispatch import DispatchConfig, available_cities, run_dispatch
from src.preprocess import load_and_clean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--orders", type=int, default=30)
    p.add_argument("--riders", type=int, default=30)
    p.add_argument("--zones", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    df = load_and_clean(args.data)
    stats = derive_model_stats(df)
    cities = [c["code"] for c in available_cities(df)]

    rows = []
    for city in cities:
        for seed in range(args.seeds):
            cfg = DispatchConfig(city=city, n_orders=args.orders, n_riders=args.riders,
                                 n_zones=args.zones, seed=seed)
            result = run_dispatch(df, stats, cfg)
            for s in result["strategies"]:
                rows.append({"city": city, "seed": seed, **{k: s[k] for k in (
                    "name", "avg_minutes", "total_minutes", "delayed_orders", "profit",
                    "improvement_vs_random_pct", "improvement_vs_nearest_pct")}})
    table = pd.DataFrame(rows)

    summary = table.groupby("name", sort=False).agg(
        avg_minutes=("avg_minutes", "mean"), std_minutes=("avg_minutes", "std"),
        late_orders=("delayed_orders", "mean"), profit=("profit", "mean"),
        faster_than_random_pct=("improvement_vs_random_pct", "mean"),
        faster_than_nearest_pct=("improvement_vs_nearest_pct", "mean")).round(2)
    print(f"\n{len(cities)} cities x {args.seeds} seeds = {len(cities) * args.seeds} batches of "
          f"{args.orders} orders / {args.riders} riders (times are model estimates)\n")
    print(summary.to_string())

    # how often is the optimal assignment at least as good as the greedy rule?
    wide = table.pivot_table(index=["city", "seed"], columns="name", values="total_minutes")
    ok = (wide["Hungarian (optimal)"] <= wide["Nearest rider first"] + 1e-9).mean() * 100
    print(f"\nHungarian total time <= nearest-rider-first in {ok:.0f}% of the batches (it must be 100%).")
    if "Hungarian + rider rebalancing" in wide:
        better = (wide["Hungarian + rider rebalancing"] < wide["Hungarian (optimal)"] - 1e-9).mean() * 100
        print(f"Rebalancing riders first lowered the total time in {better:.0f}% of the batches.")

    os.makedirs("outputs", exist_ok=True)
    table.to_csv("outputs/benchmark.csv", index=False)
    print("All runs saved to outputs/benchmark.csv")


if __name__ == "__main__":
    main()
