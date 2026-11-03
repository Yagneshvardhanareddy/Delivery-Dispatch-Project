"""
run_demo.py - run the whole project from the command line (no web page needed).

Examples
    python run_demo.py --data data/train.csv --city BANG --orders 30 --riders 30
    python run_demo.py --synthetic                       # fake data, works without Kaggle file
    python run_demo.py --data data/train.csv --eda       # also save EDA charts to outputs/eda
    python run_demo.py --data data/train.csv --nn        # use the optional neural network
"""
import argparse
import json
import os

import pandas as pd

from src.cost_matrix import derive_model_stats
from src.dispatch import DispatchConfig, available_cities, run_dispatch
from src.eda import save_eda_charts
from src.preprocess import clean_dataframe, load_and_clean
from src.synthetic import make_raw_dataframe
from typing import Any


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.csv", help="path to Kaggle train.csv")
    p.add_argument("--synthetic", action="store_true", help="use fake data in the same format")
    p.add_argument("--city", default=None, help="city code, e.g. BANG (default: the biggest city)")
    p.add_argument("--orders", type=int, default=30)
    p.add_argument("--riders", type=int, default=30)
    p.add_argument("--zones", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lp-method", choices=["two_phase", "big_m"], default="two_phase")
    p.add_argument("--no-rebalance", action="store_true")
    p.add_argument("--nn", action="store_true", help="use the optional PyTorch cost model")
    p.add_argument("--eda", action="store_true", help="save EDA charts to outputs/eda")
    args = p.parse_args()

    if args.synthetic:
        df = clean_dataframe(make_raw_dataframe(6000))
        print("Using SYNTHETIC data (random numbers, only for testing the code).")
    else:
        df = load_and_clean(args.data)
    print(f"Rows: {df.attrs['rows_raw']} raw -> {df.attrs['rows_clean']} clean")
    print("Cities:", ", ".join(f"{c['code']}({c['orders']})" for c in available_cities(df)[:8]), "...")

    stats = derive_model_stats(df)
    print(f"Time model: {stats['fixed_min']} min fixed + {stats['min_per_km']} min/km (delivery leg, fitted) + "
          f"{stats['pickup_min_per_km']} min/km (pickup leg, assumed); traffic x{stats['traffic_mult']}")

    if args.eda:
        for path in save_eda_charts(df):
            print("saved", path)

    bundle = None
    if args.nn:
        from src.nn_cost_model import load_or_train
        bundle = load_or_train(df, stats)

    cfg = DispatchConfig(city=args.city, n_orders=args.orders, n_riders=args.riders, n_zones=args.zones,
                         seed=args.seed, rebalance=not args.no_rebalance, use_nn=args.nn,
                         lp_method=args.lp_method)
    result: Any = run_dispatch(df, stats, cfg, nn_bundle=bundle)

    table = pd.DataFrame(result["strategies"])[
        ["name", "served", "total_minutes", "avg_minutes", "delayed_orders", "profit",
         "improvement_vs_random_pct", "improvement_vs_nearest_pct"]]
    print("\n=== Strategy comparison (times are model estimates) ===")
    print(table.round(2).to_string(index=False))

    z = result["zones"]
    print(f"\nTransportation: VAM start {z['initial_vam_cost_km']:.1f} km -> optimal {z['transport_cost_km']:.1f} km "
          f"({z['modi_iterations']} MODI steps), riders moved: {z['riders_moved']}")
    lp = result["profit_plan"]
    print(f"Profit LP ({lp['method']}): status={lp['status']}, expected profit={lp['objective']:.1f}, "
          f"min-service share used={lp['share_used']}")
    if result["neural_network"]:
        print("Neural network:", result["neural_network"]["metrics"])
    print(f"Runtime: {result['runtime_ms']} ms")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/last_run.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Full result saved to outputs/last_run.json")


if __name__ == "__main__":
    main()
