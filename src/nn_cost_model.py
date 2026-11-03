"""
nn_cost_model.py - OPTIONAL small neural network (PyTorch) that predicts delivery time.

Keep this for LAST. The project is complete without it; nothing else imports torch.

What it does
    A feed-forward network (MLP) learns   time = f(distance, traffic, weather, rider rating)
    from the historical rows. We then use its predictions as an alternative cost matrix:

        cost_nn(order i, rider j) = NN time of the restaurant->customer leg for order i, rider j's rating
                                    + pickup time of rider j to the restaurant (distance formula)

    It can be switched on or off with  DispatchConfig(use_nn=True/False)  or  --nn  in run_demo.py.

How to judge it honestly
    We hold out 20% of the rows and compare the MAE (mean absolute error, in minutes) of
        - the neural network
        - the simple distance/traffic/weather formula from cost_matrix.py
    If the network is not clearly better, say so in the report - that is a valid result.

GPU: the model is tiny; it uses CUDA automatically when available, otherwise the CPU.
"""
import os

import numpy as np

from src.cost_matrix import _lookup, formula_leg_minutes
from src.preprocess import haversine_km

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]
WEATHER_TYPES = ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"]
MODEL_PATH = os.path.join("models", "delivery_time_nn.pt")


def encode_features(distance_km, traffic, weather, rating):
    """
    Turn raw values into a numeric matrix (no torch needed).
    columns: distance_km, traffic level (0..3), rider rating, then one-hot weather.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    traffic_idx = np.array([TRAFFIC_LEVELS.index(t) if t in TRAFFIC_LEVELS else 1 for t in traffic], dtype=float)
    one_hot = np.zeros((len(distance_km), len(WEATHER_TYPES)))
    for row, w in enumerate(weather):
        if w in WEATHER_TYPES:
            one_hot[row, WEATHER_TYPES.index(w)] = 1.0
    return np.column_stack([distance_km, traffic_idx, np.asarray(rating, dtype=float), one_hot])


def predict_minutes(bundle, distance_km, traffic, weather, rating):
    """Predict delivery minutes for many rows with a trained bundle."""
    X = encode_features(distance_km, traffic, weather, rating)
    X[:, :3] = (X[:, :3] - bundle["mean"]) / bundle["std"]     # same scaling as in training
    return np.asarray(bundle["predict"](X), dtype=float)


def build_nn_cost_matrix(orders, riders, stats, bundle):
    """Alternative (n_orders x n_riders) cost matrix that uses the neural network."""
    n_o, n_r = len(orders), len(riders)
    leg_km = haversine_km(orders["rest_lat"].to_numpy(float), orders["rest_lon"].to_numpy(float),
                          orders["cust_lat"].to_numpy(float), orders["cust_lon"].to_numpy(float))
    # one row of features for every (order, rider) pair
    order_idx = np.repeat(np.arange(n_o), n_r)
    rider_idx = np.tile(np.arange(n_r), n_o)
    leg_minutes = predict_minutes(
        bundle, leg_km[order_idx], orders["traffic"].to_numpy()[order_idx],
        orders["weather"].to_numpy()[order_idx], riders["rating"].to_numpy(float)[rider_idx],
    ).reshape(n_o, n_r)

    d_pickup = haversine_km(riders["rider_lat"].to_numpy(float)[None, :], riders["rider_lon"].to_numpy(float)[None, :],
                            orders["rest_lat"].to_numpy(float)[:, None], orders["rest_lon"].to_numpy(float)[:, None])
    pickup = (d_pickup * stats.get("pickup_min_per_km", stats["min_per_km"])
              * _lookup(stats["traffic_mult"], orders["traffic"])[:, None]
              * _lookup(stats["weather_mult"], orders["weather"])[:, None]
              * _lookup(stats["vehicle_mult"], riders["vehicle"])[None, :])
    return leg_minutes + pickup, d_pickup


def train_model(df, stats, epochs=40, lr=1e-3, batch_size=256, seed=0, save_path=MODEL_PATH):
    """
    Train the network on the cleaned dataframe and return a 'bundle' dict.
    Bundle keys: predict (function), mean, std, metrics (MAE of the NN and of the formula).
    """
    import torch
    from torch import nn

    ok = df.dropna(subset=["time_taken_min"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ok))
    cut = int(0.8 * len(ok))
    train, val = ok.iloc[order[:cut]], ok.iloc[order[cut:]]

    X_train = encode_features(train["distance_km"], train["traffic"], train["weather"], train["rating"])
    X_val = encode_features(val["distance_km"], val["traffic"], val["weather"], val["rating"])
    mean = X_train[:, :3].mean(axis=0)
    std = X_train[:, :3].std(axis=0) + 1e-8
    X_train[:, :3] = (X_train[:, :3] - mean) / std
    X_val[:, :3] = (X_val[:, :3] - mean) / std
    y_train = train["time_taken_min"].to_numpy(float)
    y_val = val["time_taken_min"].to_numpy(float)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(X_train.shape[1], 32), nn.ReLU(),
                          nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()

    model.eval()

    def predict(X):
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
            return model(tensor).squeeze(1).cpu().numpy()

    nn_mae = float(np.abs(predict(X_val) - y_val).mean())
    formula_pred = formula_leg_minutes(stats, val["distance_km"], val["traffic"], val["weather"])
    formula_mae = float(np.abs(formula_pred - y_val).mean())
    bundle = {"predict": predict, "mean": mean, "std": std,
              "metrics": {"nn_mae_min": round(nn_mae, 3), "formula_mae_min": round(formula_mae, 3),
                          "val_rows": int(len(val)), "device": str(device)}}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "in_dim": X_train.shape[1], "metrics": bundle["metrics"]}, save_path)
    return bundle


def load_or_train(df, stats, path=MODEL_PATH, **train_kwargs):
    """Load a saved model if it exists, otherwise train a new one (takes about a minute)."""
    import torch
    from torch import nn

    if os.path.exists(path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = nn.Sequential(nn.Linear(saved["in_dim"], 32), nn.ReLU(),
                              nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        model.load_state_dict(saved["state_dict"])
        model.eval()

        def predict(X):
            with torch.no_grad():
                return model(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1).numpy()

        return {"predict": predict, "mean": saved["mean"], "std": saved["std"], "metrics": saved["metrics"]}
    return train_model(df, stats, save_path=path, **train_kwargs)
