"""
train_nn.py - OPTIONAL: train the PyTorch delivery-time model and save it to models/delivery_time_nn.pt.

    pip install torch
    python train_nn.py                     # uses data/train.csv
    python train_nn.py --epochs 60

After this, the dashboard enables the "Use neural-network costs" switch.
Uses the GPU automatically when CUDA is available.
"""
import argparse

from src.cost_matrix import derive_model_stats
from src.nn_cost_model import MODEL_PATH, train_model
from src.preprocess import load_and_clean

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    frame = load_and_clean(args.data)
    bundle = train_model(frame, derive_model_stats(frame), epochs=args.epochs)
    m = bundle["metrics"]
    print(f"Saved to {MODEL_PATH}")
    print(f"Validation MAE: neural network {m['nn_mae_min']} min vs formula {m['formula_mae_min']} min "
          f"({m['val_rows']} held-out rows, device {m['device']})")
    print("If the network is not clearly better, say so in the report - that is a valid result.")
