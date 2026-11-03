"""
baseline.py - the two simple dispatch rules we compare the optimiser against.

  (a) nearest-rider-first : orders are handled one by one in arrival order and each order
                            grabs the closest rider that is still free (greedy).
  (b) random assignment   : riders are given to orders by pure chance.

Both return (rows, cols): order rows[k] is given to rider cols[k].
If there are fewer riders than orders only min(n_orders, n_riders) orders get a rider.
"""
import numpy as np


def nearest_rider_first(d_rider_to_rest):
    """Greedy rule. d_rider_to_rest is the (n_orders x n_riders) distance matrix in km."""
    n_orders, n_riders = d_rider_to_rest.shape
    free = np.ones(n_riders, dtype=bool)
    rows, cols = [], []
    for order in range(min(n_orders, n_riders)):
        distances = np.where(free, d_rider_to_rest[order], np.inf)   # ignore busy riders
        rider = int(np.argmin(distances))
        free[rider] = False
        rows.append(order)
        cols.append(rider)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def random_assignment(n_orders, n_riders, rng):
    """Random one-to-one assignment of min(n_orders, n_riders) pairs."""
    k = min(n_orders, n_riders)
    rows = np.sort(rng.permutation(n_orders)[:k])
    cols = rng.permutation(n_riders)[:k]
    return rows, cols
