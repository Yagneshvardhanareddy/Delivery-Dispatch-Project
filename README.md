# Optimal Food Delivery Dispatch

Course project for **Optimization Techniques**.
Assign delivery orders to riders so the total delivery time is as small as possible and profit as large as possible, using the methods from class:
**Hungarian (Assignment)**, **Vogel + MODI (Transportation)** and **Simplex (Two-Phase / Big-M)**, all written from scratch in NumPy.

Dataset: Kaggle *Food Delivery Dataset* by Gaurav Malik (`train.csv`).

> **Status: complete and tested on the real Kaggle file.** The optional neural network needs `torch` and is the only part not yet run by the author of this README. See `docs/PROJECT_SUMMARY.md` for results, the work split of the 7 members, the presentation outline and Q&A preparation.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put `train.csv` in the `data/` folder (`test.csv` and `Sample_Submission.csv` are not needed, because only `train.csv` contains the delivery time).

## 2. Run

```bash
python run_demo.py --data data/train.csv --city BANG --orders 30 --riders 30 --eda
python run_demo.py --synthetic        # fake data in the same format, to test without the Kaggle file
pytest -q                             # 180 tests (solvers are checked against SciPy)
python benchmark.py                   # average over all cities and seeds: the numbers to quote in the report
```

### Web dashboard (FastAPI + plain HTML/JS)

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. The page needs internet the first time (fonts, Tailwind, Leaflet, Plotly and the map tiles load from a CDN); there is no build step. Without `data/train.csv` the server starts with fake test data and shows a warning in the top bar. Interactive API docs are at <http://127.0.0.1:8000/docs>.

| Page | What it shows |
| --- | --- |
| Dispatch | batch settings, map with every rider-to-restaurant-to-customer route (drawn one after another), strategy comparison table and chart, zone repositioning (Transportation), profit plan (Simplex) |
| Dataset | cleaned-data statistics, delivery time against distance / traffic / weather / hour, orders per city, the learned time model |
| How it works | the three algorithms explained step by step, with textbook examples solved live by our own code, and the assumptions to state |

Useful command-line options: `--zones 6`, `--lp-method big_m`, `--no-rebalance`, `--nn` (optional PyTorch model).
City codes come from `Delivery_person_ID` (for example `INDO` or `BANG`). The program prints the list of cities it found.

## 3. Folder structure

```
data/train.csv           Kaggle file (not committed to git)
src/preprocess.py        cleaning + Haversine distance
src/eda.py               numbers and PNG charts for distance / traffic / weather vs time
src/cost_matrix.py       learns the time model from data; builds the order x rider cost matrix
src/hungarian.py         Hungarian method (row/column reduction, line cover, unbalanced case)
src/transportation.py    Vogel's Approximation + MODI, dummy node for unbalanced problems
src/simplex.py           Simplex tableau, Two-Phase and Big-M
src/profit_lp.py         zone-level profit LP (uses simplex.py)
src/baseline.py          random and nearest-rider-first rules
src/dispatch.py          run_dispatch(): the whole pipeline, returns a JSON-friendly dict
src/nn_cost_model.py     OPTIONAL PyTorch delivery-time model (write/run last)
app.py                   FastAPI server: /api/status, /api/eda, /api/algorithms, /api/dispatch
web/index.html           the three pages
web/style.css            design tokens and components
web/app.js               API calls, Leaflet map, Plotly charts
tests/                   unit tests (hand-solved examples, SciPy comparison, API)
run_demo.py              command-line demo
benchmark.py             average results over all cities and several random seeds
docs/PROJECT_SUMMARY.md  results, work split for 7 members, PPT outline, Q&A preparation
train_nn.py              OPTIONAL: trains and saves the neural network
```

## 4. How it works

1. **Cleaning.** Strips `(min) ` from `Time_taken(min)`, removes the `conditions ` prefix of the weather, turns the text `NaN` into real missing values, fixes negative coordinates, drops points outside India or far from their city, clips ratings to 1-5.
2. **Time model (learned from data).** `time = fixed + minutes_per_km x distance` (a straight-line fit), plus multipliers for traffic, weather and vehicle (`multiplier = actual mean time / time predicted from distance alone`). On the real file this gives about 21 minutes fixed and 0.54 min per km, with traffic multipliers Low 0.84, Medium 0.99, Jam 1.13, High 1.16.
3. **Cost matrix.** `cost[i, j] = fixed + (dist(rider j to restaurant i) x pickup_pace + dist(restaurant i to customer i) x min_per_km) x traffic x weather x vehicle`. The pickup pace is an assumption (3 min per km, 20 km/h), see section 5.
4. **Hungarian method.** Optimal one-to-one assignment. If there are more orders than riders, dummy riders are added and the unserved orders are the ones that would cost the most.
5. **Transportation.** KMeans cuts the city into zones. Supply = riders per zone, demand = orders per zone, cost = distance between zone centres. Vogel gives the start, MODI makes it optimal. The riders it tells to move are repositioned, then the Hungarian method runs again.
6. **Profit LP.** Maximise `revenue - rider cost - delay penalty` over zone-to-zone flows, subject to rider capacity, a maximum delivery time and a minimum service share per zone (the `>=` rows are why Two-Phase or Big-M is needed).
7. **Comparison.** Random (average of 200 draws) vs nearest-rider-first vs Hungarian vs Hungarian + rebalancing.

## 5. Assumptions and honest limits (say these in the presentation)

- **Pickup pace is an assumption.** The data only has the restaurant-to-customer distance, and distance explains little of the delivery time (correlation 0.32), so its fitted 0.54 min/km is not a riding speed. For the rider's trip to the restaurant we assume 3 min per km (`PICKUP_MIN_PER_KM` in `src/cost_matrix.py`). Change it and re-run `benchmark.py` to see how the gains depend on it.
- The model ignores `multiple_deliveries` and `Festival`, which change delivery times a lot in the data (festival orders take about 20 minutes longer). They are not part of the batch simulation.
- The dataset has **no live rider locations**. Riders are placed near real restaurant locations of the chosen city with about 1 km of random spread.
- There is **no ground truth** for "what if another rider had delivered this order". All times in the comparison are **model estimates**, not measurements.
- Revenue per order (90), rider cost (1 per minute), delay penalty (1.5 per minute) and the 30-minute promise are **assumptions**. The dataset has no prices. They can be changed in `DispatchConfig`.
- The `City` column of the dataset is the city *type*; the real city is read from `Delivery_person_ID`.
- With fewer riders than orders the Hungarian method also chooses WHICH orders to serve, while the baselines do not, so their averages are not directly comparable (the dashboard says so). Compare strategies with equal orders and riders.
- The optional neural network is judged only by its error on held-out real rows (MAE) compared with the formula. If it is not better, report that.
- Tested on the real `train.csv` (45,593 rows, 41,384 after cleaning; about 3,640 rows have all-zero coordinates). The zip also tests on fake data in the same format, so teammates can run `pytest` without the file.
- A KMeans zone can end up with no riders and no orders (it was built around customer points only). That is normal, not a bug.
- Hungarian minimises TOTAL time, not the number of late orders, so on some batches it can have slightly more late orders than the greedy rule.

## 6. Plan

| Session | Content |
| --- | --- |
| 1 | Python engine, tests, command-line demo - done |
| 2 | FastAPI server and the dashboard (map, charts, algorithm pages) - done |
| 3 | Run on the real train.csv, time-model fix, benchmark, final summary with work split for 7 members - done. Optional neural network: train it on your machine with `python train_nn.py` |
