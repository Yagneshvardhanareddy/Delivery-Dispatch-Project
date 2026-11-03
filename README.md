# Delivery Dispatch Optimiser


A small decision tool for food delivery. Given a batch of waiting orders and a pool of riders, it decides **which rider should take which order** so that the total delivery time is as low as possible. It can also move riders between city zones before dispatch, and it can plan which zone-to-zone deliveries to serve to keep profit high.

The three optimisation methods are written from scratch in NumPy and checked against SciPy:

| Step | Method | What it decides |
| --- | --- | --- |
| 1 | Assignment problem, **Hungarian method** | one rider for each order, minimum total time |
| 2 | Transportation problem, **Vogel's approximation + MODI** | how many riders to move from zone to zone before dispatch |
| 3 | Linear programming, **Simplex (Two-Phase and Big-M)** | which zone-to-zone deliveries to serve to maximise profit |

A web dashboard shows the routes on a map, compares the result with simple rules, and explains each algorithm with small live examples.

Dataset: the Kaggle *Food Delivery Dataset* by Gaurav Malik (`train.csv`).

Working Hosted Webapp link- https://delivery-dispatch-project.onrender.com/

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Ways to run it](#2-ways-to-run-it)
3. [The web dashboard](#3-the-web-dashboard)
4. [How it works](#4-how-it-works)
5. [Benchmarks](#5-benchmarks)
6. [Project structure](#6-project-structure)
7. [Testing](#7-testing)
8. [Hosting it online](#8-hosting-it-online)
9. [Settings you can change](#9-settings-you-can-change)
10. [Assumptions and limits](#10-assumptions-and-limits)
11. [Ideas for improvement](#11-ideas-for-improvement)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Setup

You need Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The data file is expected at `data/train.csv`. To use another location, set the `DATA_PATH` environment variable.

## 2. Run

**Web dashboard**

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Fonts, Leaflet, Plotly and the map tiles load from the internet, so you need a connection. API docs are at <http://127.0.0.1:8000/docs>.

**Command line**

```bash
python run_demo.py --data data/train.csv --city BANG --orders 30 --riders 30 --eda
```

| Option | What it does |
| --- | --- |
| `--city BANG` | pick a city by its code (the program prints the list it found) |
| `--orders 30 --riders 30` | batch size |
| `--zones 6` | number of city zones |
| `--lp-method big_m` | use Big-M instead of Two-Phase for the profit plan |
| `--no-rebalance` | skip the rider repositioning step |
| `--eda` | also save the exploratory charts as PNG files |
| `--nn` | use the neural network time model |

**Tests and benchmark**

```bash
pytest -q
python benchmark.py
```

---

## 3. The dashboard

| Page | What you see |
| --- | --- |
| **Dispatch** | batch settings, a map of every route, a comparison of the strategies, the zone plan and the profit plan |
| **Dataset** | cleaned-data statistics, delivery time against distance, traffic, weather and hour, orders per city, and the learned time model |
| **How it works** | the three algorithms explained step by step, with small examples solved live by the project's own code |

**Reading the map**

| Symbol | Meaning |
| --- | --- |
| Blue dot | rider |
| Orange dot | restaurant |
| Black dot | customer |
| Dashed circle | centre of a city zone |
| Blue dashed line | rider travelling to the restaurant |
| Orange solid line | restaurant to customer |

The lines are straight, not real roads, because the data has coordinates but no road network. Hover over a route to highlight it, or untick **Show routes on the map** to hide all lines.

**Batch settings:** city, number of orders, number of riders (fewer riders than orders leaves some orders unserved), number of zones, random seed (same seed gives the same batch), rebalancing on or off, the profit LP method, and the neural network switch.

---

## 4. How it works

**Cleaning.** Removes the `(min) ` text from the time column, removes the `conditions ` prefix from the weather, turns the text `NaN` into real missing values, fixes negative coordinates, drops points outside India or far from their city, keeps ratings between 1 and 5, and computes the Haversine distance between restaurant and customer. The real city code is read from `Delivery_person_ID`, because the `City` column is only the city type.

**Time model.** Learned from the data:

```
time = fixed + minutes_per_km x distance
```

with multipliers for traffic, weather and vehicle. On the real file this gives about 21 minutes fixed and 0.54 minutes per km. Traffic multipliers are Low 0.84, Medium 0.99, Jam 1.13, High 1.16.

**Cost matrix.** For every order *i* and rider *j*:

```
cost[i, j] = fixed
           + ( dist(rider j -> restaurant i) x pickup_pace
             + dist(restaurant i -> customer i) x minutes_per_km )
             x traffic x weather x vehicle
```

**Hungarian method.** Finds the one-to-one assignment with the lowest total cost using row and column reduction and line covering. With more orders than riders, dummy riders are added, and the orders given to them are left unserved.

**Vogel + MODI.** KMeans cuts the city into zones. Supply is riders per zone, demand is orders per zone, and cost is the distance between zone centres. Vogel gives a good starting plan and MODI makes it optimal. The riders it tells to move are repositioned, then the Hungarian method runs again.

**Simplex.** A zone-level linear program maximises `revenue - rider cost - late penalty`, subject to rider capacity, a maximum delivery time and a minimum service share per zone. The "at least" rows need an artificial start, which is why both Two-Phase and Big-M are implemented. They give the same answers.

**Comparison.** Four strategies run on the same batch: random assignment (average of 200 draws), nearest rider first, Hungarian, and Hungarian with rebalancing.

---

## 5. Benchmarks

```bash
python benchmark.py                                  # all cities, 5 seeds, 30 orders and 30 riders
python benchmark.py --orders 50 --riders 50 --seeds 10
```

Every run is saved to `outputs/benchmark.csv`.

Average over **110 batches** (22 cities x 5 seeds), 30 orders and 30 riders. Times are model estimates.

| Strategy | Avg minutes per order | Faster than nearest rider | Late orders (of 30) | Profit per batch (Rs) |
| --- | --- | --- | --- | --- |
| Random assignment | 40.0 | -23.8% | 23.2 | 1,023 |
| Nearest rider first | 32.1 | baseline | 14.5 | 1,568 |
| Hungarian | 30.8 | 3.8% | 13.6 | 1,667 |
| Hungarian + rebalancing | 30.0 | 5.9% | 13.0 | 1,720 |

- Hungarian was at least as good as nearest rider first in **100%** of batches.
- Rebalancing lowered the total time in **75%** of batches.
- With 60 orders and 60 riders (66 batches), results are similar: about 3.8% and 6.2% faster than nearest rider first.
- Results are steadier too: the spread of average time between batches is 1.6 minutes with rebalancing, against 9.0 minutes for random assignment.

The gain over a good greedy rule is modest (4 to 6%) because most of a delivery's time is fixed preparation and handover, which dispatch cannot change. Compare strategies with equal orders and riders, because with fewer riders than orders Hungarian also chooses which orders to serve.

---

## 6. Project structure

```
app.py                   FastAPI server and API
run_demo.py              command line demo
benchmark.py             averages over all cities and seeds
train_nn.py              trains and saves the neural network

data/train.csv           Kaggle file
models/                  saved neural network
outputs/                 benchmark results and charts

src/preprocess.py        cleaning and Haversine distance
src/eda.py               exploratory numbers and charts
src/cost_matrix.py       time model and order x rider cost matrix
src/hungarian.py         Hungarian method
src/transportation.py    Vogel's approximation + MODI
src/simplex.py           Simplex, Two-Phase and Big-M
src/profit_lp.py         zone-level profit LP
src/baseline.py          random and nearest-rider-first rules
src/dispatch.py          run_dispatch(): the whole pipeline
src/nn_cost_model.py     neural network time model
src/synthetic.py         fake data in the same format, used by tests

web/index.html           the three pages
web/style.css            styling
web/app.js               API calls, map, charts

tests/                   unit tests
```

**API:** `GET /api/status`, `GET /api/eda`, `GET /api/algorithms`, `POST /api/dispatch`, `GET /docs`.

---

## 7. Testing

```bash
pytest -q
```

Tests cover hand-solved examples for each method, comparison with SciPy on random problems (including unbalanced, degenerate, infeasible and unbounded cases), the full pipeline, and the API. SciPy is used only in the tests, to prove the from-scratch solvers give the same answers.

---

## 8. Hosting

The app is one FastAPI server that also serves the web page. On Render's free plan:

1. Push the project to GitHub with `app.py` at the root.
2. On render.com choose **New → Web Service** and select the repository.
3. Build command: `pip install -r requirements-deploy.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy.

`requirements-deploy.txt` has no PyTorch, so the neural network switch stays off on that host. To keep it, install `torch` in the build command and commit `models/delivery_time_nn.pt` (it is in `.gitignore` by default). The free plan sleeps when idle, so the first visit can take 30 to 60 seconds.

---

## 9. Assumptions and limits

- **All times are model estimates.** The data records only the rider who really delivered each order, so there is no measured time for a different rider.
- **No live rider locations.** Riders are placed near real restaurants of the chosen city with about 1 km of random spread.
- **Pickup pace is an assumption.** Distance explains little of delivery time in this data (correlation 0.32), so the fitted 0.54 min/km is not a riding speed. The rider's trip to the restaurant uses 3 minutes per km (`PICKUP_MIN_PER_KM` in `src/cost_matrix.py`). The gains depend on this number.
- **Prices are assumptions.** Revenue per order (90), rider cost (1 per minute), late penalty (1.5 per minute) and the 30-minute promise are not in the data. Change them in `DispatchConfig` in `src/dispatch.py`.
- **`multiple_deliveries` and `Festival` are ignored.** Festival orders take about 20 minutes longer in the data.
- **Hungarian minimises total time, not the number of late orders**, so some batches can have slightly more late orders than the greedy rule.
- **Empty zones are normal.** KMeans zones are built around customer points only.

---

## 10. Neural network time model

As an extra feature, a small feed-forward network (PyTorch) predicts delivery time from distance, traffic level, rider rating and weather. Its predictions can replace the formula in the cost matrix:

```
cost_nn[i, j] = NN time of the restaurant -> customer leg
              + pickup time of rider j to the restaurant
```

**Train it**

```bash
python train_nn.py                 # saves models/delivery_time_nn.pt
python train_nn.py --epochs 60
```

It holds out 20% of the rows and prints the mean absolute error (in minutes) of the network next to the error of the simple formula. It uses a GPU automatically when one is available.

**Result** (8,277 held-out rows, trained on CPU):

| Model | Mean absolute error |
| --- | --- |
| Map shows "API KEY REQUIRED" | the old CARTO tile address now needs a key. Use the OpenStreetMap address in `initMap()` in `web/app.js` |
| Old behaviour after editing `app.js` | hard-refresh the browser (Ctrl+F5) |
| Top bar warns about fake data | `data/train.csv` was not found. Put the file in `data/` or set `DATA_PATH` |
| "Neural network is not available" | install PyTorch and run `python train_nn.py` |
| `pip install` is very slow or fails on torch | use `pip install -r requirements-deploy.txt` |
| Charts or map are blank | check your internet connection, because libraries load from a CDN |
| Port already in use | start with `uvicorn app:app --port 8001` |

#contributors
