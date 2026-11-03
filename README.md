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

## 1. Quick start

You need Python 3.9 or newer.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the packages
pip install -r requirements.txt

# 3. put the Kaggle file here:  data/train.csv

# 4. start the dashboard
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> in your browser.

Only `train.csv` is needed. `test.csv` and `Sample_Submission.csv` do not contain the delivery time, so the project does not use them.

If you do not have the Kaggle file yet, the app still starts. It uses fake data in the same format and shows a warning in the top bar. This is handy for trying the interface.

> `requirements.txt` includes PyTorch, which is large and only needed for the optional neural network. If you do not need it, install the lighter list instead: `pip install -r requirements-deploy.txt`.

---

## 2. Ways to run it

### Web dashboard

```bash
uvicorn app:app --reload
```

The page loads fonts, Leaflet, Plotly and the map tiles from the internet, so you need a connection the first time. There is no build step.

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

Environment variables (all optional):

| Variable | Meaning |
| --- | --- |
| `DATA_PATH` | path to the Kaggle csv (default `data/train.csv`) |
| `USE_SYNTHETIC=1` | ignore the csv and use fake data |

### Command line demo

```bash
python run_demo.py --data data/train.csv --city BANG --orders 30 --riders 30 --eda
python run_demo.py --synthetic
```

Useful options:

| Option | What it does |
| --- | --- |
| `--city BANG` | pick a city by its code (the program prints the list it found) |
| `--orders 30 --riders 30` | batch size |
| `--zones 6` | number of city zones |
| `--lp-method big_m` | use Big-M instead of Two-Phase for the profit plan |
| `--no-rebalance` | skip the rider repositioning step |
| `--eda` | also save the exploratory charts as PNG files |
| `--nn` | use the optional neural network time model |
| `--synthetic` | use fake data |

### Benchmark

```bash
python benchmark.py
```

See [Benchmarks](#5-benchmarks).

### Optional neural network

```bash
pip install torch
python train_nn.py
```

This trains a small time-prediction model and saves it in `models/`. After that, the dashboard enables the "Use neural network" option. It is only worth keeping if it beats the simple formula on held-out rows (see [Assumptions and limits](#10-assumptions-and-limits)).

---

## 3. The web dashboard

The dashboard has three pages.

| Page | What you see |
| --- | --- |
| **Dispatch** | batch settings, a map of every rider → restaurant → customer route, a comparison table and chart of the strategies, the zone repositioning plan, and the profit plan |
| **Dataset** | cleaned-data statistics, delivery time against distance, traffic, weather and hour of day, orders per city, and the learned time model |
| **How it works** | the three algorithms explained step by step, with textbook examples solved live by the project's own code |

### Reading the map

| Symbol | Meaning |
| --- | --- |
| Blue dot | rider |
| Orange dot | restaurant |
| Black dot | customer |
| Dashed circle | the centre of a city zone |
| Blue dashed line | the rider travelling to the restaurant (pickup leg) |
| Orange solid line | the restaurant to the customer (delivery leg) |

The lines are straight, not real roads, because the data has coordinates but no road network. Hover over a route to highlight it. Untick **Show routes on the map** in the settings panel to hide all lines and see only the points.

### Batch settings

- **City:** riders only work inside one city.
- **Orders in the batch / Riders available:** if there are fewer riders than orders, some orders stay unserved.
- **City zones:** how many KMeans zones to cut the city into.
- **Random seed:** the same seed always gives the same batch. Change it to draw a new one.
- **Move riders between zones first:** turns the Transportation step on or off.
- **Profit LP method:** Two-Phase or Big-M.

---

## 4. How it works

### 4.1 Cleaning

The raw file has 45,593 rows. After cleaning, 41,384 remain. The cleaning step:

- removes the `(min) ` text from `Time_taken(min)`,
- removes the `conditions ` prefix from the weather,
- turns the text `NaN` into real missing values,
- fixes negative coordinates,
- drops points outside India or far from their city (about 3,640 rows have all-zero coordinates),
- keeps ratings between 1 and 5,
- computes the Haversine distance between restaurant and customer.

The `City` column in the file is the city type, not the city. The real city code is read from `Delivery_person_ID`, which gives 22 cities.

### 4.2 The delivery time model

The model is learned from the data, not hard-coded:

```
time = fixed + minutes_per_km x distance
```

on top of which come multipliers for traffic, weather and vehicle. Each multiplier is `actual mean time / time predicted from distance alone`.

On the real file this gives about **21 minutes fixed** and **0.54 minutes per km**. The traffic multipliers are Low 0.84, Medium 0.99, Jam 1.13 and High 1.16.

### 4.3 The cost matrix

For every order *i* and rider *j*:

```
cost[i, j] = fixed
           + ( dist(rider j -> restaurant i) x pickup_pace
             + dist(restaurant i -> customer i) x minutes_per_km )
             x traffic x weather x vehicle
```

The pickup pace (3 minutes per km) is an assumption, explained in section 10.

### 4.4 Hungarian method (assignment)

Finds the one-to-one assignment with the lowest total cost. It works by row and column reduction, covering zeros with lines, and adjusting until a full assignment exists. If there are more orders than riders, dummy riders are added. The orders given to dummy riders are the ones left unserved.

### 4.5 Vogel + MODI (transportation)

KMeans cuts the city into zones. Supply is the number of riders in each zone, demand is the number of orders, and the cost is the distance between zone centres. Vogel's approximation builds a good starting plan and MODI improves it to the optimum. Riders the plan tells to move are repositioned, and then the Hungarian method runs again on the new positions. Dummy nodes handle unbalanced cases.

### 4.6 Simplex (profit plan)

A zone-level linear program maximises:

```
revenue - rider cost - late penalty
```

subject to rider capacity, a maximum delivery time, and a minimum service share per zone. The "at least" rows need an artificial start, which is why both **Two-Phase** and **Big-M** are implemented. They give the same answers.

### 4.7 Comparison

Four strategies are compared on the same batch:

1. Random assignment (average of 200 draws)
2. Nearest rider first
3. Hungarian
4. Hungarian with rider rebalancing

---

## 5. Benchmarks

One batch can be lucky, so the project reports averages over many cities and random seeds.

### Run it

```bash
python benchmark.py                                 # all cities, 5 seeds, 30 orders and 30 riders
python benchmark.py --orders 50 --riders 50 --seeds 10
```

Every run is saved to `outputs/benchmark.csv`, so you can open it in Excel or pandas.

### Results on the real data

Average over **110 batches** (22 cities x 5 seeds), 30 orders and 30 riders per batch. Times are model estimates.

| Strategy | Avg minutes per order | Faster than nearest rider | Late orders (of 30) | Profit per batch (Rs) |
| --- | --- | --- | --- | --- |
| Random assignment | 40.0 | -23.8% | 23.2 | 1,023 |
| Nearest rider first | 32.1 | baseline | 14.5 | 1,568 |
| Hungarian | 30.8 | 3.8% | 13.6 | 1,667 |
| Hungarian + rebalancing | 30.0 | 5.9% | 13.0 | 1,720 |

What the numbers say:

- Hungarian was at least as good as nearest rider first in **100%** of batches. It has to be, because it is optimal for the total time.
- Rebalancing riders first lowered the total time in **75%** of batches, not all of them.
- With 60 orders and 60 riders (66 batches) the picture is the same: about 3.8% and 6.2% faster than nearest rider first.
- The results are steadier too. The spread of average time between batches is 1.6 minutes with rebalancing, against 9.0 minutes for random assignment.
- A single batch can look better than the average. For example, the default JAP batch in the dashboard (seed 42) is 12.1% faster than nearest rider first. Quote the averages, not one good batch.

### Why the gain is modest

The gain over a good greedy rule is about 4 to 6%, and about 22% over random assignment. Most of a delivery's time is fixed preparation and handover, which no dispatch rule can change. Dispatch only changes the riding part.

### Tip for fair comparisons

Compare strategies with **equal orders and riders**. With fewer riders than orders, Hungarian also chooses *which* orders to serve while the baselines do not, so the averages are not directly comparable.

---

## 6. Project structure

```
app.py                   FastAPI server: /api/status, /api/eda, /api/algorithms, /api/dispatch
run_demo.py              command line demo
benchmark.py             averages over all cities and several seeds
train_nn.py              optional: trains and saves the neural network
requirements.txt         all packages (includes PyTorch)
requirements-deploy.txt  light list for hosting (no PyTorch, no plotting, no tests)

data/train.csv           the Kaggle file (not committed to git)
models/                  saved neural network (optional)
outputs/                 benchmark results and charts

src/preprocess.py        cleaning and Haversine distance
src/eda.py               numbers and charts for distance, traffic and weather vs time
src/cost_matrix.py       learns the time model, builds the order x rider cost matrix
src/hungarian.py         Hungarian method
src/transportation.py    Vogel's approximation + MODI
src/simplex.py           Simplex tableau, Two-Phase and Big-M
src/profit_lp.py         zone-level profit LP
src/baseline.py          random and nearest-rider-first rules
src/dispatch.py          run_dispatch(): the whole pipeline, returns a JSON-friendly dict
src/synthetic.py         fake data in the same format as the Kaggle file
src/nn_cost_model.py     optional PyTorch time model

web/index.html           the three pages
web/style.css            design and components
web/app.js               API calls, Leaflet map, Plotly charts

tests/                   unit tests
docs/                    project notes and results
```

### API

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/status` | data source, cities, learned time model, whether the neural network is available |
| GET | `/api/eda` | numbers behind the dataset charts |
| GET | `/api/algorithms` | small textbook examples solved live |
| POST | `/api/dispatch` | runs one full dispatch (body = `DispatchConfig` fields) |
| GET | `/docs` | interactive API documentation |

---

## 7. Testing

```bash
pytest -q
```

The tests do not need the Kaggle file, because they can use the fake data. They cover:

- hand-solved examples for each method,
- comparison with SciPy on random problems (Hungarian, Transportation and Simplex),
- unbalanced, degenerate, infeasible and unbounded cases,
- the whole pipeline on fake data,
- the API endpoints.

SciPy is used **only** in the tests, to prove the from-scratch solvers give the same answers.

---

## 8. Hosting it online

The app is a single FastAPI server that also serves the web page, so one service is enough. These steps use Render's free plan.

1. Put the project on GitHub, with `app.py` at the root of the repository.
   - `.gitignore` excludes `data/*.csv`. If `train.csv` is under about 100 MB you can remove that line and commit it. Otherwise use fake data (step 3).
2. On render.com choose **New → Web Service** and select the repository.
   - Runtime: Python 3
   - Build command: `pip install -r requirements-deploy.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
3. If you did not commit the csv, add the environment variable `USE_SYNTHETIC` = `1`.
4. Deploy. You get a link like `https://your-app.onrender.com`.

Things to know:

- The free plan sleeps when idle, so the first visit after a while can take 30 to 60 seconds.
- Without PyTorch the neural network option stays disabled. Everything else works.
- The map uses OpenStreetMap tiles. That is fine for a project or demo. For heavy public traffic, use a keyed provider such as MapTiler or Stadia Maps.

---

## 9. Settings you can change

| Setting | Where | Default |
| --- | --- | --- |
| Pickup pace (min per km for the rider's trip to the restaurant) | `PICKUP_MIN_PER_KM` in `src/cost_matrix.py` | 3 |
| Revenue per order | `DispatchConfig` in `src/dispatch.py` | 90 |
| Rider cost per minute | `DispatchConfig` | 1 |
| Late penalty per minute | `DispatchConfig` | 1.5 |
| Delivery promise | `DispatchConfig` | 30 minutes |
| Map tiles | `initMap()` in `web/app.js` | OpenStreetMap |

After changing the pickup pace or the money settings, run `python benchmark.py` again to see how the results depend on them.

---

## 10. Assumptions and limits

Please read these before quoting any number from this project.

- **All times are model estimates.** The data records only the rider who really delivered each order. There is no measured time for "what if a different rider had taken it".
- **There are no live rider locations in the data.** Riders are placed near real restaurant locations of the chosen city, with about 1 km of random spread.
- **The pickup pace is an assumption.** Distance explains little of the delivery time in this data (correlation 0.32), so the fitted 0.54 min/km is not a riding speed. For the rider's trip to the restaurant the project assumes 3 minutes per km (20 km/h). The gains depend on this number.
- **Prices are assumptions.** Revenue per order (90), rider cost (1 per minute), late penalty (1.5 per minute) and the 30-minute promise are not in the dataset.
- **Some factors are ignored.** `multiple_deliveries` and `Festival` change delivery times a lot in the data (festival orders take about 20 minutes longer), but they are not part of the batch simulation.
- **Hungarian minimises total time, not late orders.** On some batches it can have slightly more late orders than the greedy rule.
- **Empty zones are normal.** A KMeans zone can end up with no riders and no orders, because zones are built around customer points only.
- **The neural network** is judged only by its error on held-out real rows compared with the formula. If it is not better, do not use it.
- **The profit plan** works at zone level, so fractional flows in its answer are expected.

---

## 11. Ideas for improvement

Realism
- Use real rider GPS positions instead of simulated ones.
- Let one rider carry more than one order (`multiple_deliveries`).
- Add festival and rating effects to the time model.
- Replace straight lines with real road routes (OSRM or a similar routing service).

Operations
- Handle orders arriving over time, in short rolling batches, instead of one fixed batch.
- Add rider shifts, breaks and vehicle capacity.
- Add restaurant preparation time per order.
- Minimise the number of late orders, not only the total time.

Models and speed
- Compare more time models (gradient boosting, for example) with the simple formula.
- Use real prices and tips in the profit plan.
- Cache cost matrices for big batches, and test how far the Hungarian method scales (it is O(n³)).

Product
- Export the dispatch plan as CSV.
- Add a button to save and compare several runs side by side.
- Add a login and a simple database if it is hosted for real use.

---

## 12. Troubleshooting

| Problem | Fix |
| --- | --- |
| Map shows "API KEY REQUIRED" | the old CARTO tile address now needs a key. Use the OpenStreetMap address in `initMap()` in `web/app.js` |
| Old behaviour after editing `app.js` | hard-refresh the browser (Ctrl+F5) |
| Top bar warns about fake data | `data/train.csv` was not found. Put the file in `data/` or set `DATA_PATH` |
| "Neural network is not available" | install PyTorch and run `python train_nn.py` |
| `pip install` is very slow or fails on torch | use `pip install -r requirements-deploy.txt` |
| Charts or map are blank | check your internet connection, because libraries load from a CDN |
| Port already in use | start with `uvicorn app:app --port 8001` |