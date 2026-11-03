# Project summary: Optimal Food Delivery Dispatch

Course: Optimization Techniques. Team of 7. Dataset: Kaggle *Food Delivery Dataset* (Gaurav Malik), `train.csv`.

## 1. What the project does

A delivery platform has a batch of waiting orders and a pool of riders. Our program decides **which rider takes which order** so the total delivery time is as small as possible, and plans **where riders should wait** and **which orders to serve** to keep profit high. It uses three methods from the course, all written from scratch in NumPy and checked against SciPy:

| Step | Method | What it decides |
| --- | --- | --- |
| 1 | **Assignment problem**, Hungarian method | one rider for each order, minimum total time |
| 2 | **Transportation problem**, Vogel's approximation + MODI | how many riders to move from zone to zone before dispatching |
| 3 | **Linear programming**, Simplex (Two-Phase and Big-M) | which zone-to-zone deliveries to serve to maximise profit |

A web dashboard (FastAPI + HTML/JavaScript, Leaflet map, Plotly charts) shows the routes, the comparison with simple rules, the zone plan and the profit plan. A small neural network (PyTorch) is an optional extra.

## 2. Data

45,593 raw rows, 41,384 after cleaning (about 3,640 rows have all-zero coordinates and about 1,200 have missing traffic or weather). 22 cities (the city is read from the rider ID, because the `City` column is only the city type). Average delivery time 26.3 minutes, average distance 9.7 km.

Time model learned from the data: about 21 minutes fixed plus 0.54 minutes per km of delivery distance, with multipliers for traffic (Low 0.84, Medium 0.99, Jam 1.13, High 1.16), weather and vehicle.

## 3. Results (real data, estimated times)

Average over **110 batches** (all 22 cities x 5 seeds), 30 orders and 30 riders per batch. Reproduce with `python benchmark.py`.

| Strategy | Avg minutes per order | Faster than nearest rider | Late orders (of 30) | Profit per batch (Rs) |
| --- | --- | --- | --- | --- |
| Random assignment | 40.0 | -23.8% | 23.2 | 1,023 |
| Nearest rider first | 32.1 | baseline | 14.5 | 1,568 |
| Hungarian (optimal) | 30.8 | 3.8% | 13.6 | 1,667 |
| Hungarian + rider rebalancing | 30.0 | 5.9% | 13.0 | 1,720 |

- Hungarian was at least as good as nearest rider first in **100%** of the batches (it must be, it is optimal).
- Rebalancing riders first lowered the total time in **75%** of the batches, not all of them.
- With 60 orders and 60 riders (66 batches) the picture is the same: 3.8% and 6.2% faster than nearest rider first.
- The result is also steadier: the spread of average time between batches is 1.6 minutes with rebalancing against 9.0 for random assignment.

**Say this in the presentation:** the gain over a good greedy rule is real but modest (4-6%), because most of the delivery time is fixed preparation and handover time that no dispatch rule can change. The gain over random assignment is about 22%.

## 4. Assumptions and limits (state them yourselves before the examiner does)

1. All times are **model estimates**. The data records only the rider who really delivered each order, so there is no measured time for a different rider.
2. The data has **no rider locations**. Riders wait near real restaurants of the chosen city with about 1 km of random spread.
3. The rider's trip to the restaurant uses an **assumed pace of 3 min per km** (20 km/h). The fitted 0.54 min/km is not a riding speed, because distance explains little of the time in this data (correlation 0.32). The gains depend on this assumption.
4. Revenue per order (Rs 90), rider cost (Rs 1 per minute), the late penalty (Rs 1.5 per minute) and the 30-minute promise are **assumptions**; the data has no prices.
5. The model ignores `multiple_deliveries` and `Festival`, which change delivery times a lot in the real data.
6. With fewer riders than orders, Hungarian also chooses which orders to serve, so compare strategies with equal orders and riders.

## 5. Work split for 7 members

Every member owns one part, writes its tests, and presents it. Everyone must still be able to explain the whole pipeline in two minutes, because Q&A questions are not limited to your own part. Work on separate Git branches and merge with pull requests, so the commit history shows who did what.

| # | Role | Files owned | Presents and defends |
| --- | --- | --- | --- |
| 1 | Data and EDA (and the optional neural network) | `src/preprocess.py`, `src/eda.py`, `src/nn_cost_model.py`, `train_nn.py` | the dataset, every cleaning rule, the EDA charts, the neural-network result if built |
| 2 | Problem formulation and time model | `src/cost_matrix.py`, the maths of all three models | motivation, the formulation of each problem, the time model and every assumption |
| 3 | Assignment problem | `src/hungarian.py`, `src/baseline.py`, Hungarian tests | the Hungarian steps, dummy rows for unbalanced cases, the baselines |
| 4 | Transportation problem | `src/transportation.py`, KMeans zones and rider repositioning in `src/dispatch.py` | Vogel's method, MODI, degeneracy, dummy nodes, why zones |
| 5 | Linear programming | `src/simplex.py`, `src/profit_lp.py`, Simplex tests | the profit model, Two-Phase against Big-M, infeasible and unbounded cases |
| 6 | Web dashboard | `app.py`, `web/index.html`, `web/style.css`, `web/app.js`, API tests | the architecture, the live demo, how the API connects to the solver |
| 7 | Integration, testing and delivery | `src/dispatch.py`, `benchmark.py`, `run_demo.py`, `README.md`, GitHub repository, the PPT | the pipeline end to end, the benchmark results, the SciPy verification, the contribution table |

**Contribution table for the report** (replace the names):

| Member | Contribution |
| --- | --- |
| Member 1 | cleaned the data, EDA charts, optional neural network |
| Member 2 | formulation, time model, assumptions |
| Member 3 | Hungarian method and baselines |
| Member 4 | Transportation (VAM + MODI), zones |
| Member 5 | Simplex (Two-Phase, Big-M), profit LP |
| Member 6 | web dashboard and API |
| Member 7 | integration, tests, benchmark, GitHub, presentation |

## 6. Presentation outline (about 15 minutes)

| Slide | Content | Speaker |
| --- | --- | --- |
| 1 | Title, team and roles | 7 |
| 2 | Problem and motivation: dispatch decides delivery time and profit | 2 |
| 3 | Dataset and cleaning (45,593 to 41,384 rows, 22 cities) | 1 |
| 4 | What drives delivery time (EDA charts) and the learned time model | 1, 2 |
| 5 | Formulation: assignment, transportation and profit LP in symbols | 2 |
| 6 | Hungarian method with a small worked example | 3 |
| 7 | Transportation: zones, Vogel's method, MODI | 4 |
| 8 | Profit LP and Simplex, Two-Phase against Big-M | 5 |
| 9 | System architecture: data, solvers, API, dashboard | 6 |
| 10 | Live demo: run a batch, show routes, tables and the zone plan | 6 |
| 11 | Results: the benchmark table and chart (110 batches) | 7 |
| 12 | Assumptions and limits | 2 |
| 13 | Optional: neural network result, only if trained, honest either way | 1 |
| 14 | Contribution of each member | 7 |
| 15 | Questions | all |

## 7. Questions to expect

- **Why Hungarian and not a greedy rule?** Greedy decides order by order and can leave later orders with bad riders. Hungarian is optimal for the total, in O(n^3), and it never does worse (100% of our batches).
- **Why is the gain only 4-6%?** Most of the time is fixed preparation and handover. Dispatch only changes the riding part. Be ready to say how it depends on the assumed 3 min per km.
- **Is the data real?** The orders, locations, traffic and weather are real. Rider positions, prices and the pickup pace are assumptions.
- **How do you know your solvers are right?** 180 tests: hand-solved examples and comparison with SciPy on random problems, including unbalanced, degenerate, infeasible and unbounded cases.
- **Two-Phase or Big-M?** Both are implemented and give the same answers. Big-M needs a large constant that can cause rounding trouble; Two-Phase avoids it.
- **Are the solutions integers?** Assignment and Transportation problems have integer optimal solutions when the inputs are integers. The profit LP is a zone-level plan, so fractional values are expected flows.
- **What if there are more orders than riders?** Dummy riders are added, and the orders given to dummies are not served in that batch.
- **What would you improve?** Real rider GPS positions, multiple deliveries per rider, festival effects, orders arriving over time in short batches, and real prices.

## 8. Before you submit

- Run `pytest -q` (180 tests) and `python benchmark.py` on your own machine.
- Do not commit `data/train.csv` (it is in `.gitignore`); say in the README where to download it.
- Put screenshots of the dashboard in the PPT and in `docs/`.
- Train the neural network only if there is time: `pip install torch` then `python train_nn.py`. If it is not better than the formula, say so.
