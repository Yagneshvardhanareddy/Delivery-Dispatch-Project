"""
Verify our from-scratch solvers against SciPy, plus small hand-solvable examples.
Run:  pytest -q
"""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog

from src.hungarian import hungarian
from src.simplex import simplex
from src.transportation import solve_transportation


# ------------------------------------------------------------------ Hungarian
def test_hungarian_hand_example():
    # worked by hand: best choice is order0->rider1 (2), order1->rider0 (2), order2->rider2 (6)
    cost = np.array([[4, 2, 8], [2, 3, 7], [3, 1, 6]], dtype=float)
    rows, cols, total = hungarian(cost)
    assert total == 10.0
    assert list(cols[np.argsort(rows)]) == [1, 0, 2]


@pytest.mark.parametrize("seed", range(30))
def test_hungarian_matches_scipy_random(seed):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = rng.integers(1, 26, size=2)          # includes unbalanced shapes
    cost = rng.random((n_rows, n_cols)) * 100
    _, _, total = hungarian(cost)
    r, c = linear_sum_assignment(cost)
    assert total == pytest.approx(cost[r, c].sum())


def test_hungarian_unbalanced_uses_dummy():
    cost = np.array([[5.0, 9.0], [4.0, 1.0], [8.0, 3.0]])   # 3 orders, 2 riders
    rows, cols, total = hungarian(cost)
    assert len(rows) == 2                                   # one order stays unserved
    # by hand: serve order0 with rider0 (5) and order1 with rider1 (1) = 6; every other pair of
    # orders costs 7 or more, so order2 is the one left without a rider
    assert total == pytest.approx(6.0)
    r, c = linear_sum_assignment(cost)
    assert total == pytest.approx(cost[r, c].sum())


def test_hungarian_maximize():
    value = np.array([[3.0, 9.0], [8.0, 2.0]])
    _, _, total = hungarian(value, maximize=True)
    assert total == 17.0


# -------------------------------------------------------------- Transportation
def test_transportation_hand_example():
    # derived by hand: cost = 125 - 2a with a <= 10  ->  a = 10, total 105
    result = solve_transportation([10, 20], [15, 15], [[2, 3], [5, 4]])
    assert result["total_cost"] == pytest.approx(105.0)
    assert np.allclose(result["flow"], [[10, 0], [5, 15]])


def _linprog_transport(supply, demand, cost):
    supply, demand, cost = (np.asarray(v, dtype=float) for v in (supply, demand, cost))
    m, n = cost.shape
    rows_s, rows_d = [], []
    for i in range(m):
        row = np.zeros(m * n); row[i * n:(i + 1) * n] = 1; rows_s.append(row)
    for j in range(n):
        row = np.zeros(m * n); row[j::n] = 1; rows_d.append(row)
    if supply.sum() >= demand.sum():      # surplus supply: sources <=, destinations =
        kwargs = dict(A_ub=rows_s, b_ub=supply, A_eq=rows_d, b_eq=demand)
    else:                                 # surplus demand: sources =, destinations <=
        kwargs = dict(A_ub=rows_d, b_ub=demand, A_eq=rows_s, b_eq=supply)
    return linprog(cost.ravel(), bounds=(0, None), method="highs", **kwargs).fun


@pytest.mark.parametrize("seed", range(40))
def test_transportation_matches_linprog(seed):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(1, 7, size=2)
    supply = rng.integers(1, 30, m).astype(float)
    demand = rng.integers(1, 30, n).astype(float)
    if seed % 3 == 0:                      # make it balanced
        demand = np.diff(np.concatenate([[0], np.sort(rng.integers(0, int(supply.sum()) + 1, n - 1)),
                                         [int(supply.sum())]])).astype(float)
        if demand.sum() == 0:
            return
    cost = rng.integers(1, 30, (m, n)).astype(float)
    result = solve_transportation(supply, demand, cost)
    assert result["total_cost"] == pytest.approx(_linprog_transport(supply, demand, cost))
    assert (result["flow"] >= -1e-9).all()
    assert (result["flow"].sum(axis=1) <= supply + 1e-9).all()
    assert (result["flow"].sum(axis=0) <= demand + 1e-9).all()


# --------------------------------------------------------------------- Simplex
@pytest.mark.parametrize("method", ["two_phase", "big_m"])
def test_simplex_hand_maximisation(method):
    # max 3x + 5y  s.t.  x<=4, 2y<=12, 3x+2y<=18   -> (2, 6), value 36 (classic textbook LP)
    res = simplex([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True, method=method)
    assert res["status"] == "optimal"
    assert res["objective"] == pytest.approx(36.0)
    assert np.allclose(res["x"], [2, 6])


@pytest.mark.parametrize("method", ["two_phase", "big_m"])
def test_simplex_hand_minimisation_with_ge_rows(method):
    # min 2x + 3y  s.t.  x+y>=4, x+3y>=6   -> corner (3, 1), value 9
    res = simplex([2, 3], A_ge=[[1, 1], [1, 3]], b_ge=[4, 6], method=method)
    assert res["objective"] == pytest.approx(9.0)
    assert np.allclose(res["x"], [3, 1])


@pytest.mark.parametrize("method", ["two_phase", "big_m"])
def test_simplex_detects_infeasible_and_unbounded(method):
    assert simplex([1, 1], A_ub=[[1, 1]], b_ub=[1], A_ge=[[1, 1]], b_ge=[3], method=method)["status"] == "infeasible"
    assert simplex([1, 1], A_ge=[[1, -1]], b_ge=[0], maximize=True, method=method)["status"] == "unbounded"


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("method", ["two_phase", "big_m"])
def test_simplex_matches_linprog_random(seed, method):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    c = rng.integers(-5, 10, n).astype(float)
    A_ub = rng.integers(0, 6, (2, n)).astype(float); b_ub = rng.integers(5, 40, 2).astype(float)
    A_ge = rng.integers(0, 4, (1, n)).astype(float); b_ge = rng.integers(0, 10, 1).astype(float)
    ref = linprog(-c, A_ub=np.vstack([A_ub, -A_ge]), b_ub=np.concatenate([b_ub, -b_ge]),
                  bounds=(0, None), method="highs")
    res = simplex(c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, maximize=True, method=method)
    expected = {0: "optimal", 2: "infeasible", 3: "unbounded"}[ref.status]
    assert res["status"] == expected
    if expected == "optimal":
        assert res["objective"] == pytest.approx(-ref.fun, rel=1e-6, abs=1e-6)
