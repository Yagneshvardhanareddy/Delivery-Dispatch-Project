"""
transportation.py - the Transportation Problem in pure NumPy.

Problem: m sources with supply s_i, n destinations with demand d_j and a unit cost c_ij.
Choose shipments x_ij >= 0 to minimise  sum(c_ij * x_ij)  without exceeding supplies and
while meeting demands.

Method used (same as in class):
  1. Balance the problem. If total supply != total demand, add a DUMMY destination or
     source with cost 0 that absorbs the difference.
  2. Vogel's Approximation Method (VAM) gives a very good starting solution:
        penalty of a row/column = (2nd smallest cost) - (smallest cost) among cells still open
        -> pick the row/column with the biggest penalty, and fill its cheapest cell.
  3. MODI (u-v method) checks optimality and improves the solution:
        find u_i, v_j with u_i + v_j = c_ij on every basic cell,
        opportunity cost of an empty cell = c_ij - u_i - v_j.
        If all are >= 0 the solution is optimal; otherwise push flow around the closed loop
        of the most negative cell (stepping-stone move) and repeat.
"""
import numpy as np

EPS = 1e-9


def balance_problem(supply, demand, cost):
    """Make total supply equal to total demand using a zero-cost dummy row or column."""
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    cost = np.asarray(cost, dtype=float)
    m, n = cost.shape
    diff = supply.sum() - demand.sum()
    dummy = None
    if diff > EPS:        # more supply than demand -> dummy destination (column)
        cost = np.hstack([cost, np.zeros((m, 1))])
        demand = np.append(demand, diff)
        dummy = "column"
    elif diff < -EPS:     # more demand than supply -> dummy source (row)
        cost = np.vstack([cost, np.zeros((1, n))])
        supply = np.append(supply, -diff)
        dummy = "row"
    return supply, demand, cost, dummy


def vogel_initial_solution(supply, demand, cost):
    """
    Vogel's Approximation Method on a BALANCED problem.
    Returns x (shipments) and a boolean mask `basic` of the m+n-1 basic cells
    (a basic cell can hold 0 when the problem is degenerate).
    """
    s, d = supply.copy(), demand.copy()
    m, n = cost.shape
    x = np.zeros((m, n))
    basic = np.zeros((m, n), dtype=bool)
    row_on = np.ones(m, dtype=bool)
    col_on = np.ones(n, dtype=bool)

    while row_on.any() and col_on.any():
        rows = np.flatnonzero(row_on)
        cols = np.flatnonzero(col_on)

        def penalty(values):
            values = np.sort(values)
            return values[1] - values[0] if len(values) > 1 else values[0]

        row_pen = np.array([penalty(cost[r, cols]) for r in rows])
        col_pen = np.array([penalty(cost[rows, c]) for c in cols])

        if row_pen.max() >= col_pen.max():                       # biggest penalty is a row
            r = rows[np.argmax(row_pen)]
            c = cols[np.argmin(cost[r, cols])]
        else:                                                    # biggest penalty is a column
            c = cols[np.argmax(col_pen)]
            r = rows[np.argmin(cost[rows, c])]

        q = min(s[r], d[c])                                      # ship as much as possible
        x[r, c] = q
        basic[r, c] = True
        s[r] -= q
        d[c] -= q

        # cross out exhausted lines. If BOTH are exhausted cross out only one, so that
        # we always end with m+n-1 basic cells (this keeps the MODI step solvable).
        if s[r] <= EPS and d[c] <= EPS:
            if row_on.sum() > 1:
                row_on[r] = False
            else:
                col_on[c] = False
        elif s[r] <= EPS:
            row_on[r] = False
        else:
            col_on[c] = False
    return x, basic


def _potentials(cost, basic):
    """Solve u_i + v_j = c_ij on the basic cells (start with u_0 = 0)."""
    m, n = cost.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    cells = list(zip(*np.nonzero(basic)))
    changed = True
    while changed:
        changed = False
        for i, j in cells:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]
                changed = True
            elif np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = cost[i, j] - v[j]
                changed = True
    if np.isnan(u).any() or np.isnan(v).any():
        raise RuntimeError("Basic cells are not connected - cannot compute u and v")
    return u, v


def _find_loop(basic, start):
    """
    Closed loop through `start` and basic cells that alternates horizontal / vertical moves.
    Cells at even positions get +theta, cells at odd positions get -theta.
    """
    m, n = basic.shape
    cells = set(zip(*np.nonzero(basic)))
    cells.add(start)

    def dfs(path, horizontal):
        i, j = path[-1]
        if horizontal:
            candidates = [(i, c) for c in range(n) if c != j and (i, c) in cells]
        else:
            candidates = [(r, j) for r in range(m) if r != i and (r, j) in cells]
        for cand in candidates:
            if cand == start and len(path) >= 4:
                return path
            if cand in path:
                continue
            found = dfs(path + [cand], not horizontal)
            if found:
                return found
        return None

    return dfs([start], True) or dfs([start], False)


def modi_optimize(x, basic, cost, max_iter=1000):
    """Improve a starting solution with the MODI method. Returns (x, basic, iterations)."""
    x = x.copy()
    basic = basic.copy()
    for iteration in range(max_iter):
        u, v = _potentials(cost, basic)
        opportunity = cost - u[:, None] - v[None, :]
        opportunity[basic] = 0.0
        i, j = np.unravel_index(np.argmin(opportunity), opportunity.shape)
        if opportunity[i, j] >= -EPS:                  # no negative cell -> optimal
            return x, basic, iteration

        loop = _find_loop(basic, (i, j))
        if loop is None:
            raise RuntimeError("No closed loop found")
        plus, minus = loop[0::2], loop[1::2]
        theta = min(x[r, c] for r, c in minus)         # largest shift that keeps x >= 0
        for r, c in plus:
            x[r, c] += theta
        for r, c in minus:
            x[r, c] -= theta
        leaving = next((r, c) for r, c in minus if x[r, c] <= EPS)
        x[leaving] = 0.0
        basic[i, j] = True
        basic[leaving] = False
    raise RuntimeError("MODI did not converge")


def solve_transportation(supply, demand, cost):
    """
    Solve a (possibly unbalanced) transportation problem.

    Returns a dict:
        flow          m x n shipments for the REAL sources/destinations
        total_cost    cost of the optimal plan
        initial_cost  cost of the Vogel starting solution (shows how good VAM already is)
        iterations    number of MODI improvement steps
        dummy         None, "row" or "column" (which dummy node was added)
    """
    m0, n0 = np.asarray(cost).shape
    s, d, c, dummy = balance_problem(supply, demand, cost)
    x0, basic = vogel_initial_solution(s, d, c)
    x, _, iterations = modi_optimize(x0, basic, c)
    return {
        "flow": x[:m0, :n0],
        "total_cost": float((x * c).sum()),
        "initial_cost": float((x0 * c).sum()),
        "iterations": int(iterations),
        "dummy": dummy,
    }
