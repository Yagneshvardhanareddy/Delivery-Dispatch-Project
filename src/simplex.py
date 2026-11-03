"""
simplex.py - Linear Programming with the Simplex method, in pure NumPy.

Solves        max (or min)  c . x
subject to    A_ub x <= b_ub
              A_ge x >= b_ge
              A_eq x  = b_eq
              x >= 0

Two ways to start from a feasible corner when ">=" or "=" rows exist:
  * "two_phase" : Phase I minimises the sum of artificial variables to find a feasible corner,
                  Phase II then optimises the real objective.
  * "big_m"     : artificial variables get a huge penalty M in the objective, one phase only.

The tableau method:
  * every row  a.x <= b gets a SLACK variable (+1)
  * every row  a.x >= b gets a SURPLUS variable (-1) and an ARTIFICIAL variable (+1)
  * every row  a.x  = b gets an ARTIFICIAL variable (+1)
  * repeat: pick the entering column (most negative reduced cost), pick the leaving row with
    the minimum-ratio test, pivot. Stop when no reduced cost is negative.
"""
import numpy as np

EPS = 1e-9


def _build_standard_form(c_min, A_ub, b_ub, A_ge, b_ge, A_eq, b_eq):
    """
    Add slack / surplus / artificial columns.
    Returns A (m x total_cols), b, the list of artificial column indexes and the starting basis.
    """
    n = len(c_min)
    rows = []                                   # (coefficients, rhs, sense)
    for A, b, sense in ((A_ub, b_ub, "<="), (A_ge, b_ge, ">="), (A_eq, b_eq, "=")):
        if A is None or len(A) == 0:
            continue
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float).ravel()
        for a_row, rhs in zip(A, b):
            if rhs < 0:                          # make every right-hand side >= 0
                a_row, rhs = -a_row, -rhs
                sense_row = {"<=": ">=", ">=": "<=", "=": "="}[sense]
            else:
                sense_row = sense
            rows.append((a_row, rhs, sense_row))

    m = len(rows)
    n_extra = sum(1 if s in ("<=", ">=") else 0 for _, _, s in rows)   # slack / surplus columns
    n_art = sum(1 if s in (">=", "=") else 0 for _, _, s in rows)
    total = n + n_extra + n_art
    A_full = np.zeros((m, total))
    b_full = np.zeros(m)
    basis = []
    art_cols = []
    extra_next, art_next = n, n + n_extra
    for r, (a_row, rhs, sense) in enumerate(rows):
        A_full[r, :n] = a_row
        b_full[r] = rhs
        if sense == "<=":
            A_full[r, extra_next] = 1.0          # slack becomes the starting basic variable
            basis.append(extra_next)
            extra_next += 1
        elif sense == ">=":
            A_full[r, extra_next] = -1.0         # surplus (cannot start in the basis)
            extra_next += 1
            A_full[r, art_next] = 1.0            # artificial starts in the basis
            basis.append(art_next)
            art_cols.append(art_next)
            art_next += 1
        else:
            A_full[r, art_next] = 1.0
            basis.append(art_next)
            art_cols.append(art_next)
            art_next += 1
    return A_full, b_full, art_cols, basis


def _pivot(T, row, col):
    """Gauss-Jordan pivot on T[row, col] (including the objective row)."""
    T[row] = T[row] / T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    T -= np.outer(factors, T[row])


def _price_out(T, cost_vector, basis):
    """Fill the last row with reduced costs: c_j - c_B * (column j), so basic columns are 0."""
    T[-1, :-1] = cost_vector
    T[-1, -1] = 0.0
    for r, b_col in enumerate(basis):
        T[-1] -= cost_vector[b_col] * T[r]


def _run_simplex(T, basis, allowed, max_iter):
    """Iterate the simplex method on tableau T (minimisation). Returns (status, iterations)."""
    m = T.shape[0] - 1
    stall = 0
    for it in range(max_iter):
        reduced = T[-1, :-1]
        candidates = np.flatnonzero((reduced < -EPS) & allowed)
        if candidates.size == 0:
            return "optimal", it
        # Dantzig rule (most negative). After many zero-progress steps switch to Bland's rule
        # (smallest index) which can never cycle.
        col = candidates[0] if stall > 50 else candidates[np.argmin(reduced[candidates])]

        column = T[:m, col]
        positive = column > EPS
        if not positive.any():
            return "unbounded", it
        ratios = np.full(m, np.inf)
        ratios[positive] = T[:m, -1][positive] / column[positive]
        best = ratios.min()
        tied = np.flatnonzero(ratios <= best + 1e-12)
        row = tied[np.argmin([basis[r] for r in tied])]   # Bland tie-break

        before = T[-1, -1]
        _pivot(T, row, col)
        basis[row] = col
        stall = stall + 1 if abs(T[-1, -1] - before) < EPS else 0
    return "iteration_limit", max_iter


def simplex(c, A_ub=None, b_ub=None, A_ge=None, b_ge=None, A_eq=None, b_eq=None,
            maximize=False, method="two_phase", max_iter=10000):
    """
    Solve a linear program.

    Returns a dict with keys:
        status     "optimal", "infeasible", "unbounded" or "iteration_limit"
        x          optimal values of the decision variables (None if not optimal)
        objective  optimal value of c.x in the ORIGINAL sense (max or min)
        iterations total simplex iterations
        method     "two_phase" or "big_m"
    """
    c = np.asarray(c, dtype=float).ravel()
    c_min = -c if maximize else c                 # internally we always minimise
    n = len(c)
    A, b, art_cols, basis = _build_standard_form(c_min, A_ub, b_ub, A_ge, b_ge, A_eq, b_eq)
    m, total = A.shape
    T = np.zeros((m + 1, total + 1))
    T[:m, :total] = A
    T[:m, -1] = b
    art = np.zeros(total, dtype=bool)
    art[art_cols] = True

    def result(status, x_full=None, iterations=0):
        if status != "optimal":
            return {"status": status, "x": None, "objective": None,"iterations": iterations, "method": method}
        assert x_full is not None
        x = np.clip(x_full[:n], 0.0, None)
        return {"status": "optimal", "x": x, "objective": float(c @ x),"iterations": iterations, "method": method}

    def solution_from(T_, basis_, size):
        x_full = np.zeros(size)
        for r, b_col in enumerate(basis_):
            x_full[b_col] = T_[r, -1]
        return x_full

    # ---------------------------------------------------------------- Big-M
    if method == "big_m":
        big_m = 1e6 * (1.0 + np.abs(c_min).max())
        cost_vector = np.zeros(total)
        cost_vector[:n] = c_min
        cost_vector[art] = big_m
        _price_out(T, cost_vector, basis)
        status, iters = _run_simplex(T, basis, np.ones(total, dtype=bool), max_iter)
        x_full = solution_from(T, basis, total)
        if status in ("optimal", "unbounded") and (x_full[art] > 1e-7).any():
            # an artificial variable is still positive: the real problem has no feasible point
            # (an "unbounded" signal while artificials remain is a side effect of infeasibility)
            return result("infeasible", iterations=iters)
        if status != "optimal":
            return result(status, iterations=iters)
        return result("optimal", x_full, iters)

    if method != "two_phase":
        raise ValueError("method must be 'two_phase' or 'big_m'")

    # ------------------------------------------------------- Two-Phase, Phase I
    iters = 0
    if art_cols:
        phase1_cost = np.zeros(total)
        phase1_cost[art] = 1.0
        _price_out(T, phase1_cost, basis)
        status, it1 = _run_simplex(T, basis, np.ones(total, dtype=bool), max_iter)
        iters += it1
        if status != "optimal" or -T[-1, -1] > 1e-7:     # artificials could not be driven to 0
            return result("infeasible", iterations=iters)

        # Artificial variables still in the basis (at value 0): pivot them out if possible
        keep_rows = list(range(m))
        for r in range(m):
            if art[basis[r]]:
                usable = np.flatnonzero((np.abs(T[r, :total]) > EPS) & ~art)
                if usable.size:
                    _pivot(T, r, usable[0])
                    basis[r] = usable[0]
                else:
                    keep_rows.remove(r)                  # redundant row, drop it
        T = np.vstack([T[keep_rows, :], T[-1:, :]])
        basis = [basis[r] for r in keep_rows]
        T = np.delete(T, np.flatnonzero(art), axis=1)    # remove artificial columns
        total_p2 = T.shape[1] - 1
    else:
        total_p2 = total

    # ------------------------------------------------------ Two-Phase, Phase II
    cost_vector = np.zeros(total_p2)
    cost_vector[:n] = c_min
    _price_out(T, cost_vector, basis)
    status, it2 = _run_simplex(T, basis, np.ones(total_p2, dtype=bool), max_iter)
    iters += it2
    if status != "optimal":
        return result(status, iterations=iters)
    return result("optimal", solution_from(T, basis, total_p2), iters)
