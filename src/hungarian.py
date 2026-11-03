"""
hungarian.py - the Hungarian method for the Assignment Problem, in pure NumPy.

Problem: n workers (riders) and n jobs (orders); cost[i, j] = cost if job i goes to worker j.
Find a one-to-one assignment with minimum total cost.

Textbook steps (this file follows them exactly):
  1. Row reduction:    subtract the smallest number of every row from that row.
  2. Column reduction: subtract the smallest number of every column from that column.
  3. Cover all zeros with the MINIMUM number of lines (rows/columns).
        - If that number equals n  -> an optimal assignment exists among the zeros. DONE.
  4. Otherwise let delta = smallest UNCOVERED number.
        subtract delta from every uncovered number,
        add delta to every number covered by two lines,
        and go back to step 3.

Unbalanced problems (more orders than riders, or the opposite):
  we add dummy rows / columns filled with zeros until the matrix is square.
  An order that gets a dummy rider is simply "not served in this batch".
"""
import numpy as np


def _max_zero_matching(zero_mask):
    """
    Pick as many zeros as possible so that no two share a row or column
    (a maximum bipartite matching, found with simple augmenting paths).
    Returns match_col_to_row: for each column the row matched to it, or -1.
    """
    n = zero_mask.shape[0]
    match_col_to_row = -np.ones(n, dtype=int)

    def try_row(r, seen):
        for c in np.flatnonzero(zero_mask[r]):
            if not seen[c]:
                seen[c] = True
                # column free, or the row that owns it can move somewhere else
                if match_col_to_row[c] == -1 or try_row(match_col_to_row[c], seen):
                    match_col_to_row[c] = r
                    return True
        return False

    for r in range(n):
        try_row(r, np.zeros(n, dtype=bool))
    return match_col_to_row


def _min_line_cover(zero_mask, match_col_to_row):
    """
    Minimum set of lines covering every zero (Konig's theorem).
      1. mark every row that has no matched zero
      2. mark every column that has a zero in a marked row
      3. mark every row that is matched to a marked column
      repeat 2-3 until nothing changes.
    Lines = UNmarked rows + marked columns.
    """
    n = zero_mask.shape[0]
    matched_rows = np.zeros(n, dtype=bool)
    matched_rows[match_col_to_row[match_col_to_row >= 0]] = True

    marked_rows = ~matched_rows
    marked_cols = np.zeros(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for r in np.flatnonzero(marked_rows):
            for c in np.flatnonzero(zero_mask[r]):
                if not marked_cols[c]:
                    marked_cols[c] = True
                    changed = True
        for c in np.flatnonzero(marked_cols):
            r = match_col_to_row[c]
            if r >= 0 and not marked_rows[r]:
                marked_rows[r] = True
                changed = True
    return ~marked_rows, marked_cols   # covered_rows, covered_cols


def hungarian(cost, maximize=False):
    """
    Solve the assignment problem.

    cost     : 2-D array (n_rows x n_cols), may be rectangular.
    maximize : True to maximise total value instead of minimising cost.

    Returns (rows, cols, total) where rows[k] is assigned to cols[k] and
    total is the sum of the ORIGINAL cost entries of the chosen pairs.
    Only real pairs are returned (dummy rows/columns are dropped).
    """
    original = np.asarray(cost, dtype=float)
    if original.ndim != 2 or original.size == 0:
        raise ValueError("cost must be a non-empty 2-D array")
    n_rows, n_cols = original.shape
    n = max(n_rows, n_cols)

    # maximisation -> minimisation: replace every value v by (largest value - v)
    work = (original.max() - original) if maximize else original

    # pad with dummy rows/columns of zeros to make the matrix square
    M = np.zeros((n, n))
    M[:n_rows, :n_cols] = work

    eps = 1e-9 * max(1.0, np.abs(M).max())

    # Step 1 and 2: row and column reduction
    M = M - M.min(axis=1, keepdims=True)
    M = M - M.min(axis=0, keepdims=True)

    for _ in range(10 * n * n + 100):                    # safety limit, never reached in practice
        zero = M <= eps
        match = _max_zero_matching(zero)
        if (match >= 0).sum() == n:                      # n independent zeros -> optimal
            break
        # Step 3 and 4: cover zeros with minimum lines, then create new zeros
        covered_rows, covered_cols = _min_line_cover(zero, match)
        uncovered = (~covered_rows)[:, None] & (~covered_cols)[None, :]
        twice = covered_rows[:, None] & covered_cols[None, :]
        delta = M[uncovered].min()
        M[uncovered] -= delta
        M[twice] += delta
    else:
        raise RuntimeError("Hungarian method did not converge")

    pairs = [(r, c) for c, r in enumerate(match) if r < n_rows and c < n_cols]
    pairs.sort()
    rows = np.array([p[0] for p in pairs], dtype=int)
    cols = np.array([p[1] for p in pairs], dtype=int)
    total = float(original[rows, cols].sum())
    return rows, cols, total
