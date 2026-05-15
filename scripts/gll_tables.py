#!/usr/bin/env python3
"""Regenerate the GLL tables embedded in src/warpxm/basis/nodal_basis_1d.mojo.

Usage:
    pixi run python scripts/gll_tables.py [max_P]

Prints node locations, quadrature weights, and the row-major differentiation
matrix D[i, j] = l_j'(ξ_i) for each polynomial order from 1 through `max_P`
(default 4). Copy-paste the output into nodal_basis_1d.mojo when adding new
orders.

For P ≤ 4 we use the closed-form GLL node locations:

    P = 1:  ξ ∈ {-1, +1}
    P = 2:  ξ ∈ {-1, 0, +1}
    P = 3:  ξ ∈ {-1, -√(1/5), +√(1/5), +1}
    P = 4:  ξ ∈ {-1, -√(3/7), 0, +√(3/7), +1}

For higher orders, swap in a Newton iteration on L_P'(ξ) = 0 — the rest of the
formula (weights via 2 / (P(P+1) L_P(ξ_i)²) and D from the Lagrange-derivative
identity) works unchanged.
"""

from __future__ import annotations
import sys
import numpy as np
from numpy.polynomial.legendre import Legendre


def gll_table(P: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nodes, weights, D) for GLL of polynomial order P (P + 1 nodes)."""
    if P == 1:
        x = np.array([-1.0, 1.0])
    elif P == 2:
        x = np.array([-1.0, 0.0, 1.0])
    elif P == 3:
        a = np.sqrt(1 / 5)
        x = np.array([-1.0, -a, a, 1.0])
    elif P == 4:
        a = np.sqrt(3 / 7)
        x = np.array([-1.0, -a, 0.0, a, 1.0])
    else:
        raise NotImplementedError(
            f"P = {P} requires a Newton iteration on L_P'(ξ) = 0; not implemented yet"
        )

    N = P + 1
    L_P = Legendre([0] * P + [1])

    # GLL weights.
    w = np.array([2.0 / (P * (P + 1) * L_P(xi) ** 2) for xi in x])

    # Differentiation matrix.
    D = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = L_P(x[i]) / (L_P(x[j]) * (x[i] - x[j]))
            else:
                if i == 0:
                    D[i, i] = -P * (P + 1) / 4
                elif i == N - 1:
                    D[i, i] = P * (P + 1) / 4
                else:
                    D[i, i] = 0.0
    return x, w, D


def main() -> int:
    max_P = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for P in range(1, max_P + 1):
        x, w, D = gll_table(P)
        print(f"=== P = {P} ===")
        print("nodes  :", list(map(float, x)))
        print("weights:", list(map(float, w)))
        print("D row-major:")
        for row in D:
            print("  " + ", ".join(repr(float(v)) for v in row) + ",")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
