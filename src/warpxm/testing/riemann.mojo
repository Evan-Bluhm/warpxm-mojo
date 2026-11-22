# Exact solution of the 1D Euler Riemann problem for an ideal gas.
#
# Reference solution for the shock-tube tests (Sod, Lax, Einfeldt 1-2-3).
# Follows Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
# 3rd ed., §4.2–4.5: the star pressure is the root of
# `f_L(p) + f_R(p) + u_R − u_L = 0`, and the self-similar solution is then
# sampled at `ξ = (x − x₀) / t`.
#
#     var exact = ExactRiemann(GAMMA, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
#     var w = exact.sample((x - 0.5) / t)        # (ρ, u, p)

from std.collections import Array

from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real


def _wave_fn(gamma: Real, p: Real, rho_k: Real, p_k: Real) -> Real:
    """Toro's f_K(p): velocity change across the K-wave for star pressure
    `p` (shock branch if p > p_k, rarefaction branch otherwise)."""
    if p > p_k:
        var a = Real(2.0) / ((gamma + Real(1.0)) * rho_k)
        var b = (gamma - Real(1.0)) / (gamma + Real(1.0)) * p_k
        return (p - p_k) * sqrt(a / (p + b))
    var c_k = sqrt(gamma * p_k / rho_k)
    var ex = (gamma - Real(1.0)) / (Real(2.0) * gamma)
    return Real(2.0) * c_k / (gamma - Real(1.0)) * ((p / p_k) ** ex - Real(1.0))


def _sample_left(
    gamma: Real,
    rho_k: Real, u_k: Real, p_k: Real, p_s: Real, u_s: Real, xi: Real,
) -> Array[Real, 3]:
    """(ρ, u, p) at similarity coordinate `xi` on the left of the contact
    (xi < u*). The right side follows by mirror symmetry."""
    var w = Array[Real, 3](fill=Real(0.0))
    var c_k = sqrt(gamma * p_k / rho_k)
    var gm = (gamma - Real(1.0)) / (gamma + Real(1.0))
    w[0] = rho_k; w[1] = u_k; w[2] = p_k
    if p_s > p_k:
        # Left shock.
        var s = u_k - c_k * sqrt(
            (gamma + Real(1.0)) / (Real(2.0) * gamma) * p_s / p_k
            + (gamma - Real(1.0)) / (Real(2.0) * gamma)
        )
        if xi > s:
            w[0] = rho_k * (p_s / p_k + gm) / (gm * p_s / p_k + Real(1.0))
            w[1] = u_s; w[2] = p_s
    else:
        # Left rarefaction: head, fan, tail.
        var c_s = c_k * (p_s / p_k) ** ((gamma - Real(1.0)) / (Real(2.0) * gamma))
        if xi > u_s - c_s:
            w[0] = rho_k * (p_s / p_k) ** (Real(1.0) / gamma)
            w[1] = u_s; w[2] = p_s
        elif xi > u_k - c_k:
            var c = Real(2.0) / (gamma + Real(1.0)) * (
                c_k + Real(0.5) * (gamma - Real(1.0)) * (u_k - xi)
            )
            w[0] = rho_k * (c / c_k) ** (Real(2.0) / (gamma - Real(1.0)))
            w[1] = Real(2.0) / (gamma + Real(1.0)) * (
                c_k + Real(0.5) * (gamma - Real(1.0)) * u_k + xi
            )
            w[2] = p_k * (c / c_k) ** (Real(2.0) * gamma / (gamma - Real(1.0)))
    return w^


struct ExactRiemann(Copyable, Movable):
    """Exact Riemann solution for left / right primitive states (ρ, u, p)."""

    var gamma: Real
    var rho_l: Real
    var u_l: Real
    var p_l: Real
    var rho_r: Real
    var u_r: Real
    var p_r: Real
    var p_star: Real
    """Pressure between the two nonlinear waves."""
    var u_star: Real
    """Velocity of the contact."""

    def __init__(
        out self, gamma: Real,
        rho_l: Real, u_l: Real, p_l: Real,
        rho_r: Real, u_r: Real, p_r: Real,
    ):
        self.gamma = gamma
        self.rho_l = rho_l; self.u_l = u_l; self.p_l = p_l
        self.rho_r = rho_r; self.u_r = u_r; self.p_r = p_r
        # f_L + f_R + Δu is monotone in p, so bisection is safe.
        var lo: Real = 0.0
        var hi: Real = Real(10.0) * (p_l + p_r)
        for _ in range(100):
            var mid = Real(0.5) * (lo + hi)
            var f = (
                _wave_fn(gamma, mid, rho_l, p_l)
                + _wave_fn(gamma, mid, rho_r, p_r) + u_r - u_l
            )
            if f > Real(0.0):
                hi = mid
            else:
                lo = mid
        self.p_star = Real(0.5) * (lo + hi)
        self.u_star = Real(0.5) * (u_l + u_r) + Real(0.5) * (
            _wave_fn(gamma, self.p_star, rho_r, p_r)
            - _wave_fn(gamma, self.p_star, rho_l, p_l)
        )

    def sample(self, xi: Real) -> Array[Real, 3]:
        """(ρ, u, p) at similarity coordinate `xi = (x − x₀) / t`."""
        if xi < self.u_star:
            return _sample_left(
                self.gamma, self.rho_l, self.u_l, self.p_l,
                self.p_star, self.u_star, xi,
            )
        var m = _sample_left(
            self.gamma, self.rho_r, -self.u_r, self.p_r,
            self.p_star, -self.u_star, -xi,
        )
        m[1] = -m[1]
        return m^
