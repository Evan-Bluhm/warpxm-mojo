# Per-physics `.vtu` presets: the usual primitive variables for each
# system, on top of the generic writer in `vtu.mojo`.
#
#     dump_euler_frame(path, mesh, q, gamma)       rho, u, v, w, P
#     dump_imhd_frame(path, mesh, q, gamma)        rho, u, v, w, P, Bx, By,
#                                                  Bz, B2_over_2
#     dump_imhd_glm_frame(path, mesh, q, gamma)    the above plus psi
#     dump_maxwell_frame(path, mesh, q)            E_x, E_y, E_z, B_x, B_y, B_z,
#                                                  E_energy, B_energy
#
# `mesh` is a 2D or 3D `BlockMesh`; the field list is the same
# in both. For anything else call `write_vtu` with your own closure.

from std.collections import Array, List

from warpxm.apps.euler import EULER_NC
from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.apps.imhd_glm import MHD_GLM_NC, IDX_PSI
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.io.vtu import write_vtu
from warpxm.mesh.block import BlockMesh
from warpxm.variable.state import DGState


# Euler and MHD share the first five conserved slots (ρ, ρu, E), so the
# index constants imported from `apps.imhd` serve both.

@always_inline
def _euler_primitives(v: Array[Real, EULER_NC], gamma: Real) -> Array[Real, 5]:
    var inv_rho = Real(1.0) / v[IDX_RHO]
    var u = v[IDX_MX] * inv_rho
    var vy = v[IDX_MY] * inv_rho
    var w = v[IDX_MZ] * inv_rho
    var ke = Real(0.5) * v[IDX_RHO] * (u * u + vy * vy + w * w)
    return [v[IDX_RHO], u, vy, w, (gamma - Real(1.0)) * (v[IDX_E] - ke)]


@always_inline
def _mhd_primitives[NC: Int](v: Array[Real, NC], gamma: Real) -> Array[Real, 9]:
    var inv_rho = Real(1.0) / v[IDX_RHO]
    var u = v[IDX_MX] * inv_rho
    var vy = v[IDX_MY] * inv_rho
    var w = v[IDX_MZ] * inv_rho
    var ke = Real(0.5) * v[IDX_RHO] * (u * u + vy * vy + w * w)
    var mag = Real(0.5) * (
        v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
    )
    return [
        v[IDX_RHO], u, vy, w, (gamma - Real(1.0)) * (v[IDX_E] - ke - mag),
        v[IDX_BX], v[IDX_BY], v[IDX_BZ], mag,
    ]


def _euler_names() -> List[String]:
    return ["rho", "u", "v", "w", "P"]


def _mhd_names() -> List[String]:
    return ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2"]


@always_inline
def _maxwell_fields(v: Array[Real, MAXWELL_NC]) -> Array[Real, 8]:
    """(E, B) and the two energy densities |E|²/2, |B|²/2."""
    return [
        v[0], v[1], v[2], v[3], v[4], v[5],
        Real(0.5) * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]),
        Real(0.5) * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]),
    ]


def _maxwell_names() -> List[String]:
    return ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "E_energy", "B_energy"]


def dump_euler_frame[NDIM: Int, P: Int](
    path: String, mesh: BlockMesh[NDIM, P], mut q: DGState[NDIM, P, EULER_NC],
    gamma: Real,
) raises:
    """Write (ρ, u, v, w, P) of an Euler field."""
    def fields(v: Array[Real, EULER_NC]) {imm} -> Array[Real, 5]:
        return _euler_primitives(v, gamma)
    write_vtu(path, mesh, q, _euler_names(), fields)


def dump_imhd_frame[NDIM: Int, P: Int](
    path: String, mesh: BlockMesh[NDIM, P], mut q: DGState[NDIM, P, MHD_NC],
    gamma: Real,
) raises:
    """Write the primitive variables of an ideal-MHD field."""
    def fields(v: Array[Real, MHD_NC]) {imm} -> Array[Real, 9]:
        return _mhd_primitives[MHD_NC](v, gamma)
    write_vtu(path, mesh, q, _mhd_names(), fields)


@always_inline
def _glm_fields(v: Array[Real, MHD_GLM_NC], gamma: Real) -> Array[Real, 10]:
    var p = _mhd_primitives[MHD_GLM_NC](v, gamma)
    var f = Array[Real, 10](fill=Real(0.0))
    for k in range(9):
        f[k] = p[k]
    f[9] = v[IDX_PSI]
    return f^


def _glm_names() -> List[String]:
    var names = _mhd_names()
    names.append("psi")
    return names^


def dump_imhd_glm_frame[NDIM: Int, P: Int](
    path: String, mesh: BlockMesh[NDIM, P], mut q: DGState[NDIM, P, MHD_GLM_NC],
    gamma: Real,
) raises:
    """Write the primitive variables and ψ of a GLM-MHD field."""
    def fields(v: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, 10]:
        return _glm_fields(v, gamma)
    write_vtu(path, mesh, q, _glm_names(), fields)


def dump_maxwell_frame[NDIM: Int, P: Int](
    path: String, mesh: BlockMesh[NDIM, P], mut q: DGState[NDIM, P, MAXWELL_NC],
) raises:
    """Write (E, B) and their energy densities for a Maxwell field."""
    def fields(v: Array[Real, MAXWELL_NC]) -> Array[Real, 8]:
        return _maxwell_fields(v)
    write_vtu(path, mesh, q, _maxwell_names(), fields)
