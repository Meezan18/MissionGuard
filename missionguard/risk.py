"""Collision-probability estimation in the encounter plane.

Pc is the integral of the 2-D Gaussian relative-position density over the
combined hard-body disk (the standard short-encounter assumption).
"""
from __future__ import annotations

import numpy as np

from .models import Conjunction


def bplane_basis(theta: float):
    """Return (e1, e2) spanning the B-plane for an encounter angle theta.

    Relative velocity is assumed to lie in the T-N plane, u = (0, sin, cos);
    the radial axis is then always in the B-plane.
    """
    u = np.array([0.0, np.sin(theta), np.cos(theta)])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def project_to_bplane(vec_rtn: np.ndarray, theta: float) -> np.ndarray:
    e1, e2 = bplane_basis(theta)
    return np.array([vec_rtn @ e1, vec_rtn @ e2])


def collision_probability(miss_b: np.ndarray, cov_b: np.ndarray, hbr: float,
                          n_r: int = 40, n_t: int = 72) -> float:
    """Numerically integrate the Gaussian over the hard-body disk (polar grid)."""
    inv = np.linalg.inv(cov_b)
    det = np.linalg.det(cov_b)
    norm = 1.0 / (2 * np.pi * np.sqrt(det))
    r = (np.arange(n_r) + 0.5) / n_r * hbr
    t = (np.arange(n_t) + 0.5) / n_t * 2 * np.pi
    rr, tt = np.meshgrid(r, t, indexing="ij")
    x = rr * np.cos(tt) - miss_b[0]
    y = rr * np.sin(tt) - miss_b[1]
    q = inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y
    dens = norm * np.exp(-0.5 * q)
    area = rr * (hbr / n_r) * (2 * np.pi / n_t)
    return float(np.clip((dens * area).sum(), 0.0, 1.0))


def conjunction_pc(conj: Conjunction, displacement_rtn: np.ndarray | None = None) -> tuple[float, float]:
    """Pc and miss distance (km) of a conjunction, optionally after a displacement of the primary."""
    miss = conj.miss_rtn.copy()
    if displacement_rtn is not None:
        miss = miss + displacement_rtn
    mb = project_to_bplane(miss, conj.theta)
    return collision_probability(mb, conj.cov_b, conj.hbr), float(np.linalg.norm(mb))


def risk_scale(pc: float, lo: float = 1e-7, hi: float = 1e-2) -> float:
    """Map Pc to a 0..1 risk score on a log scale."""
    if pc <= lo:
        return 0.0
    return float(np.clip((np.log10(pc) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0))
