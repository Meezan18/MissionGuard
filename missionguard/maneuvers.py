"""Candidate maneuver generation and relative-motion propagation.

Displacement of the primary after an impulsive burn is propagated with the
Clohessy-Wiltshire (Hill) equations, which are accurate for small burns on
near-circular orbits over a few revolutions.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .models import Candidate, Scenario

DV_LEVELS = (0.05, 0.1, 0.2, 0.4)          # m/s
LEAD_ORBITS = (0.5, 1.0, 2.0, 3.0)          # burn lead time before TCA, in orbits
DIRECTIONS = {
    "in-track": (np.array([0.0, 1.0, 0.0]), (+1, -1)),
    "radial": (np.array([1.0, 0.0, 0.0]), (+1, -1)),
    "cross-track": (np.array([0.0, 0.0, 1.0]), (+1, -1)),
}


def cw_displacement(dv_rtn_ms: np.ndarray, dt_s: float, n: float) -> np.ndarray:
    """Position offset (km) at time dt after an impulse dv (m/s) in RTN."""
    if dt_s <= 0:
        return np.zeros(3)
    vr, vt, vn = dv_rtn_ms / 1000.0  # km/s
    nt = n * dt_s
    s, c = np.sin(nt), np.cos(nt)
    x = vr / n * s + 2 * vt / n * (1 - c)
    y = 2 * vr / n * (c - 1) + vt / n * (4 * s - 3 * nt)
    z = vn / n * s
    return np.array([x, y, z])


def displacement_at(cand: Candidate, t_h: float, n: float) -> np.ndarray:
    if cand.burn_time is None:
        return np.zeros(3)
    return cw_displacement(cand.dv, (t_h - cand.burn_time) * 3600.0, n)


def generate_candidates(sc: Scenario) -> List[Candidate]:
    """No-action plus in-track / radial / cross-track burns at several lead times.

    Burns of the same type at different lead times are the 'time-shift'
    alternatives described in the paper.
    """
    cands = [Candidate(kind="no-action", dv=np.zeros(3), burn_time=None, label="No action")]
    tca = sc.primary.tca
    for lead in LEAD_ORBITS:
        tb = tca - lead * sc.period_h
        if tb <= 0.25:
            continue
        for kind, (axis, signs) in DIRECTIONS.items():
            for sign in signs:
                for dv in DV_LEVELS:
                    vec = axis * sign * dv
                    lbl = f"{kind} {'+' if sign > 0 else '-'}{dv:.2f} m/s @ T-{lead:g} orb"
                    cands.append(Candidate(kind=kind, dv=vec, burn_time=tb, label=lbl))
    return cands
