"""Reproducible synthetic LEO conjunction scenarios.

Phase I uses synthetic scenarios because real spacecraft telemetry and mission
schedules are not public. Phase II replaces this module with real CDM / TLE data.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .models import Activity, Conjunction, Health, Scenario, Window

SAT_NAMES = ["ESEC-EO1", "ESEC-EO2", "ESEC-COM1", "ESEC-SCI1"]


def _cov(rng, s_min=0.05, s_max=0.5):
    s1, s2 = rng.uniform(s_min, s_max, 2)
    rho = rng.uniform(-0.4, 0.4)
    return np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])


def _conjunction(rng, oid: str, tca: float, miss_scale: float) -> Conjunction:
    theta = rng.uniform(0.3, 1.4)
    miss = rng.normal(0.0, miss_scale, 3)
    return Conjunction(object_id=oid, tca=tca, miss_rtn=miss, cov_b=_cov(rng),
                       hbr=rng.uniform(0.010, 0.020), rel_speed=rng.uniform(7.0, 14.5),
                       theta=theta)


def make_scenario(sid: int, seed: int = 2026) -> Scenario:
    rng = np.random.default_rng(seed * 1000 + sid)
    alt = rng.uniform(450, 800)
    horizon = 72.0
    tca = rng.uniform(18, 48)
    # primary miss scale chosen so Pc spans roughly 1e-6 .. 1e-2
    primary = _conjunction(rng, f"DEB-{rng.integers(40000, 59999)}", tca, rng.choice([0.08, 0.15, 0.3, 0.5]))
    secondaries = [
        _conjunction(rng, f"OBJ-{rng.integers(20000, 39999)}", tca + rng.uniform(2, 20), rng.uniform(0.6, 1.6))
        for _ in range(int(rng.integers(2, 5)))
    ]
    # ground-station passes: roughly every 1.5-6 h, 8-12 min long
    comm, t = [], rng.uniform(0, 2)
    while t < horizon:
        dur = rng.uniform(8, 12) / 60
        comm.append(Window(t, t + dur))
        t += rng.uniform(1.5, 6.0)
    # payload activities (imaging / downlink / experiments)
    acts, t = [], rng.uniform(0, 1)
    k = 0
    while t < horizon:
        dur = rng.uniform(5, 25) / 60
        acts.append(Activity(f"ACT-{k:02d}", Window(t, t + dur), int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))))
        t += rng.uniform(0.6, 3.0)
        k += 1
    health = Health(*np.round(rng.uniform([0.55, 0.6, 0.55, 0.7], 1.0), 2))
    return Scenario(scenario_id=sid, sat_name=str(rng.choice(SAT_NAMES)), altitude_km=float(alt),
                    epoch_label="T0", primary=primary, secondaries=secondaries, comm_windows=comm,
                    activities=acts, health=health, fuel_available=float(rng.uniform(0.6, 5.0)),
                    fuel_reserve=0.3, cdm_age_h=float(rng.uniform(1, 36)))


def make_scenarios(n: int = 48, seed: int = 2026) -> List[Scenario]:
    return [make_scenario(i, seed) for i in range(n)]
