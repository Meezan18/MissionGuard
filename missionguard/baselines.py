"""Baseline decision rules used for comparison (paper Section VI)."""
from __future__ import annotations

from .decision import PC_MAX, PC_SCREEN, evaluate_candidate, utility
from .maneuvers import generate_candidates
from .models import Candidate, Scenario

RF_WEIGHTS = {"S": 0.7, "F": 0.3, "M": 0.0, "O": 0.0, "Q": 0.0, "H": 0.0}


def _physical(c: Candidate) -> bool:
    """Every method must respect propellant and uplink limits."""
    return "Insufficient propellant" not in c.violations and "No comm window for uplink" not in c.violations


def _evaluated(sc: Scenario):
    from .decision import DEFAULT_WEIGHTS
    return [evaluate_candidate(sc, c, DEFAULT_WEIGHTS) for c in generate_candidates(sc)]


def probability_threshold(sc: Scenario) -> Candidate:
    """Maneuver if Pc > 1e-4, choosing the burn that minimises Pc."""
    cands = _evaluated(sc)
    if cands[0].pc <= PC_SCREEN:
        return cands[0]
    opts = [c for c in cands[1:] if _physical(c)] or cands[:1]
    return min(opts, key=lambda c: (c.pc, c.dv_mag))


def fuel_minimization(sc: Scenario) -> Candidate:
    """Lowest-dv burn that brings Pc below the limit."""
    cands = _evaluated(sc)
    if cands[0].pc <= PC_SCREEN:
        return cands[0]
    ok = [c for c in cands[1:] if _physical(c) and c.pc <= PC_MAX]
    if not ok:
        return probability_threshold(sc)
    return min(ok, key=lambda c: (c.dv_mag, c.pc))


def risk_plus_fuel(sc: Scenario) -> Candidate:
    """Two-objective utility (risk + fuel) without mission context or explainability."""
    cands = _evaluated(sc)
    if cands[0].pc <= PC_SCREEN:
        return cands[0]
    ok = [c for c in cands if _physical(c) and c.pc <= PC_MAX] or [c for c in cands if _physical(c)]
    return max(ok, key=lambda c: utility(c.objectives, RF_WEIGHTS))
