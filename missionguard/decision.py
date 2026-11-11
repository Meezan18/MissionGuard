"""Mission-aware multi-objective decision layer (paper Section IV).

U(a) = ws*S(a) - wf*F(a) - wm*M(a) - wo*O(a) - wq*Q(a) - wh*H(a),  S = 1 - R
subject to  Pc(a) <= Pc_max,  dv(a) <= dv_available - dv_reserve,
            command uplinked in a communication window before the burn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .maneuvers import displacement_at, generate_candidates
from .models import Candidate, Scenario
from .risk import conjunction_pc, risk_scale

OBJECTIVES = ("S", "F", "M", "O", "Q", "H")
OBJECTIVE_NAMES = {
    "S": "Safety benefit (1 - residual risk)",
    "F": "Propellant use",
    "M": "Mission disruption",
    "O": "Operational infeasibility",
    "Q": "New-conjunction risk",
    "H": "Health penalty",
}
SIGN = {"S": +1, "F": -1, "M": -1, "O": -1, "Q": -1, "H": -1}

DEFAULT_WEIGHTS = {"S": 0.35, "F": 0.15, "M": 0.20, "O": 0.05, "Q": 0.15, "H": 0.10}

PC_SCREEN = 1e-4         # screening threshold tau_screen
PC_MAX = 1e-5            # maximum acceptable post-maneuver Pc
PC_ALERT = 1e-5          # a secondary event above this counts as a (new) conjunction
DV_REF = 0.5             # m/s used to normalise propellant use
UPLINK_LEAD_H = 10 / 60  # command must be uplinked at least 10 min before the burn
UPLINK_MAX_AGE_H = 8.0   # ... and no more than 8 h before it
BLACKOUT_PRE_H = 10 / 60
BLACKOUT_POST_H = {"in-track": 15 / 60, "radial": 20 / 60, "cross-track": 30 / 60}


@dataclass
class Decision:
    scenario_id: int
    pc_before: float
    screened: bool
    candidates: List[Candidate]
    ranked: List[Candidate] = field(default_factory=list)

    @property
    def recommended(self) -> Candidate:
        return self.ranked[0] if self.ranked else self.candidates[0]


# ---------------------------------------------------------------- objectives
def uplink_window(sc: Scenario, burn_time: float):
    """Latest communication pass that can uplink the command for this burn."""
    best = None
    for w in sc.comm_windows:
        if w.end <= burn_time - UPLINK_LEAD_H and burn_time - w.end <= UPLINK_MAX_AGE_H:
            if best is None or w.end > best.end:
                best = w
    return best


def mission_disruption(sc: Scenario, c: Candidate) -> float:
    if c.burn_time is None:
        return 0.0
    a = c.burn_time - BLACKOUT_PRE_H
    b = c.burn_time + BLACKOUT_POST_H[c.kind]
    total = sum(x.priority for x in sc.activities) or 1
    hit = sum(x.priority for x in sc.activities if x.window.overlaps(a, b))
    # weight relative to the activities in a 24 h band around the burn
    band = sum(x.priority for x in sc.activities if x.window.overlaps(c.burn_time - 12, c.burn_time + 12)) or total
    return float(np.clip(hit / band * 4, 0, 1))


def health_penalty(sc: Scenario, c: Candidate) -> float:
    if c.burn_time is None:
        return 0.0
    h = sc.health
    dv_n = min(c.dv_mag / DV_REF, 1.0)
    slew = {"in-track": 0.2, "radial": 0.5, "cross-track": 1.0}[c.kind]
    pen = 0.5 * (1 - h.propulsion) * (0.5 + dv_n) + 0.3 * (1 - h.adcs) * slew + 0.2 * (1 - h.power) * (0.5 + dv_n)
    return float(np.clip(pen * 2, 0, 1))


def operational_infeasibility(sc: Scenario, c: Candidate) -> float:
    if c.burn_time is None:
        return 0.0
    w = uplink_window(sc, c.burn_time)
    if w is None:
        return 1.0
    age = c.burn_time - w.end
    return float(np.clip(age / UPLINK_MAX_AGE_H, 0, 1)) * (1.2 - sc.health.comms)


def evaluate_candidate(sc: Scenario, c: Candidate, weights: Dict[str, float]) -> Candidate:
    n = sc.mean_motion
    d = displacement_at(c, sc.primary.tca, n)
    c.pc, c.miss_km = conjunction_pc(sc.primary, d)
    c.secondary_pcs = [conjunction_pc(s, displacement_at(c, s.tca, n))[0] for s in sc.secondaries]
    q_tot = 1 - np.prod([1 - p for p in c.secondary_pcs]) if c.secondary_pcs else 0.0
    c.objectives = {
        "S": 1 - risk_scale(c.pc),
        "F": min(c.dv_mag / DV_REF, 1.0),
        "M": mission_disruption(sc, c),
        "O": operational_infeasibility(sc, c),
        "Q": risk_scale(q_tot),
        "H": health_penalty(sc, c),
    }
    c.violations = []
    if c.pc > PC_MAX:
        c.violations.append("Pc above limit")
    if c.dv_mag > sc.fuel_available - sc.fuel_reserve + 1e-9:
        c.violations.append("Insufficient propellant")
    if c.burn_time is not None and uplink_window(sc, c.burn_time) is None:
        c.violations.append("No comm window for uplink")
    c.feasible = not c.violations
    c.utility = utility(c.objectives, weights)
    return c


def utility(obj: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(SIGN[k] * weights[k] * obj[k] for k in OBJECTIVES))


# ---------------------------------------------------------------- pipeline
def decide(sc: Scenario, weights: Dict[str, float] | None = None) -> Decision:
    """Algorithm 1 of the paper."""
    from .explain import attach_confidence, attach_shap

    weights = weights or DEFAULT_WEIGHTS
    cands = generate_candidates(sc)
    for c in cands:
        evaluate_candidate(sc, c, weights)
    pc0 = cands[0].pc
    dec = Decision(sc.scenario_id, pc0, pc0 >= PC_SCREEN, cands)
    if not dec.screened:
        dec.ranked = [cands[0]]          # no action; continue monitoring
        cands[0].confidence = 1.0
        return dec
    feasible = [c for c in cands if c.feasible]
    if not feasible:
        # fall back to the safest physically possible option, flagged for the operator
        possible = [c for c in cands if "Insufficient propellant" not in c.violations] or cands
        feasible = sorted(possible, key=lambda c: c.pc)[:5]
    dec.ranked = sorted(feasible, key=lambda c: c.utility, reverse=True)
    attach_shap(dec.ranked, weights)
    attach_confidence(sc, dec, weights)
    return dec
