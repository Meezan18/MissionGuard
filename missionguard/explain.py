"""Explainability and confidence (paper Section IV-A).

SHAP: the Phase I suitability model is the linear utility U(a). For a linear
model with independent features, Kernel/Exact SHAP reduces to the closed form
    phi_j = sign_j * w_j * (x_j - E[x_j]),
where the expectation is taken over the feasible candidate set. The
attributions therefore sum exactly to U(a) - E[U]. Phase II swaps in a learned
(non-linear) suitability model and the `shap` library.

Confidence: C(a) = alpha*C_model + beta*C_data + gamma*C_agreement.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .decision import OBJECTIVES, SIGN, Decision, utility
from .models import Candidate, Scenario

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
LOW_CONFIDENCE = 0.5


def attach_shap(cands: List[Candidate], weights: Dict[str, float]) -> None:
    if not cands:
        return
    mean = {k: float(np.mean([c.objectives[k] for c in cands])) for k in OBJECTIVES}
    for c in cands:
        c.shap = {k: SIGN[k] * weights[k] * (c.objectives[k] - mean[k]) for k in OBJECTIVES}


def _agreement(dec: Decision, weights: Dict[str, float], n: int = 200, seed: int = 7) -> float:
    """Share of perturbed weight vectors that keep the same top recommendation."""
    rng = np.random.default_rng(seed + dec.scenario_id)
    base = np.array([weights[k] for k in OBJECTIVES])
    top = dec.ranked[0]
    same = 0
    for _ in range(n):
        w = rng.dirichlet(base * 40)
        wd = dict(zip(OBJECTIVES, w))
        best = max(dec.ranked, key=lambda c: utility(c.objectives, wd))
        same += (best.kind, best.burn_time) == (top.kind, top.burn_time)
    return same / n


def attach_confidence(sc: Scenario, dec: Decision, weights: Dict[str, float]) -> None:
    ranked = dec.ranked
    top = ranked[0]
    # margin over the best *materially different* option (other maneuver type or burn time)
    alt = [c for c in ranked[1:] if (c.kind, c.burn_time) != (top.kind, top.burn_time)]
    c_model = 1.0 if not alt else float(np.clip((top.utility - alt[0].utility) / 0.03, 0, 1))
    sig = np.sqrt(np.trace(sc.primary.cov_b) / 2)
    c_data = float(np.clip(1 - sc.cdm_age_h / 96, 0, 1) * np.clip(1 - (sig - 0.05) / 0.6, 0.3, 1))
    c_agree = _agreement(dec, weights)
    top.confidence_parts = (c_model, c_data, c_agree)
    top.confidence = ALPHA * c_model + BETA * c_data + GAMMA * c_agree
    if not top.feasible:
        top.confidence = min(top.confidence, 0.3)
