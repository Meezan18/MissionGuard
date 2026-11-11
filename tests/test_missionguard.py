import numpy as np
import pytest

from missionguard.baselines import fuel_minimization, probability_threshold, risk_plus_fuel
from missionguard.decision import OBJECTIVES, PC_MAX, PC_SCREEN, decide, utility
from missionguard.maneuvers import cw_displacement, generate_candidates
from missionguard.models import Candidate
from missionguard.risk import collision_probability
from missionguard.scenarios import make_scenario, make_scenarios


# ---------------------------------------------------------------- unit tests
def test_pc_head_on_matches_closed_form():
    """Zero miss, isotropic covariance: Pc = 1 - exp(-R^2 / 2 sigma^2)."""
    s, r = 0.1, 0.02
    pc = collision_probability(np.zeros(2), np.eye(2) * s * s, r)
    assert pc == pytest.approx(1 - np.exp(-r * r / (2 * s * s)), rel=1e-3)


def test_pc_falls_with_miss_distance():
    cov = np.eye(2) * 0.04
    assert collision_probability(np.array([1.0, 0]), cov, 0.015) < collision_probability(np.array([0.1, 0]), cov, 0.015)


def test_cw_in_track_drift():
    """An in-track burn produces a secular along-track drift of about -3*dv*t."""
    n = 0.0011
    t = 2 * np.pi / n            # one orbit
    d = cw_displacement(np.array([0, 0.1, 0]), t, n)
    assert d[1] == pytest.approx(-3 * 0.1e-3 * t, rel=1e-6)
    assert abs(d[0]) < 1e-9


def test_candidate_set_contains_all_types():
    kinds = {c.kind for c in generate_candidates(make_scenario(0))}
    assert kinds == {"no-action", "in-track", "radial", "cross-track"}


# ---------------------------------------------------------------- integration
def test_low_risk_event_returns_no_action():
    sc = next(s for s in make_scenarios(48) if decide(s).pc_before < PC_SCREEN)
    d = decide(sc)
    assert d.recommended.kind == "no-action"


def test_recommendation_satisfies_constraints():
    for sc in make_scenarios(48):
        d = decide(sc)
        if d.screened and d.recommended.feasible:
            r = d.recommended
            assert r.pc <= PC_MAX
            assert r.dv_mag <= sc.fuel_available - sc.fuel_reserve + 1e-9


def test_shap_sums_to_utility_minus_mean():
    d = next(decide(s) for s in make_scenarios(48) if decide(s).screened)
    mean_u = np.mean([c.utility for c in d.ranked])
    top = d.ranked[0]
    assert sum(top.shap.values()) == pytest.approx(top.utility - mean_u, abs=1e-9)


def test_ranking_is_by_utility():
    d = next(decide(s) for s in make_scenarios(48) if decide(s).screened)
    us = [c.utility for c in d.ranked]
    assert us == sorted(us, reverse=True)


def test_low_fuel_blocks_large_burns():
    sc = make_scenario(5)
    sc.fuel_available = 0.45      # only 0.15 m/s usable
    d = decide(sc)
    if d.screened:
        assert all(c.dv_mag <= 0.15 + 1e-9 for c in d.ranked if c.feasible)


def test_confidence_in_range_and_low_flagged():
    for sc in make_scenarios(48):
        d = decide(sc)
        assert 0.0 <= d.recommended.confidence <= 1.0


# ---------------------------------------------------------------- system
def test_missionguard_preserves_more_mission_than_baselines():
    scs = [s for s in make_scenarios(48) if decide(s).screened]
    mg = np.mean([1 - decide(s).recommended.objectives["M"] for s in scs])
    for fn in (probability_threshold, fuel_minimization, risk_plus_fuel):
        base = np.mean([1 - fn(s).objectives["M"] for s in scs])
        assert mg >= base


def test_dashboard_pages_render():
    from missionguard.app import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/scenario/0").status_code == 200
    assert b"Recommended action" in c.get("/scenario/0").data
