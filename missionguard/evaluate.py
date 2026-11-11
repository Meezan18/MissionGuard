"""Compare MissionGuard with the three baselines on the synthetic scenarios.

Usage:  python -m missionguard.evaluate [--n 48] [--seed 2026] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .baselines import fuel_minimization, probability_threshold, risk_plus_fuel
from .decision import PC_ALERT, decide
from .scenarios import make_scenarios

METHODS = {
    "Probability threshold": probability_threshold,
    "Fuel minimization": fuel_minimization,
    "Risk + Fuel": risk_plus_fuel,
    "MissionGuard": lambda sc: decide(sc).recommended,
}


def metrics_for(sc, cand, pc0, sec0):
    rr = 1 - cand.pc / pc0 if pc0 > 0 else 0.0
    new_conj = sum(1 for p, p0 in zip(cand.secondary_pcs, sec0) if p > PC_ALERT and p0 <= PC_ALERT)
    return {
        "risk_reduction": rr,
        "dv": cand.dv_mag,
        "mission_preservation": 1 - cand.objectives["M"],
        "new_conjunctions": new_conj,
        "safe": cand.pc <= 1e-4,
    }


def run(n=48, seed=2026):
    scs = make_scenarios(n, seed)
    rows = {m: [] for m in METHODS}
    timing = []
    screened = 0
    low_conf = 0
    for sc in scs:
        t0 = time.perf_counter()
        dec = decide(sc)
        timing.append(time.perf_counter() - t0)
        if not dec.screened:
            continue                      # below threshold: every method takes no action
        screened += 1
        if dec.recommended.confidence < 0.5:
            low_conf += 1
        nc = dec.candidates[0]
        for m, fn in METHODS.items():
            c = dec.recommended if m == "MissionGuard" else fn(sc)
            rows[m].append(metrics_for(sc, c, nc.pc, nc.secondary_pcs))
    summary = {}
    for m, rs in rows.items():
        summary[m] = {
            "risk_reduction": float(np.mean([r["risk_reduction"] for r in rs])),
            "dv": float(np.mean([r["dv"] for r in rs])),
            "mission_preservation": float(np.mean([r["mission_preservation"] for r in rs])),
            "new_conjunctions": float(np.mean([r["new_conjunctions"] for r in rs])),
            "safe_rate": float(np.mean([r["safe"] for r in rs])),
        }
    return {
        "n_scenarios": n,
        "n_screened": screened,
        "low_confidence": low_conf,
        "mean_decision_time_s": float(np.mean(timing)),
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = run(a.n, a.seed)
    print(f"Scenarios: {res['n_scenarios']}  screened (Pc >= 1e-4): {res['n_screened']}  "
          f"low-confidence: {res['low_confidence']}  mean decision time: {res['mean_decision_time_s']:.3f} s")
    print(f"{'Method':24s}{'Risk red.':>10s}{'dv m/s':>9s}{'Miss.pres':>11s}{'New conj':>10s}{'Safe':>7s}")
    for m, s in res["summary"].items():
        print(f"{m:24s}{s['risk_reduction']:10.3f}{s['dv']:9.3f}{s['mission_preservation']:11.3f}"
              f"{s['new_conjunctions']:10.3f}{s['safe_rate']:7.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
