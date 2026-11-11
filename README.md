# MissionGuard – Phase I prototype

Explainable multi-objective decision support for autonomous satellite collision avoidance.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # 12 tests
python -m missionguard.evaluate                        # comparison on 48 scenarios
python -m missionguard.app                             # open http://127.0.0.1:5000
```

## Phase I results (seed 2026, 48 scenarios, 37 above Pc ≥ 1e-4)
| Method | Risk reduction | Avg. Δv (m/s) | Mission preservation | New conj./event |
|---|---|---|---|---|
| Probability threshold | 1.000 | 0.335 | 0.893 | 0.000 |
| Fuel minimization | 1.000 | 0.053 | 0.871 | 0.027 |
| Risk + Fuel | 1.000 | 0.055 | 0.852 | 0.081 |
| **MissionGuard** | **1.000** | **0.066** | **0.981** | **0.000** |

14 of 37 recommendations were flagged low-confidence (C < 0.5); mean decision time ≈ 0.09 s.

## Modelling assumptions (Phase I)
- Synthetic scenarios; no real telemetry or mission schedules.
- Short-encounter Pc (2-D Gaussian in the B-plane), Clohessy-Wiltshire relative motion,
  impulsive burns, near-circular LEO.
- The suitability model is the linear utility, so SHAP values are computed exactly in closed form.

See `AGENTS.md` for architecture and the Phase II backlog.
