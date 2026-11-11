# AGENTS.md – MissionGuard

Guidance for coding agents (opencode, Claude Code, etc.) working on this repository.

## What this project is
MissionGuard is an explainable, mission-aware decision-support system for satellite
collision avoidance (M.Tech. CSE PW-I/PW-II, Erode Sengunthar Engineering College;
paper presented at IEEE CEEE 2026, paper ID 3585).

For each conjunction it generates candidate avoidance maneuvers, scores each on six
normalised objectives, filters by hard constraints, ranks by a utility function,
explains the top choice with SHAP and attaches a confidence score. A human operator
approves or rejects the recommendation on a web dashboard.

## Status
- **Phase I (done, this code):** working prototype on 48 reproducible synthetic LEO
  scenarios, 3 baselines, Flask dashboard, 12 passing tests.
- **Phase II (to build):** see "Phase II backlog" below. Do the work packages in order.

## Commands
```bash
pip install -r requirements.txt
python -m pytest -q                              # all tests must pass before any commit
python -m missionguard.evaluate --json evaluation.json   # 48-scenario comparison table
python -m missionguard.app                       # dashboard on http://127.0.0.1:5000
```

## Layout
| File | Responsibility |
|---|---|
| `missionguard/models.py` | Dataclasses: Scenario, Conjunction, Candidate, Window, Activity, Health |
| `missionguard/scenarios.py` | Seeded synthetic scenario generator (`make_scenarios(48, seed=2026)`) |
| `missionguard/risk.py` | B-plane projection, Pc by 2-D Gaussian integration over the hard-body disk, log risk scale |
| `missionguard/maneuvers.py` | Clohessy-Wiltshire propagation; candidate set (no action, in-track, radial, cross-track × ±, 4 Δv levels, 4 lead times) |
| `missionguard/decision.py` | Objectives S,F,M,O,Q,H; constraints; utility U(a); `decide()` = Algorithm 1 |
| `missionguard/explain.py` | Exact linear SHAP, confidence C(a)=0.4·model+0.3·data+0.3·agreement |
| `missionguard/baselines.py` | Probability threshold, fuel minimisation, risk + fuel |
| `missionguard/evaluate.py` | Metrics and comparison table |
| `missionguard/app.py`, `templates/` | Flask dashboard (queue, scenario detail, evaluation, approve/reject log) |
| `tests/` | Unit, integration and system tests |

## Key equations (keep these consistent with the paper)
- `U(a) = wS·S − wF·F − wM·M − wO·O − wQ·Q − wH·H`, `S = 1 − R`, weights sum to 1
  (defaults in `decision.DEFAULT_WEIGHTS`).
- Constraints: `Pc(a) ≤ 1e-5`, `Δv(a) ≤ fuel_available − fuel_reserve`, command uplinked in a
  comm pass 10 min – 8 h before the burn.
- Screening: no action if `Pc < 1e-4`.

## Conventions
- Python 3.11, type hints, no global state except the dashboard's cached scenarios.
- Units: time in hours from epoch, distances in km, Δv in m/s, Pc dimensionless.
- Every new module gets tests in `tests/`. Keep results reproducible (seeded RNG).
- Never hard-code results into slides or the dashboard; always compute them.

## Phase II backlog (Dec 2026 – Apr 2027)
1. **WP1 Real-data pipeline** – `missionguard/data/`: load ESA Kelvins Collision Avoidance
   Challenge CDM CSV (Uriot et al., 2022) and TLEs (Space-Track API with user credentials
   from env vars, CelesTrak fallback); propagate with `sgp4`. Produce `Scenario` objects from
   real events. *Done when* `decide()` runs on ≥ 500 real events.
2. **WP2 Learned risk model** – gradient-boosting model predicting final risk from the CDM
   time series. *Done when* it beats the "latest CDM" baseline (MSE/F2) on the held-out set.
3. **WP3 Learned suitability + adaptive weights** – learn weights from logged operator
   decisions (`decisions.jsonl`); replace linear SHAP with the `shap` library on the learned model.
4. **WP4 Forecasting & maneuver verification** – multi-revolution post-maneuver screening
   against a catalogue; detect executed burns from TLE history to update the state.
5. **WP5 Multi-satellite coordination** – multi-agent RL (PettingZoo environment) so burns
   of satellites in one constellation do not conflict.
6. **WP6 Operator study & delivery** – dashboard hardening, decision-time and trust study,
   final report and journal paper.
