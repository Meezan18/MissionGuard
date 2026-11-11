"""MissionGuard operator dashboard (Flask).

Run:  python -m missionguard.app   then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from flask import Flask, abort, redirect, render_template, request, url_for

from .decision import OBJECTIVE_NAMES, OBJECTIVES, PC_MAX, PC_SCREEN, DEFAULT_WEIGHTS, decide
from .explain import LOW_CONFIDENCE
from .scenarios import make_scenarios

N_SCENARIOS = int(os.environ.get("MG_SCENARIOS", 48))
LOG_PATH = os.environ.get("MG_DECISION_LOG", "decisions.jsonl")
EVAL_CACHE = os.environ.get("MG_EVAL_CACHE", "evaluation.json")

app = Flask(__name__)
SCENARIOS = make_scenarios(N_SCENARIOS)


@lru_cache(maxsize=None)
def decision_for(sid: int):
    return decide(SCENARIOS[sid])


def read_log():
    if not os.path.exists(LOG_PATH):
        return {}
    out = {}
    with open(LOG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["scenario_id"]] = rec
    return out


# ------------------------------------------------------------------ svg helpers
def shap_svg(shap: dict) -> str:
    items = [(OBJECTIVE_NAMES[k], shap[k]) for k in OBJECTIVES]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    m = max(abs(v) for _, v in items) or 1
    w, row, lab = 520, 30, 230
    mid = lab + (w - lab) / 2
    half = (w - lab) / 2 - 50
    out = [f'<svg viewBox="0 0 {w} {row * len(items) + 10}" class="chart" role="img" aria-label="SHAP attributions">']
    out.append(f'<line x1="{mid}" y1="0" x2="{mid}" y2="{row * len(items)}" class="axis"/>')
    for i, (name, v) in enumerate(items):
        y = i * row + 6
        bw = abs(v) / m * half
        x = mid if v >= 0 else mid - bw
        cls = "pos" if v >= 0 else "neg"
        out.append(f'<text x="0" y="{y + 14}" class="lbl">{name}</text>')
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(bw, 1):.1f}" height="18" rx="3" class="{cls}"/>')
        tx = mid + bw + 6 if v >= 0 else mid - bw - 6
        anchor = "start" if v >= 0 else "end"
        out.append(f'<text x="{tx:.1f}" y="{y + 14}" class="val" text-anchor="{anchor}">{v:+.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def timeline_svg(sc, rec) -> str:
    end = max(sc.primary.tca + 4, max(s.tca for s in sc.secondaries) + 2)
    start = max(0.0, (rec.burn_time or sc.primary.tca) - 10)
    end = min(end, sc.primary.tca + 14)
    w, left = 1400, 110
    sx = lambda t: left + (t - start) / (end - start) * (w - left - 10)
    o = [f'<svg viewBox="0 0 {w} 150" class="chart" role="img" aria-label="Mission timeline">']
    lanes = [("Payload ops", 20), ("Comm passes", 60), ("Events", 100)]
    for name, y in lanes:
        o.append(f'<text x="0" y="{y + 14}" class="lbl">{name}</text>')
        o.append(f'<line x1="{left}" y1="{y + 10}" x2="{w - 10}" y2="{y + 10}" class="lane"/>')
    for a in sc.activities:
        if a.window.end < start or a.window.start > end:
            continue
        x1, x2 = sx(max(a.window.start, start)), sx(min(a.window.end, end))
        o.append(f'<rect x="{x1:.1f}" y="20" width="{max(x2 - x1, 2):.1f}" height="20" rx="2" class="p{a.priority}">'
                 f'<title>{a.name} priority {a.priority}</title></rect>')
    for c in sc.comm_windows:
        if c.end < start or c.start > end:
            continue
        x1, x2 = sx(max(c.start, start)), sx(min(c.end, end))
        o.append(f'<rect x="{x1:.1f}" y="60" width="{max(x2 - x1, 2):.1f}" height="20" rx="2" class="comm"/>')
    if rec.burn_time is not None:
        x = sx(rec.burn_time)
        o.append(f'<line x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="125" class="burn"/>')
        o.append(f'<text x="{x - 4:.1f}" y="122" class="evt" text-anchor="end">burn T+{rec.burn_time:.1f} h</text>')
    x = sx(sc.primary.tca)
    o.append(f'<line x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="125" class="tca"/>')
    o.append(f'<text x="{x + 4:.1f}" y="108" class="evt">TCA T+{sc.primary.tca:.1f} h</text>')
    for s in sc.secondaries:
        if start <= s.tca <= end:
            xs = sx(s.tca)
            o.append(f'<circle cx="{xs:.1f}" cy="110" r="4" class="sec"><title>{s.object_id}</title></circle>')
    for t in range(int(start), int(end) + 1, 2):
        o.append(f'<text x="{sx(t):.1f}" y="146" class="tick" text-anchor="middle">T+{t}h</text>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    log = read_log()
    rows = []
    for sc in SCENARIOS:
        d = decision_for(sc.scenario_id)
        rows.append({"sc": sc, "d": d, "rec": d.recommended, "log": log.get(sc.scenario_id)})
    rows.sort(key=lambda r: -r["d"].pc_before)
    return render_template("index.html", rows=rows, PC_SCREEN=PC_SCREEN, LOW=LOW_CONFIDENCE)


@app.route("/scenario/<int:sid>")
def scenario(sid: int):
    if not 0 <= sid < len(SCENARIOS):
        abort(404)
    sc = SCENARIOS[sid]
    d = decision_for(sid)
    rec = d.recommended
    log = read_log().get(sid)
    return render_template("scenario.html", sc=sc, d=d, rec=rec, log=log,
                           shap=shap_svg(rec.shap) if rec.shap else "",
                           timeline=timeline_svg(sc, rec), ranked=d.ranked[:8],
                           weights=DEFAULT_WEIGHTS, names=OBJECTIVE_NAMES, PC_MAX=PC_MAX, LOW=LOW_CONFIDENCE)


@app.route("/scenario/<int:sid>/decision", methods=["POST"])
def record(sid: int):
    d = decision_for(sid)
    rec = {
        "scenario_id": sid,
        "decision": request.form.get("decision"),
        "maneuver": d.recommended.label,
        "confidence": round(d.recommended.confidence, 3),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return redirect(url_for("scenario", sid=sid))


@app.route("/evaluation")
def evaluation():
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE) as f:
            res = json.load(f)
    else:
        from .evaluate import run
        res = run(N_SCENARIOS)
        with open(EVAL_CACHE, "w") as f:
            json.dump(res, f, indent=2)
    return render_template("evaluation.html", res=res)


@app.template_filter("sci")
def sci(v):
    return f"{v:.1e}" if v >= 1e-12 else "< 1e-12"


def main():
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)


if __name__ == "__main__":
    main()
