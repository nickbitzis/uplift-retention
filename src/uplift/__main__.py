"""Command line: serve, eval, gate.

    uv run uplift serve                 # API and UI on http://127.0.0.1:8830 (trains on first start, then cached)
    uv run uplift eval                  # full study, writes reports/eval.json and reports/summary.md
    uv run uplift eval --gate           # same, exit 1 unless the best uplift policy beats churn risk targeting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def summary_markdown(rep: dict) -> str:
    from .pipeline import markdown_policy_table

    lines = ["# Study summary", "", f"Champion uplift model (best validation Qini): `{rep['champion']}`", ""]
    lines += ["## Policies at 20% of the test set (discount offer)", "", markdown_policy_table(rep["policies"]), ""]
    lines += ["## Qini (test)", "", "| Model | Qini coefficient | Uplift AUC | True gain AUC |", "|---|---:|---:|---:|"]
    for m in rep["qini"]["models"]:
        lines.append(f"| {m['label']} | {m['qini_coefficient']:,.1f} | {m['uplift_auc']:,.1f} | "
                     f"{m['true_gain_auc']:,.1f} |")
    b = rep["budget"]
    lines += ["", f"## Budget ${b['budget']:,.0f}", "", "| Allocation | Targeted | Cost | Expected net | True net | "
              "Discount | Free month | Downgrade |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in b["rows"]:
        o = r["offers"]
        lines.append(f"| {r['label']} | {r['targeted']:,} | ${r['cost']:,.0f} | ${r['expected_net_value']:,.0f} | "
                     f"${r['net_value']:,.0f} | {o['discount']:,} | {o['free_month']:,} | {o['downgrade']:,} |")
    if b.get("lp_bound"):
        lines.append(f"\nLP relaxation bound: ${b['lp_bound']:,.0f} (greedy reaches {b['greedy_share_of_lp']:.1%}).")
    return "\n".join(lines) + "\n"


def gate(rep: dict) -> tuple[bool, str]:
    rows = {r["policy"]: r for r in rep["policies"]}
    best = max(("t_learner", "s_learner", "x_learner", "class_transform"), key=lambda k: rows[k]["net_value"])
    b, c = rows[best]["net_value"], rows["churn_risk"]["net_value"]
    ok = b > c
    return ok, f"best uplift policy {best} true net value ${b:,.0f} vs churn risk ${c:,.0f}: {'PASS' if ok else 'FAIL'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uplift")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8830")))
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=200_000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", default="reports")
    e.add_argument("--gate", action="store_true", help="exit 1 unless the best uplift policy beats churn risk")
    e.add_argument("--save", default=None, help="also pickle the fitted study here (the server loads it)")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("uplift.app:create_app", factory=True, host=args.host, port=args.port, access_log=False)
        return 0

    from .pipeline import Study, StudyConfig

    n = args.n
    cfg = StudyConfig(n=n, pilot_n=n // 2, rollout_n=n // 2, seed=args.seed)
    study = Study(cfg).run()
    rep = _clean(study.report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(rep, indent=1))
    (out / "summary.md").write_text(summary_markdown(rep))
    if args.save:
        study.save(Path(args.save))
    ok, msg = gate(rep)
    print(summary_markdown(rep))
    print(msg)
    if args.gate and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
