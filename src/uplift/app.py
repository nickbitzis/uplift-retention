"""Retention targeting API and UI.

The fitted study lives in memory. Training happens once at startup (or loads from a cached pickle), and the
budget endpoint re-runs only the allocation, which takes milliseconds.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .generator import FEATURES, GENRES, GROUPS
from .logs import get_logger, log_event
from .metrics import Metrics
from .offers import PLANS
from .pipeline import POLICY_LABELS, Study, StudyConfig, customer_frame

STATIC = Path(__file__).parent / "static"


class CustomerIn(BaseModel):
    tenure_months: int = Field(ge=0, le=240)
    plan: Literal[PLANS]  # type: ignore[valid-type]
    price_increase: int = Field(ge=0, le=1)
    weekly_hours: float = Field(ge=0, le=168)
    usage_trend: float = Field(ge=-1, le=1)
    support_tickets: int = Field(ge=0, le=50)
    payment_failures: int = Field(ge=0, le=20)
    primary_genre: Literal[GENRES]  # type: ignore[valid-type]
    genre_count: int = Field(ge=1, le=10)
    device_count: int = Field(ge=1, le=10)
    promo_history: int = Field(ge=0, le=20)


class PolicyIn(BaseModel):
    policy: Literal[tuple(POLICY_LABELS)]  # type: ignore[valid-type]
    share: float = Field(default=0.2, gt=0, le=1)


def load_study() -> Study:
    path = Path(os.environ.get("UPLIFT_STUDY", "data/study.pkl"))
    log = get_logger()
    if path.exists():
        try:
            study = Study.load(path)
            log_event(log, "study loaded", path=str(path))
            return study
        except Exception as exc:  # stale or incompatible pickle: retrain
            log_event(log, "study cache unreadable, retraining", path=str(path), error=str(exc))
    n = int(os.environ.get("UPLIFT_N", "200000"))
    cfg = StudyConfig(n=n, pilot_n=max(n // 2, 4000), rollout_n=max(n // 2, 4000))
    study = Study(cfg).run()
    study.save(path)
    log_event(log, "study trained", path=str(path), **study.report["timings"])
    return study


def create_app(study: Study | None = None) -> FastAPI:
    study = study or load_study()
    metrics = Metrics()
    log = get_logger()
    test = study.test
    by_id = {cid: i for i, cid in enumerate(test["customer_id"])}
    rep = study.report

    best = max((r for r in rep["policies"] if r["policy"] in study.learners), key=lambda r: r["net_value"])
    churn_row = next(r for r in rep["policies"] if r["policy"] == "churn_risk")
    metrics.set_gauge("uplift_test_customers", "Customers in the test split.", len(test))
    metrics.set_gauge("uplift_best_policy_net_value", "True net value of the best uplift policy (test).",
                      best["net_value"])
    metrics.set_gauge("uplift_churn_policy_net_value", "True net value of churn risk targeting (test).",
                      churn_row["net_value"])
    for m in rep["qini"]["models"]:
        metrics.set_gauge(f"uplift_qini_{m['model']}", f"Qini coefficient on test: {m['label']}.",
                          m["qini_coefficient"])

    app = FastAPI(title="Uplift Retention", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        path = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(log, "request", request_id=rid, method=request.method, path=request.url.path,
                      status=response.status_code, ms=round(dt * 1000, 1))
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "test_customers": len(test), "champion": study.champion}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        return rep

    @app.get("/api/budget")
    def budget(amount: float = Query(ge=0, le=10_000_000), lp: bool = False):
        return study.allocate(amount, with_lp=lp)

    @app.post("/api/policy/evaluate")
    def evaluate(body: PolicyIn):
        return study.evaluate_policy(body.policy, body.share)

    def score_one(row) -> dict:
        res = study.score(row)[0]
        metrics.recommendation(res["action"])
        return res

    @app.post("/api/score")
    def score(body: CustomerIn):
        return score_one(customer_frame(body.model_dump()))

    @app.get("/api/customers/sample")
    def sample(per_group: int = Query(default=2, ge=1, le=10)):
        rng = np.random.default_rng(int(time.time()) // 3600)
        out = []
        for g in GROUPS:
            ids = test.loc[test["true_group"] == g, "customer_id"].to_numpy()
            out += [{"customer_id": c, "true_group": g} for c in rng.choice(ids, min(per_group, len(ids)), False)]
        return out

    @app.get("/api/customers/{customer_id}")
    def customer(customer_id: str):
        i = by_id.get(customer_id.strip().upper())
        if i is None:
            raise HTTPException(404, f"customer {customer_id} is not in the test split")
        row = test.iloc[[i]]
        res = score_one(row)
        feats = {f: (v.item() if hasattr(v, "item") else v) for f, v in row[list(FEATURES)].iloc[0].items()}
        truth = {
            "group": row["true_group"].iloc[0],
            "true_churn_no_offer": round(float(row["true_p0"].iloc[0]), 4),
            "true_uplift": {k: round(float(row[f"true_uplift_{k}"].iloc[0]), 4)
                            for k in ("discount", "free_month", "downgrade")},
        }
        return {"customer_id": test["customer_id"].iloc[i], "features": feats, "score": res,
                "ground_truth": truth, "note": "Ground truth exists only because the data is synthetic."}

    return app
