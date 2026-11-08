import pytest

from uplift.__main__ import gate, summary_markdown
from uplift.pipeline import UPLIFT_MODELS, customer_frame, recommend


def test_champion_is_an_uplift_model(study):
    assert study.champion in UPLIFT_MODELS
    assert study.report["champion"] == study.champion


def test_report_sections(study):
    for key in ("data", "qini", "deciles", "policies", "budget", "experiment", "timings"):
        assert key in study.report


def test_uplift_models_beat_churn_risk_on_qini(study):
    q = {m["model"]: m["qini_coefficient"] for m in study.report["qini"]["models"]}
    assert max(q[m] for m in UPLIFT_MODELS) > q["churn_risk"] > q["random"]


def test_best_uplift_policy_beats_churn_risk(study):
    rows = {r["policy"]: r for r in study.report["policies"]}
    assert max(rows[m]["net_value"] for m in UPLIFT_MODELS) > rows["churn_risk"]["net_value"]


def test_churn_risk_hits_more_lost_causes(study):
    rows = {r["policy"]: r for r in study.report["policies"]}
    assert rows["churn_risk"]["groups"]["lost_cause"] > rows[study.champion]["groups"]["lost_cause"]


def test_oracle_targets_only_persuadables(study):
    o = next(r for r in study.report["policies"] if r["policy"] == "oracle")
    assert o["groups"]["persuadable"] == o["targeted"] == o["incremental_retained"]


def test_budget_rows_stay_within_budget(study):
    for amount in (0, 2000, 20000):
        for r in study.allocate(amount)["rows"]:
            assert r["cost"] <= amount + 0.01


def test_greedy_reaches_lp_bound(study):
    b = study.allocate(5000, with_lp=True)
    assert b["greedy_share_of_lp"] > 0.99


def test_rollout_readout(study):
    e = study.report["experiment"]
    assert e["readout"]["churn_reduction"] > 0
    assert e["plan"][0]["holdout_share"] == 0.1
    assert e["versus_churn_risk"]["total"] > e["plan"][-1]["total"]


def test_gate_and_summary(study):
    ok, msg = gate(study.report)
    assert ok and "PASS" in msg
    md = summary_markdown(study.report)
    assert "Highest churn risk" in md and "LP relaxation bound" in md


def test_gate_fails_when_churn_risk_wins(study):
    rep = {"policies": [dict(r) for r in study.report["policies"]]}
    for r in rep["policies"]:
        if r["policy"] == "churn_risk":
            r["net_value"] = 1e12
    assert gate(rep)[0] is False


def offers(*specs):
    return [{"offer": k, "label": k, "eligible": True, "uplift": u, "cost": 5.0, "expected_net_value": v}
            for k, u, v in specs]


@pytest.mark.parametrize(
    "risk, specs, action, group",
    [
        (0.5, [("discount", 0.2, 30.0), ("free_month", 0.1, 5.0)], "discount", "persuadable"),
        (0.2, [("discount", -0.05, -20.0), ("free_month", -0.03, -25.0)], "none", "sleeping_dog"),
        (0.7, [("discount", 0.0, -5.0), ("free_month", 0.005, -9.0)], "none", "lost_cause"),
        (0.05, [("discount", 0.0, -5.0)], "none", "sure_thing"),
        (0.25, [("discount", 0.02, -3.0)], "none", "uncertain"),
    ],
)
def test_recommend_branches(risk, specs, action, group):
    r = recommend(risk, offers(*specs))
    assert r["action"] == action and r["likely_group"] == group and r["reason"]


def test_customer_frame_requires_all_fields():
    with pytest.raises(ValueError, match="missing"):
        customer_frame({"plan": "basic"})


def test_score_matches_offers(study):
    out = study.score(study.test.head(5))
    assert len(out) == 5
    for s in out:
        assert 0 <= s["churn_risk"] <= 1 and len(s["offers"]) == 3
