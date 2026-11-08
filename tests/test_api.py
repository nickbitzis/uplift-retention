import pytest
from fastapi.testclient import TestClient

from uplift.app import create_app

CUSTOMER = {"tenure_months": 8, "plan": "premium", "price_increase": 1, "weekly_hours": 6.5, "usage_trend": -0.1,
            "support_tickets": 1, "payment_failures": 0, "primary_genre": "drama", "genre_count": 3,
            "device_count": 2, "promo_history": 1}


@pytest.fixture(scope="module")
def client(study):
    return TestClient(create_app(study))


def test_healthz(client, study):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["champion"] == study.champion


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Qini curves" in r.text


def test_summary(client):
    j = client.get("/api/summary").json()
    assert {"qini", "policies", "budget", "experiment"} <= set(j)
    assert len(j["qini"]["curves"]["x_learner"]) > 50


def test_budget_endpoint(client):
    j = client.get("/api/budget", params={"amount": 3000, "lp": True}).json()
    assert j["budget"] == 3000 and j["lp_bound"] is not None and len(j["rows"]) == 4


def test_budget_validation(client):
    assert client.get("/api/budget", params={"amount": -1}).status_code == 422


def test_policy_evaluate(client):
    j = client.post("/api/policy/evaluate", json={"policy": "churn_risk", "share": 0.1}).json()
    assert j["targeted"] == round(0.1 * 6000, -1) or j["targeted"] > 0
    assert set(j["groups"]) == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}


def test_policy_evaluate_rejects_unknown(client):
    assert client.post("/api/policy/evaluate", json={"policy": "vibes"}).status_code == 422


def test_score_customer(client):
    j = client.post("/api/score", json=CUSTOMER).json()
    assert 0 <= j["churn_risk"] <= 1
    assert {o["offer"] for o in j["offers"]} == {"discount", "free_month", "downgrade"}
    assert j["action"] in {"discount", "free_month", "downgrade", "none"} and j["reason"]


def test_score_validation(client):
    assert client.post("/api/score", json={**CUSTOMER, "plan": "platinum"}).status_code == 422
    assert client.post("/api/score", json={k: v for k, v in CUSTOMER.items() if k != "plan"}).status_code == 422


def test_basic_plan_downgrade_ineligible(client):
    j = client.post("/api/score", json={**CUSTOMER, "plan": "basic"}).json()
    d = next(o for o in j["offers"] if o["offer"] == "downgrade")
    assert d["eligible"] is False and j["action"] != "downgrade"


def test_sample_and_lookup(client):
    s = client.get("/api/customers/sample").json()
    assert {x["true_group"] for x in s} == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
    j = client.get(f"/api/customers/{s[0]['customer_id'].lower()}").json()
    assert j["customer_id"] == s[0]["customer_id"]
    assert j["ground_truth"]["group"] == s[0]["true_group"]
    assert "plan" in j["features"]


def test_lookup_unknown(client):
    assert client.get("/api/customers/C999999").status_code == 404


def test_request_id_and_metrics(client):
    r = client.get("/healthz", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    client.post("/api/score", json=CUSTOMER)
    m = client.get("/metrics").text
    assert "uplift_http_requests_total" in m and "uplift_recommendations_total" in m
    assert "uplift_best_policy_net_value" in m and "uplift_http_request_duration_seconds_bucket" in m
