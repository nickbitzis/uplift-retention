"""End to end study: generate, split, train every model, evaluate policies, allocate budget, design the test.

`Study.run()` does everything once. The API keeps the fitted study in memory and re-runs only the budget
allocation when the slider moves.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import budget as bud
from . import evaluation as ev
from . import experiment as exp
from . import policy as pol
from .generator import FEATURES, GROUPS, GenConfig, generate, split
from .learners import LEARNERS, ChurnModel, MultiOfferModel, default_classifier, default_regressor, matrix
from .offers import OFFER_KEYS, OFFERS, offer_economics

POLICY_LABELS = {
    "churn_risk": "Highest churn risk",
    "random": "Random",
    "t_learner": "Uplift: T-learner",
    "s_learner": "Uplift: S-learner",
    "x_learner": "Uplift: X-learner",
    "class_transform": "Uplift: class transformation",
    "oracle": "Oracle (true persuadables only)",
}
UPLIFT_MODELS = ("t_learner", "s_learner", "x_learner", "class_transform")


@dataclass(frozen=True)
class StudyConfig:
    n: int = 200_000
    pilot_n: int = 100_000
    rollout_n: int = 100_000
    seed: int = 42
    target_share: float = 0.20  # fixed top-k policies target this share of the test set
    budget_per_customer: float = 1.00  # default budget, in dollars per test customer
    holdout_share: float = 0.20
    max_iter: int = 200

    @property
    def default_budget(self) -> float:
        return round(self.budget_per_customer * self.n * 0.2, -2)


class Study:
    def __init__(self, cfg: StudyConfig | None = None):
        self.cfg = cfg or StudyConfig()
        self.report: dict = {}
        self.timings: dict[str, float] = {}

    # ---------- training ----------

    def _clf(self, offset: int = 0):
        cfg = self.cfg
        return lambda: default_classifier(cfg.seed + offset, cfg.max_iter)

    def run(self) -> Study:
        cfg = self.cfg
        t0 = time.perf_counter()
        main = generate(GenConfig(n=cfg.n, seed=cfg.seed, design="binary"))
        main["split"] = split(main, cfg.seed)
        pilot = generate(GenConfig(n=cfg.pilot_n, seed=cfg.seed + 100, design="multi"))
        pilot["customer_id"] = "P" + pilot["customer_id"].str[1:]
        self.timings["generate_s"] = time.perf_counter() - t0

        tr = main[main["split"] == "train"]
        va = main[main["split"] == "valid"].reset_index(drop=True)
        te = main[main["split"] == "test"].reset_index(drop=True)
        Xtr, Xva, Xte = matrix(tr), matrix(va), matrix(te)
        ttr, ytr = tr["treated"].to_numpy(), tr["churned"].to_numpy()

        t1 = time.perf_counter()
        self.churn = ChurnModel(self._clf(1)).fit(Xtr, ttr, ytr)
        self.learners = {}
        for i, cls in enumerate(LEARNERS):
            if cls.name == "x_learner":
                m = cls(self._clf(10 + i), lambda: default_regressor(cfg.seed + 20, max(cfg.max_iter * 3 // 4, 10)))
            else:
                m = cls(self._clf(10 + i))
            self.learners[cls.name] = m.fit(Xtr, ttr, ytr)
        both = pd.concat([tr.assign(arm=tr["treated"]), pilot], ignore_index=True)
        # Smaller trees for the offer models: the pilot arms are 25,000 customers each, and shallower trees cut
        # the noise that greedy allocation would otherwise chase (winner's curse).
        self.offer_model = MultiOfferModel(
            len(OFFER_KEYS), lambda: default_classifier(cfg.seed + 30, cfg.max_iter, leaves=15)
        ).fit(
            matrix(both), both["arm"].to_numpy(), both["churned"].to_numpy()
        )
        self.timings["train_s"] = time.perf_counter() - t1

        self.valid, self.test = va, te
        self.Xte = Xte
        self._score_test(Xva, Xte)
        self._evaluate()
        self.timings["total_s"] = time.perf_counter() - t0
        self.report["timings"] = {k: round(v, 1) for k, v in self.timings.items()}
        return self

    def _score_test(self, Xva, Xte):
        va, te = self.valid, self.test
        self.valid_qini = {
            name: ev.qini_coefficient(m.predict_uplift(Xva), va["treated"], 1 - va["churned"])
            for name, m in self.learners.items()
        }
        self.champion = max(self.valid_qini, key=self.valid_qini.get)
        self.scores = {"churn_risk": self.churn.predict_churn(Xte)}
        for name, m in self.learners.items():
            self.scores[name] = m.predict_uplift(Xte)
        self.scores["random"] = np.random.default_rng(self.cfg.seed + 7).random(len(te))
        self.econ = offer_economics(te["plan"].to_numpy(), te["tenure_months"].to_numpy())
        self.p0, self.pk = self.offer_model.predict(Xte)
        self.values = pol.expected_values(self.p0, self.pk, self.econ)

    # ---------- evaluation ----------

    def _evaluate(self):
        cfg, te = self.cfg, self.test
        t, r = te["treated"].to_numpy(), 1 - te["churned"].to_numpy()
        true_eff = (te["y0"] - te["y_discount"]).to_numpy()
        curves, models = {}, []
        for name in ("churn_risk", *UPLIFT_MODELS, "random"):
            s = self.scores[name]
            x, q = ev.qini_curve(s, t, r)
            curves[name] = ev.downsample(x, q)
            xg, g = ev.true_gain_curve(s, true_eff)
            models.append(
                {
                    "model": name,
                    "label": POLICY_LABELS[name],
                    "qini_coefficient": round(float(ev.qini_coefficient(s, t, r)), 1),
                    "uplift_auc": round(ev.uplift_auc(s, t, r), 1),
                    "true_gain_auc": round(float(ev.area(xg, g) - g[-1] / 2), 1),
                    "valid_qini": round(float(self.valid_qini[name]), 1) if name in self.valid_qini else None,
                }
            )
        self.report["config"] = asdict(cfg)
        self.report["champion"] = self.champion
        self.report["data"] = self._data_summary()
        self.report["qini"] = {"curves": curves, "models": models, "treated_in_test": int(t.sum())}
        self.report["deciles"] = {
            name: ev.deciles(self.scores[name], t, r, true_eff) for name in ("churn_risk", self.champion)
        }
        self.report["policies"] = self.policy_table(cfg.target_share)
        self.report["budget"] = self.allocate(cfg.default_budget, with_lp=True)
        self.report["experiment"] = self._experiment()

    def _data_summary(self) -> dict:
        te = self.test
        g = te["true_group"].value_counts()
        return {
            "customers": self.cfg.n,
            "test_customers": len(te),
            "pilot_customers": self.cfg.pilot_n,
            "treated_share": round(float(te["treated"].mean()), 4),
            "churn_control": round(float(te.loc[te["treated"] == 0, "churned"].mean()), 4),
            "churn_treated": round(float(te.loc[te["treated"] == 1, "churned"].mean()), 4),
            "groups": {k: int(g.get(k, 0)) for k in GROUPS},
        }

    def policy_assign(self, name: str, share: float) -> np.ndarray:
        n = len(self.test)
        if name == "oracle":
            return np.where(self.test["true_group"].to_numpy() == "persuadable", 0, pol.NONE)
        if name not in self.scores:
            raise KeyError(name)
        return pol.top_k(self.scores[name], int(round(share * n)), offer=0)

    def evaluate_policy(self, name: str, share: float) -> dict:
        a = self.policy_assign(name, share)
        out = pol.evaluate(self.test, a, self.econ)
        out.update(policy=name, label=POLICY_LABELS[name], share=share)
        return out

    def policy_table(self, share: float) -> list[dict]:
        return [self.evaluate_policy(n, share) for n in ("churn_risk", "random", *UPLIFT_MODELS, "oracle")]

    # ---------- budget ----------

    def _prefix_within(self, score: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
        """Fixed top-k: walk down the ranking giving the discount until the budget runs out."""
        order = np.argsort(-score, kind="stable")
        k = int(np.searchsorted(np.cumsum(cost[order]), budget, side="right"))
        a = np.full(len(score), pol.NONE)
        a[order[:k]] = 0
        return a

    def allocate(self, budget: float, with_lp: bool = False) -> dict:
        te, econ, values = self.test, self.econ, self.values
        t0 = time.perf_counter()
        a_greedy = bud.greedy(values, econ["cost"], budget)
        greedy_ms = (time.perf_counter() - t0) * 1000
        disc_only = np.where(np.arange(3)[None, :] == 0, values, -np.inf)
        a_disc = bud.greedy(disc_only, econ["cost"], budget)
        rows = []
        plans = [
            ("greedy_multi", "Greedy, three offers, value per dollar", a_greedy),
            ("greedy_discount", "Greedy, discount only", a_disc),
            ("topk_uplift", f"Fixed top-k by {POLICY_LABELS[self.champion].split(': ')[1]}, discount",
             self._prefix_within(self.scores[self.champion], econ["cost"][:, 0], budget)),
            ("topk_churn", "Fixed top-k by churn risk, discount",
             self._prefix_within(self.scores["churn_risk"], econ["cost"][:, 0], budget)),
        ]
        for key, label, a in plans:
            row = pol.evaluate(te, a, econ)
            row.update(pol.expected(a, values, econ))
            row.update(policy=key, label=label)
            rows.append(row)
        out = {"budget": budget, "rows": rows, "greedy_ms": round(greedy_ms, 1), "lp_bound": None,
               "lp_available": bud.HAVE_SCIPY}
        if with_lp and bud.HAVE_SCIPY:
            bound = bud.lp_bound(values, econ["cost"], budget)
            out["lp_bound"] = round(bound, 2) if bound is not None else None
            if bound:
                out["greedy_share_of_lp"] = round(rows[0]["expected_net_value"] / bound, 4)
        return out

    # ---------- experiment ----------

    def _experiment(self) -> dict:
        cfg = self.cfg
        a = bud.greedy(self.values, self.econ["cost"], cfg.default_budget)
        idx = np.where(a != pol.NONE)[0]
        p_h = float(self.p0[idx].mean())
        p_t = float(self.pk[idx, a[idx]].mean())
        # Rollout: a fresh month of subscribers, same budget per customer, 20% of targeted held out.
        roll = generate(GenConfig(n=cfg.rollout_n, seed=cfg.seed + 200, design="binary"))
        Xr = matrix(roll)
        econ = offer_economics(roll["plan"].to_numpy(), roll["tenure_months"].to_numpy())
        p0, pk = self.offer_model.predict(Xr)
        values = pol.expected_values(p0, pk, econ)
        ar = bud.greedy(values, econ["cost"], cfg.budget_per_customer * cfg.rollout_n)
        tidx = np.where(ar != pol.NONE)[0]
        rng = np.random.default_rng(cfg.seed + 300)
        hold = rng.random(len(tidx)) < cfg.holdout_share
        y0, yk = pol.potential_outcomes(roll)
        hi, ti = tidx[hold], tidx[~hold]
        y_hold, y_treat = y0[hi], yk[ti, ar[ti]]
        v_hold = (1 - y_hold) * econ["clv0"][hi]
        v_treat = (1 - y_treat) * econ["clv_k"][ti, ar[ti]] - econ["cost"][ti, ar[ti]]
        rd = exp.readout(y_hold, y_treat, v_hold, v_treat)
        true_red = float((y0[ti] - yk[ti, ar[ti]]).mean())
        true_val = float((v_treat - (1 - y0[ti]) * econ["clv0"][ti]).mean())
        rd["true_churn_reduction"] = round(true_red, 5)
        rd["true_net_value_per_customer"] = round(true_val, 3)
        rd["ci_covers_truth"] = bool(rd["churn_reduction_ci"][0] <= true_red <= rd["churn_reduction_ci"][1])
        # Second design: does the policy beat churn risk targeting across the whole base? Randomize customers
        # 50/50 between the two policies at the same budget; the metric is overall churn, so the gap is small.
        n = len(self.test)
        churn_base = float(self.p0.mean())
        a_churn = self._prefix_within(self.scores["churn_risk"], self.econ["cost"][:, 0], cfg.default_budget)
        gain_policy = float((self.p0[idx] - self.pk[idx, a[idx]]).sum())
        cidx = np.where(a_churn != pol.NONE)[0]
        gain_churn = float((self.p0[cidx] - self.pk[cidx, 0]).sum())
        versus = exp.sample_size(churn_base - gain_churn / n, churn_base - gain_policy / n, 0.5)
        return {
            "versus_churn_risk": versus,
            "expected_churn_holdout": round(p_h, 4),
            "expected_churn_treated": round(p_t, 4),
            "targeted_per_month": int(len(tidx)),
            "plan": exp.plan_test(p_h, p_t, int(len(tidx))),
            "chosen_holdout_share": cfg.holdout_share,
            "power_at_rollout": round(exp.achieved_power(p_h, p_t, int(hold.sum()), int((~hold).sum())), 4),
            "rollout_customers": cfg.rollout_n,
            "readout": rd,
        }

    # ---------- scoring ----------

    def score(self, rows: pd.DataFrame) -> list[dict]:
        X = matrix(rows)
        churn = self.churn.predict_churn(X)
        champ = self.learners[self.champion].predict_uplift(X)
        p0, pk = self.offer_model.predict(X)
        econ = offer_economics(rows["plan"].to_numpy(), rows["tenure_months"].to_numpy())
        values = pol.expected_values(p0, pk, econ)
        out = []
        for i in range(len(rows)):
            offers = []
            for j, o in enumerate(OFFERS):
                ok = bool(econ["eligible"][i, j])
                offers.append(
                    {
                        "offer": o.key,
                        "label": o.label,
                        "eligible": ok,
                        "uplift": round(float(p0[i] - pk[i, j]), 4) if ok else None,
                        "cost": round(float(econ["cost"][i, j]), 2),
                        "expected_net_value": round(float(values[i, j]), 2) if ok else None,
                    }
                )
            out.append(
                {
                    "churn_risk": round(float(churn[i]), 4),
                    "discount_uplift_champion": round(float(champ[i]), 4),
                    "clv": round(float(econ["clv0"][i]), 2),
                    "offers": offers,
                    **recommend(float(churn[i]), offers),
                }
            )
        return out

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> Study:
        with path.open("rb") as f:
            return pickle.load(f)


def recommend(churn_risk: float, offers: list[dict]) -> dict:
    """Pick the offer with the highest expected net value, or explain why no offer is the right call."""
    ok = [o for o in offers if o["eligible"]]
    best = max(ok, key=lambda o: o["expected_net_value"])
    best_up = max(o["uplift"] for o in ok)
    if best["expected_net_value"] > 0:
        return {
            "action": best["offer"],
            "reason": (
                f"{best['label']}: expected to cut churn risk by {best['uplift'] * 100:.1f} points, worth "
                f"${best['expected_net_value']:.2f} after its ${best['cost']:.2f} cost."
            ),
            "likely_group": "persuadable",
        }
    if best_up < -0.01:
        return {
            "action": "none",
            "reason": f"No offer: every offer raises churn risk (best {best_up * 100:+.1f} points). "
            "Likely a sleeping dog, so leave them alone.",
            "likely_group": "sleeping_dog",
        }
    if churn_risk >= 0.4:
        return {
            "action": "none",
            "reason": f"No offer: churn risk is high ({churn_risk:.0%}) but no offer moves it enough "
            f"(best {best_up * 100:+.1f} points). Likely a lost cause.",
            "likely_group": "lost_cause",
        }
    if churn_risk < 0.15:
        return {
            "action": "none",
            "reason": f"No offer: churn risk is low ({churn_risk:.0%}), so they will probably stay anyway. "
            "Likely a sure thing.",
            "likely_group": "sure_thing",
        }
    return {
        "action": "none",
        "reason": f"No offer: the best one ({best['label']}) is expected to lose "
        f"${-best['expected_net_value']:.2f} after its cost.",
        "likely_group": "uncertain",
    }


def customer_frame(payload: dict) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    return pd.DataFrame([{f: payload[f] for f in FEATURES}])


def markdown_policy_table(rows: list[dict]) -> str:
    lines = [
        "| Policy | Targeted | Cost | Incremental retained | Net value | Persuadables | Lost causes | Sleeping dogs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        g = r["groups"]
        lines.append(
            f"| {r['label']} | {r['targeted']:,} | ${r['cost']:,.0f} | {r['incremental_retained']:,} | "
            f"${r['net_value']:,.0f} | {g['persuadable']:,} | {g['lost_cause']:,} | {g['sleeping_dog']:,} |"
        )
    return "\n".join(lines)
