"""Mechanical verification of springer_project/TabStress.tex.

The numbers in the manuscript are substituted from the analysis outputs at build time, so this
script checks what substitution cannot guarantee:

  build        the build is final (no unresolved token, no dropped %%IF block) and the log is clean
  counts       every experiment block is complete
  independent  key statistics recomputed from the primary cells by separate code (AUDC, MRR,
               mean ranks, Friedman and Iman-Davenport statistics, native deltas, the threshold
               gap, seed reliability)
  claims       every qualitative sentence of the text holds in the data
  harness      the extended harness reproduces stored seed-42 cells
  document     citations, bibliography integrity, figures, house style, stale claims
  originality  no 8-word run copied from the corpus abstracts or the mined gap sentences

Exit status is non-zero on any failure."""

import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_audit import TIE_DECIMALS

ROOT = "springer_project"
TEX = os.path.join(ROOT, "TabStress.tex")
SUPP = os.path.join(ROOT, "TabStress_Supplementary.tex")
REVISED_BIB = os.path.join(ROOT, "revision", "references_revised_seminal.bib")
LOG = os.path.join(ROOT, "TabStress.log")
PRIM = "results/analysis/audit_primary"
CELLS = "results/analysis/primary/cells.csv"
BY_SEED = "results/analysis/primary/cells_by_seed.csv"
REV = "results/analysis/revision/revision.json"
FOUND = "results/analysis/revision/foundation.json"
REV_METRICS = "results/metrics/revision"
TABICL_ENV = ".venv-tabpfn/lib/python3.12/site-packages"
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
NAME = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "catboost": "CatBoost", "mlp": "MLP"}
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
# per seed and dataset the base variant stores the clean fit once per model: 6 + 6 x 4 operators x 10 tiers
EXPECTED = {"base": 2 * 20 * (6 + 6 * 4 * 10), "cc_binary": 3 * 14 * 6 * 10, "native_nan": 3 * 20 * 5 * 10,
            "catboost_default": 20 * (1 + 4 * 10), "xgb_default": 20 * (1 + 4 * 10)}
BANNED = ["delve", "testament", "tapestry", "beacon", "pivotal", "furthermore", "moreover", "crucial",
          "vital", "landscape", "revolutioni", "game-changer", "plethora", "myriad", "underscore",
          "in summary", "it is worth noting", "sheds light", "foster", "notably", "in conclusion",
          "firstly", "secondly", "paradigm", "seamless", "realm", "leverag", "pristine benchmark bias"]

failures, passes = [], 0


def check(label, ok, detail=""):
    global passes
    if ok:
        passes += 1
    else:
        failures.append(f"{label}: {detail}")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- build
def check_build(tex):
    check("no unresolved token", "@@{" not in tex, re.findall(r"@@\{[^}]*\}@@", tex)[:3])
    dropped = open(os.path.join(ROOT, "build", "dropped_blocks.txt")).read().split() \
        if os.path.exists(os.path.join(ROOT, "build", "dropped_blocks.txt")) else ["missing dropped_blocks.txt"]
    check("final build (no %%IF block dropped)", not dropped, str(dropped))
    log = open(LOG).read() if os.path.exists(LOG) else ""
    check("build log present", bool(log))
    for pat in ["Overfull \\hbox", "undefined", "multiply defined", "! "]:
        check(f"log free of '{pat}'", pat not in log)


def check_original_patch():
    """The seed-42 Gaussian cells recomputed after the constant-column correction are the ones in the grid."""
    rec_path, log_path = "results/metrics/original_recomputed.csv", "results/metrics/original_patch_log.csv"
    check("original-grid patch log present", os.path.exists(log_path) and os.path.exists(rec_path))
    if not (os.path.exists(log_path) and os.path.exists(rec_path)):
        return
    rec = pd.read_csv(rec_path)
    rec["eta"] = rec.eta.round(2)
    grid = pd.read_csv("results/metrics/benchmark_results.csv")
    grid["eta"] = grid.eta.round(2)
    m = rec.merge(grid, on=["dataset", "model", "modality", "eta"], suffixes=("_rec", "_grid"))
    check("360 recomputed cells, all present in the grid", len(rec) == 360 and len(m) == 360, (len(rec), len(m)))
    worst = max(float((m[f"{c}_rec"] - m[f"{c}_grid"]).abs().max()) for c in ["mean_acc", "std_acc", "mean_bal_acc", "mean_f1"])
    check("grid holds the recomputed values", worst == 0.0, worst)


def check_counts(extra):
    c = extra["counts"]
    check("original grid has 5,280 cells", c["original_grid"] == 5280, c["original_grid"])
    for v, n in EXPECTED.items():
        got = c["extended_by_variant"].get(v, 0)
        check(f"extended block complete: {v}", got == n, f"{got}/{n}")
    check("three seeds in the primary grid", c["n_seeds"] == 3 and c["seeds_complete"] == [42, 43, 44], c["seeds_complete"])
    check("total cells = 22,280", c["total_cells"] == 5280 + sum(EXPECTED.values()) == 22280, c["total_cells"])
    check("fits = 5 x cells", c["total_fits"] == 5 * c["total_cells"], c["total_fits"])


# --------------------------------------------------------------------------- independent recomputation
def trapezoid_scores(g):
    g = g.sort_values("eta")
    x, y = g.eta.to_numpy(float), g.mean_acc.to_numpy(float)
    audc = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))
    return audc, audc / y[0], y[0], y[-1]


def check_independent(prim, extra):
    cells = pd.read_csv(CELLS)
    rows = []
    for (c, d, m), g in cells.groupby(["condition", "dataset", "model"]):
        audc, mrr, a0, a1 = trapezoid_scores(g)
        rows.append((c, d, m, audc, mrr, a0, a1))
    t = pd.DataFrame(rows, columns=["condition", "dataset", "model", "audc", "mrr", "acc0", "acc_end"])
    check("600 seed-averaged trajectories", len(t) == 600, len(t))
    worst = 0.0
    for metric in ["mrr", "audc"]:
        for c in CONDITIONS:
            pv = t[t.condition == c].pivot(index="dataset", columns="model", values=metric)[MODELS].round(TIE_DECIMALS)
            ranks = np.vstack([stats.rankdata(-row) for row in pv.to_numpy()])
            R = ranks.mean(axis=0)
            stored = prim["mean_ranks"][f"{metric}|{c}"]
            worst = max(worst, max(abs(R[i] - stored[m]) for i, m in enumerate(MODELS)))
            n, k = ranks.shape
            chi2 = 12 * n / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
            ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
            f = prim["friedman"][f"{metric}|{c}"]
            check(f"Friedman chi2 & F_ID recomputed ({metric}, {c})",
                  abs(chi2 - f["chi2_F"]) < 1e-8 and abs(ff - f["F_ID"]) < 1e-8, f"{chi2} vs {f['chi2_F']}")
    check("mean ranks recomputed independently (max |diff| < 1e-9)", worst < 1e-9, worst)

    nat = t[t.condition == "mcar_native"].set_index(["dataset", "model"])
    imp = t[t.condition == "mcar_imputed"].set_index(["dataset", "model"])
    for m in MODELS[:5]:
        d = (nat.xs(m, level="model").audc - imp.xs(m, level="model").audc)
        check(f"native median delta AUDC recomputed ({m})",
              abs(d.median() - extra["native"][m]["median_delta_audc"]) < 1e-12)

    inv = prim["inventory"]
    cc = t[(t.condition == "class_conditional")]
    imb = cc[cc.dataset.map(lambda x: inv[x]["K"] == 2 and inv[x]["majority_rate"] >= 0.85)]
    gap = (imb.acc_end - imb.dataset.map(lambda x: inv[x]["majority_rate"])).abs().max()
    check("threshold gap recomputed", abs(gap - extra["threshold"]["imbalanced_max_abs_gap"]) < 1e-12, gap)

    by = pd.read_csv(BY_SEED)
    srows = []
    for (s, c, d, m), g in by.groupby(["seed", "condition", "dataset", "model"]):
        audc, mrr, _, _ = trapezoid_scores(g)
        srows.append((s, c, d, m, audc, mrr))
    st = pd.DataFrame(srows, columns=["seed", "condition", "dataset", "model", "audc", "mrr"])
    w = st.pivot_table(index=["condition", "dataset", "model"], columns="seed", values="audc")
    rho = [stats.spearmanr(w[a], w[b]).statistic for a, b in itertools.combinations(sorted(w.columns), 2)]
    rel = extra["seeds"]["reliability"]["audc"]
    check("seed reliability of AUDC recomputed",
          abs(min(rho) - rel["spearman_min"]) < 1e-12 and abs(max(rho) - rel["spearman_max"]) < 1e-12,
          f"{min(rho)}..{max(rho)} vs {rel}")

    # Independent recomputation of split class-conditional kernel statistics and variance ratios
    bin_ds = [d for d in inv if inv[d]["K"] == 2]
    mc_ds = [d for d in inv if inv[d]["K"] >= 3]
    pv_bin = cc[cc.dataset.isin(bin_ds)].pivot(index="dataset", columns="model", values="mrr")[MODELS]
    chi2_bin, _ = stats.friedmanchisquare(*[pv_bin[m] for m in MODELS])
    check("independent binary Friedman chi2 recomputed", abs(chi2_bin - 19.63) < 0.05, f"{chi2_bin}")
    pv_mc = cc[cc.dataset.isin(mc_ds)].pivot(index="dataset", columns="model", values="mrr")[MODELS]
    chi2_mc, _ = stats.friedmanchisquare(*[pv_mc[m] for m in MODELS])
    check("independent multiclass Friedman chi2 recomputed", abs(chi2_mc - 23.90) < 0.05, f"{chi2_mc}")

    for c in CONDITIONS:
        g_c = t[t.condition == c]
        vr_val = float(g_c.groupby("dataset").mrr.mean().var() / g_c.groupby("model").mrr.mean().var())
        check(f"variance ratio independently recomputed ({c})",
              abs(vr_val - extra["variance_ratio"][c]) < 1e-9, f"{vr_val} vs {extra['variance_ratio'][c]}")
    return t


# --------------------------------------------------------------------------- qualitative claims
def claim(tex, phrase, ok, detail=""):
    """A sentence (identified by a phrase that must appear in the text) and the data predicate it asserts."""
    check(f"text has: '{phrase[:60]}'", phrase in tex)
    check(f"data support: '{phrase[:60]}'", bool(ok), detail)


def holm_adjust(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def check_ties(tex, prim, t):
    """Section 5: ties are rare, and the tie-corrected Iman-Davenport test reaches the same
    Holm-adjusted decisions as the uncorrected one in every block."""
    rows = tied = 0
    same = True
    n, k = 20, len(MODELS)
    for metric in ["mrr", "audc"]:
        p_tc = []
        for c in CONDITIONS:
            pv = t[t.condition == c].pivot(index="dataset", columns="model", values=metric)[MODELS]
            rows += len(pv)
            tied += int(sum(len(set(np.round(r, 12))) < len(r) for r in pv.to_numpy()))
            chi2 = stats.friedmanchisquare(*[pv[m] for m in MODELS]).statistic        # tie-corrected
            ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
            p_tc.append(stats.f.sf(ff, k - 1, (k - 1) * (n - 1)))
        for c, p_adj in zip(CONDITIONS, holm_adjust(np.array(p_tc))):
            same &= (p_adj < 0.05) == (prim["friedman"][f"{metric}|{c}"]["p_holm"] < 0.05)
    claim(tex, "Tied ranks are rare and the tie-corrected statistic gives the same decisions",
          tied / rows < 0.05 and same, (tied, rows, same))


def check_claims(tex, prim, extra, orig, t):
    """Qualitative sentences of the revised text that rest on the primary (Protocol A) grid."""
    rev = load_json(REV)
    cf, tests = rev["confirmatory"], rev["confirmatory"]["tests"]
    th, nat = extra["threshold"], extra["native"]
    L = prim["leaders"]
    claim(tex, "LightGBM has the best mean rank on balanced AUDC under",
          cf["leader_counts"]["audc_bal"].get("lightgbm") == 4 and tests["audc_bal|symmetric_label"]["best"] == "catboost",
          cf["leader_counts"]["audc_bal"])
    claim(tex, "and the MLP under class-conditional noise",
          tests["audc_bal|class_conditional"]["worst"] == "mlp" and cf["worst_counts"]["audc_bal"].get("cart") == 4)
    cvr = rev["clean_vs_robust"]
    claim(tex, "The answer to RQ1 is negative",
          cvr["same_best_share_mrr_overall"] < 0.2 and cvr["same_best_share_audc_bal_overall"] < 0.5, cvr)
    claim(tex, "is negative under every condition", cvr["tau_mrr_max"] < 0, cvr["tau_mrr_max"])
    claim(tex, "CatBoost leads on all three confirmatory scores",
          all(tests[f"{m}|symmetric_label"]["best"] == "catboost" and tests[f"{m}|symmetric_label"]["worst"] == "cart"
              for m in ["audc_bal", "audc", "mrr"]))
    claim(tex, "its lead is spent early in the sweep",
          prim["acc0_best_model"] == "LightGBM" and L["mrr|symmetric_label"]["best"] != "LightGBM")
    claim(tex, "The MLP, second on MRR", L["mrr|symmetric_label"]["order"][1] == "MLP", L["mrr|symmetric_label"]["order"])
    claim(tex, "under class-conditional label noise does not distinguish the families",
          tests["mrr_bal|class_conditional"]["p_holm_all"] > 0.05)
    vr = extra["variance_ratio"]
    claim(tex, "depends on the dataset far more than on the model",
          vr["class_conditional"] > 5 and vr["class_conditional"] > 5 * vr["symmetric_label"], vr)
    claim(tex, "whereas under symmetric noise the two coincide",
          th["median_acc_end_class_conditional"] - th["median_bal_end_class_conditional"] > 0.1
          and abs(th["median_acc_end_symmetric"] - th["median_bal_end_symmetric"]) < 0.03)
    ra = cf["rank_agreement"]["amrr_vs_mrr_tau"]
    claim(tex, "Under symmetric noise aMRR ranks the families exactly as MRR does and under Gaussian noise almost so",
          ra["symmetric_label"] > 0.999 and 0.95 < ra["gaussian_feature"] < 0.999, ra)
    gauss_first = [tests[f"{k}|gaussian_feature"]["ranks"]["mlp"] == min(tests[f"{k}|gaussian_feature"]["ranks"].values())
                   for k in ("mrr", "amrr", "amrr_bal")]
    claim(tex, "the MLP remaining first, or tied for first with CatBoost, so the MLP's lead in retention is not an artefact",
          all(gauss_first) and tests["mrr|gaussian_feature"]["best"] == "mlp", gauss_first)
    claim(tex, "lead disappears once the majority-class baseline is subtracted",
          tests["mrr|class_conditional"]["best"] == "mlp" and tests["amrr|class_conditional"]["best"] != "mlp")
    claim(tex, "reject the hypothesis of equal mean ranks after joint Holm adjustment",
          cf["confirmatory_n_rejected"] == cf["n_confirmatory"] == 15)
    not_sig = sorted(k for k, v in tests.items() if v["p_holm_all"] >= 0.05)
    claim(tex, "the exceptions are balanced MRR under class-conditional noise and imputed masking",
          not_sig == sorted(["mrr_bal|class_conditional", "mrr_bal|mcar_imputed", "amrr_bal|class_conditional",
                             "mrr|class_conditional"]), not_sig)
    ph = prim["post_hoc_counts"]
    claim(tex, "Wilcoxon is usually the more liberal",
          sum(v["wilcoxon"] for v in ph.values()) >= sum(v["nemenyi"] for v in ph.values()))
    kv = rev["knee_validation"]
    claim(tex, "Without noise the spline knee recovers every true knee to within $0.001$",
          max(kv["noiseless_spline_error"].values()) <= 0.001, kv["noiseless_spline_error"])
    claim(tex, "which uses the tiers directly, recovers all three within one tier",
          kv["discrete_within_one_tier_min"] > kv["spline_within_one_tier_min"])
    claim(tex, "cannot tell a knee from its absence", kv["no_knee_kappa_ratio"] > 0.5, kv["no_knee_kappa_ratio"])
    ks = rev["knee_seeds"]
    claim(tex, "the discrete estimator does better",
          ks["eta_discrete"]["spearman_min"] > ks["eta_knee"]["spearman_max"]
          and ks["eta_discrete"]["spearman_max"] < ks["mrr"]["spearman_min"])
    rr = orig["binary_sym_vs_asym_test_retest"]
    claim(tex, "MRR replicates almost perfectly", rr["mrr_pearson"] > 0.95 and rr["eta_spearman"] < 0.6)
    claim(tex, "AUDC and MRR are reliable robustness descriptors",
          ks["mrr"]["spearman_min"] > 0.95 and ks["audc_bal"]["spearman_min"] > 0.95)
    traj = pd.read_csv(os.path.join("results/analysis/revision", "trajectories_revision.csv"))
    base = rev["baselines"]
    zero = sorted({(r.dataset, r.model) for r in traj.itertuples() if r.acc0 - base[r.dataset]["pi_max"] < 0.01})
    expected = sorted([(d, m) for d in ["amazon_access", "ozone_level"] for m in MODELS] + [("bank-marketing", "cart")])
    claim(tex, "those of all six models on amazon\\_access and ozone\\_level and of CART on bank-marketing",
          zero == expected, zero)
    claim(tex, "The six families are \\emph{classical supervised tabular learners}, trained from scratch in every fold",
          not {"tabpfn", "tabicl", "torch"} & set(re.findall(r"[a-z]+", open("requirements.txt").read().lower())))
    acc = rev["accounting"]
    claim(tex, "it rests on", acc["total_cells"] == 22280 and acc["primary_distinct"] == 17760)
    claim(tex, "Four of the five point estimates favour imputation" if "Four of the five" in tex
          else "never significantly improves on median imputation", nat["summary"]["sig_native_better"] == 0)
    claim(tex, "CART has the lowest absolute endurance" if "CART has the lowest absolute endurance" in tex
          else "CART is last under", all(L[f"audc|{c}"]["worst"] == "CART" for c in CONDITIONS))
    if "rerun_audit" in rev:
        claim(tex, "These corrections were identified during independent pipeline validation and were applied before the final statistical analysis",
              rev["rerun_audit"]["max_abs_diff"] <= 1e-12, rev["rerun_audit"])
    check_revision_claims(tex, rev)


def check_revision_claims(tex, rev):
    """Sentences on Protocol B, the additional mechanisms, tuning and dataset artifacts."""
    for phrase, predicate in REVISION_CLAIMS:
        claim(tex, phrase, predicate(rev))


def _dep(rev):
    return rev["deploy"]


def _native_b(rev):
    return rev["deploy"]["native_vs_imputed"]


REVISION_CLAIMS = [
    ("and never significantly higher: a learner that is refitted on corrupted data adapts to it",
     lambda r: _dep(r)["summary"]["n_higher_sig"] == 0 and _dep(r)["summary"]["n_lower_sig"] >= 6),
    ("The median losses are small in absolute terms, about one point of accuracy",
     lambda r: all(-0.02 < v["median_delta"] < 0 for c in _dep(r)["comparison"].values() for k, v in c.items()
                   if k != "audc_bal")),
    ("Under Gaussian noise the MLP, whose decision function is smooth in its inputs, leads on balanced AUDC",
     lambda r: _dep(r)["tests"]["audc_bal|gaussian_feature"]["best"] == "mlp"),
    ("LightGBM keeps the lead on balanced AUDC under MCAR, MAR and MNAR alike",
     lambda r: all(_dep(r)["tests"][f"audc_bal|{m}_imputed"]["best"] == "lightgbm" for m in ["mcar", "mar", "mnar"])),
    ("native handling lowers the AUDC of all three boosters relative to median imputation",
     lambda r: _native_b(r)["summary"]["boosters_max_delta"] < 0 and _native_b(r)["summary"]["n_native_worse_sig"] >= 8),
    ("CART and the random forest, whose scikit-learn implementation sends missing values",
     lambda r: all(_native_b(r)[f"{m}|{x}"]["median_delta"] > 0 for m in ["cart", "randomforest"] for x in ["mcar", "mar", "mnar"])),
    ("the random forest leads on every score under every native-handling condition of Protocol~B",
     lambda r: all(v["best"] == "randomforest" for k, v in _dep(r)["tests"].items() if k.endswith("_native"))),
    ("and XGBoost or CatBoost is last on balanced AUDC,",
     lambda r: all(_dep(r)["tests"][f"audc_bal|{m}_native"]["worst"] in ("xgboost", "catboost")
                   for m in ["mcar", "mar", "mnar"])),
    ("whereas under Protocol~A native handling made no material difference",
     lambda r: load_json(os.path.join(PRIM, "extras.json"))["native"]["summary"]["max_abs_median_delta_audc"] < 0.01),
    ("The negative answer to RQ1 holds under deployment corruption",
     lambda r: max(_dep(r)["clean_vs_robust"].values()) == _dep(r)["clean_vs_robust"]["gaussian_feature"] <= 0.2),
    ("AUDC and MRR name different leading families under four of the five primary conditions",
     lambda r: sum(r["confirmatory"]["tests"][f"audc|{c}"]["best"] != r["confirmatory"]["tests"][f"mrr|{c}"]["best"]
                   for c in CONDITIONS) == 4),
    ("depends on the configuration (configuration robustness)",
     lambda r: r["tuned"]["complete"] and r["tuned"]["summary"]["same_best"] <= r["tuned"]["summary"]["n_blocks"] / 2),
    ("The endurance scores themselves move little",
     lambda r: r["tuned"]["summary"]["max_abs_median_delta"] < 0.01),
    ("The bottom of the ranking is more stable",
     lambda r: r["tuned"]["summary"]["n_cart_worst_both"] > r["tuned"]["summary"]["same_best"]),
    ("The principal conclusions are therefore not driven by the known dataset artifacts",
     lambda r: r["artifacts"]["cleaned_complete"] and r["artifacts"]["cleaned"]["tau_min"] > 0.8
     and r["artifacts"]["cleaned"]["n_sig_holm"] == 15 and r["artifacts"]["excluded"]["n_sig_holm"] >= 14),
    ("the families still differ on balanced AUDC under MAR and MNAR masking",
     lambda r: all(r["mechanisms"]["tests"][f"audc|{c}"]["best"] == "lightgbm"
                   and r["mechanisms"]["tests"][f"audc_bal|{c}"]["best"] == "lightgbm"
                   and r["mechanisms"]["tests"][f"audc|{c}"]["worst"] == r["mechanisms"]["tests"][f"mrr|{c}"]["worst"] == "cart"
                   and set(sorted(r["mechanisms"]["tests"][f"audc_bal|{c}"]["ranks"],
                                  key=r["mechanisms"]["tests"][f"audc_bal|{c}"]["ranks"].get)[-2:]) == {"cart", "mlp"}
                   for c in r["mechanisms"]["conditions"]) and r["mechanisms"]["all_planned_done"]
     and r["mechanisms"]["tests"]["mrr|mar_imputed"]["best"] == "catboost"
     and set(sorted(r["mechanisms"]["tests"]["mrr|mnar_imputed"]["ranks"],
                    key=r["mechanisms"]["tests"]["mrr|mnar_imputed"]["ranks"].get)[:2]) == {"xgboost", "catboost"}
     and abs(r["mechanisms"]["tests"]["mrr|mnar_imputed"]["ranks"]["xgboost"]
             - r["mechanisms"]["tests"]["mrr|mnar_imputed"]["ranks"]["catboost"]) < 0.25
     and r["confirmatory"]["tests"]["audc|mcar_imputed"]["best"] == "lightgbm"
     and r["confirmatory"]["tests"]["mrr|mcar_imputed"]["best"] == "catboost"),
    ("on all twenty datasets, with seed",
     lambda r: r["mechanisms"]["n_datasets"] == 20 and set(r["mechanisms"]["seeds"]) <= {42, 43, 44}
     and r["mechanisms"]["n_seeds"] == len(r["mechanisms"]["seeds"]) >= 1
     and 42 in r["mechanisms"]["seeds"]),
    ("Informative missingness costs more than random missingness at the same mean rate",
     lambda r: r["mechanisms"]["audc_bal_median_a"]["mnar_imputed"] < r["mechanisms"]["audc_bal_median_a"]["mar_imputed"]),
    ("from weak agreement among datasets about the order of the families (MRR under class-conditional noise)",
     lambda r: min(r["confirmatory"]["tests"].items(), key=lambda kv: kv[1]["W"] if kv[0].split("|")[0] in ("audc_bal", "audc", "mrr") else 9)[0] == "mrr|class_conditional"
     and max(((k, v) for k, v in r["confirmatory"]["tests"].items() if k.split("|")[0] in ("audc_bal", "audc", "mrr")), key=lambda kv: kv[1]["W"])[0] == "audc|mcar_native"),
    ("Replacing nominal values by valid but wrong categories costs slightly more than Gaussian noise on their codes",
     lambda r: r["categorical"]["A"]["cat_aware_feature"]["vs_gaussian_median_delta"] < 0
     and r["categorical"]["A"]["cat_aware_feature"]["vs_gaussian_p"] < 0.05
     and r["categorical"]["B"]["cat_aware_feature"]["vs_gaussian_p"] >= 0.05),
    ("Corrupting the nominal attributes alone costs little",
     lambda r: all(r["categorical"][p]["cat_only_feature"]["terminal_loss_median"] < 0.03 for p in "AB")),
    ("LightGBM has the highest mean balanced AUDC under every categorical condition",
     lambda r: all(r["categorical"][p][o]["best_audc_bal"] == "lightgbm" for p in "AB"
                   for o in ["cat_aware_feature", "cat_only_feature"])),
    ("and only about half the time the one with the highest balanced AUDC",
     lambda r: 0.35 <= r["clean_vs_robust"]["same_best_share_audc_bal_overall"] <= 0.6),
]


# --------------------------------------------------------------------------- pretrained foundation model (TabICL)
def _fb(f, score, cond):
    return f["blocks"][f"{score}|{cond}"]


FOUNDATION_CLAIMS = [
    ("TabICL, most accurate on clean data, is among the two least retentive models under every deployment corruption",
     lambda f: f["clean_accuracy"]["mean_acc"]["tabicl"] == max(f["clean_accuracy"]["mean_acc"].values())
     and f["headline"]["mrr"]["position_min"] >= 6),
    ("TabICL has the highest mean clean accuracy (",
     lambda f: f["clean_accuracy"]["mean_acc"]["tabicl"] == max(f["clean_accuracy"]["mean_acc"].values())),
    ("and the best mean clean rank among the seven (",
     lambda f: f["clean_accuracy"]["mean_rank"]["tabicl"] == min(f["clean_accuracy"]["mean_rank"].values())),
    ("and it is the most accurate clean model on",
     lambda f: f["clean_accuracy"]["tabicl_best_count"] == max(
         int((pd.DataFrame(f["clean_accuracy"]["per_dataset"]).T.idxmax(axis=1) == m).sum()) for m in f["pool"])),
    ("Under MCAR and MAR masking TabICL ranks first or second of the seven on both balanced AUDC and AUDC",
     lambda f: all(_fb(f, sc, c)["tabicl_position"] <= 2 for sc in ("audc_bal", "audc")
                   for c in ("mcar_imputed", "mar_imputed"))),
    ("in absolute value under both mechanisms on both scores",
     lambda f: f["headline"]["masking_abs_gap_max"] == max(abs(f["gap"][c][m]["median_gap"])
                                                            for c in ("mcar_imputed", "mar_imputed")
                                                            for m in ("audc_bal", "audc"))
     and f["headline"]["masking_abs_gap_max"] < 0.01),
    ("Under MNAR masking classical families lead both scores, LightGBM on balanced AUDC",
     lambda f: _fb(f, "audc_bal", "mnar_imputed")["leader7"] == "lightgbm"
     and _fb(f, "audc", "mnar_imputed")["leader7"] != "tabicl"),
    ("under Gaussian feature noise TabICL falls to mean rank",
     lambda f: _fb(f, "audc_bal", "gaussian_feature")["tabicl_position"] >= 4
     and _fb(f, "audc_bal", "gaussian_feature")["leader7"] == "mlp"),
    ("TabICL ranks sixth or seventh on MRR under every operator",
     lambda f: f["headline"]["mrr"]["position_min"] >= 6 and f["headline"]["mrr"]["n_tabicl_last"] >= 1),
    ("Adding it to the pool leaves the order of the six classical families essentially unchanged",
     lambda f: f["conclusion_changes"]["tau_min"] > 0.8),
    ("The negative answer to RQ1 also stands with seven models",
     lambda f: f["clean_vs_robust_7"]["mean"]["mrr"] <= 0.2 and f["clean_vs_robust_7"]["mean"]["audc_bal"] <= 0.6),
    ("TabICL starts from the highest clean accuracy and ranks first or second on the absolute scores",
     lambda f: all(f["gap"][c]["mrr"]["median_gap"] < 0 for c in f["gap"])
     and f["clean_accuracy"]["median_acc"]["tabicl"] > max(v for m, v in f["clean_accuracy"]["median_acc"].items()
                                                           if m != "tabicl")),
    ("TabICL starts from the highest clean accuracy and ranks first or second on the absolute scores under MCAR and MAR masking, yet it ranks sixth or seventh of seven on retention",
     lambda f: f["headline"]["clean_mean_acc_highest"] and f["headline"]["mrr"]["position_min"] >= 6
     and all(_fb(f, sc, c)["tabicl_position"] <= 2 for sc in ("audc_bal", "audc")
             for c in ("mcar_imputed", "mar_imputed"))),
    ("One tabular foundation model, TabICL, is covered as a secondary analysis under Protocol~B only",
     lambda f: len(f["pool"]) == 7 and f["pool"][-1] == "tabicl"),
    ("Bioresponse, with 1,776 columns, is excluded because the column-wise embedding does not fit in memory",
     lambda f: f["excluded_datasets"] == ["Bioresponse"] and "Bioresponse" not in f["datasets"]
     and f["n_datasets"] == 19 and "1,776 features" in open(os.path.join(REV_METRICS, "foundation_failures.txt")).read()),
    ("it was not run under corruption-aware training, and no other foundation model was run",
     lambda f: not any("tabicl" in open(os.path.join(REV_METRICS, n)).read() for n in os.listdir(REV_METRICS)
                       if n.endswith(".csv") and not n.startswith(("foundation__", "foundation_default__")))),
    ("is the TabICLv2 release",
     lambda f: "tabicl-classifier-v2-20260212.ckpt" in open(os.path.join(
         TABICL_ENV, "tabicl", "_sklearn", "classifier.py")).read()),
    ("median imputation fitted on the clean training fold before the model, so that TabICL never receives a missing value",
     lambda f: 'SimpleImputer(strategy="median")' in open("experiments/run_foundation.py").read()),
    ("a stratified subsample of the training fold, seeded per seed and fold",
     lambda f: "StratifiedShuffleSplit" in open("experiments/run_foundation.py").read()
     and "random_state=seed * 1_000 + fold" in open("experiments/run_foundation.py").read()),
]


def check_foundation(tex):
    """TabICL: the analysed seeds are complete, the stored statistics reproduce from the trajectories, and every
    qualitative sentence about the model is supported by foundation.json."""
    if not os.path.exists(FOUND):
        check("foundation.json present", False, FOUND)
        return
    f = load_json(FOUND)
    check("foundation settings in the analysis equal the run's settings file",
          f["settings"] == load_json(os.path.join(REV_METRICS, "foundation_settings.json")))
    for pkg_dir, ver in [("tabicl", f["settings"]["tabicl_version"]), ("torch", f["settings"]["torch"])]:
        dist = [d for d in os.listdir(TABICL_ENV) if d.startswith(pkg_dir + "-") and d.endswith(".dist-info")]
        check(f"{pkg_dir} version in the TabICL environment matches the text token", dist == [f"{pkg_dir}-{ver}.dist-info"], dist)
    raw = pd.concat([pd.read_csv(os.path.join(REV_METRICS, n)) for n in os.listdir(REV_METRICS)
                     if n.startswith("foundation__")], ignore_index=True)
    raw = raw[~raw.dataset.isin(f["excluded_datasets"])].drop_duplicates(["seed", "dataset", "modality", "eta"])
    planned = f["n_datasets"] * 4 * 11
    per_seed = raw.groupby("seed").size()
    check("the analysed TabICL seeds are exactly the complete seeds",
          sorted(int(x) for x in per_seed[per_seed == planned].index) == f["seeds"], f["seeds"])
    check("TabICL row and fit counts", f["accounting"]["rows"] == len(f["seeds"]) * planned
          and f["accounting"]["fits"] == 5 * len(f["seeds"]) * f["n_datasets"], f["accounting"])
    traj = pd.read_csv("results/analysis/revision/trajectories_foundation.csv")
    t = {"audc_bal|mcar_imputed": None, "mrr|gaussian_feature": None}
    for key in t:
        metric, cond = key.split("|")
        pv = traj[traj.condition == cond].pivot(index="dataset", columns="model", values=metric)
        ranks = (-pv).rank(axis=1).mean()
        check(f"TabICL mean rank reproduces from the trajectories ({key})",
              abs(ranks["tabicl"] - f["blocks"][key]["tabicl_rank"]) < 1e-9, ranks["tabicl"])
    check("Holm-adjusted p-values of the seven-model block are valid and the block is the twelve tests",
          f["conclusion_changes"]["n_blocks"] == 12
          and all(0 <= b["p_holm"] <= 1 and b["p_holm"] >= b["p"] for b in f["blocks"].values()))
    for phrase, predicate in FOUNDATION_CLAIMS:
        try:
            ok = predicate(f)
        except Exception as exc:                      # a claim whose data are missing fails, it does not crash
            ok, phrase = False, f"{phrase} [{exc!r}]"
        claim(tex, phrase.split(" [")[0], ok, phrase)
    check("tab_accounting lists the TabICL evaluations",
          "TabICL (B)" in open("results/analysis/revision/tab_accounting.tex").read())
    check("TabICL table is wired into Section 5.8", "tab:foundation" in tex)


SENS = "results/analysis/revision/sensitivity_extra.json"
SENS_TABLE = "results/analysis/revision/tab_sensitivity_extra.tex"


def _independent_compound_loss(op, parts, datasets):
    """Median over dataset-model pairs of compound loss minus the two single losses (accuracy AUDC), recomputed
    from the raw files with the trapezoid rule."""
    seeds = [42, 43, 44]
    comp = pd.concat([pd.read_csv(os.path.join(REV_METRICS, f"compound__seed{s}.csv")) for s in seeds])
    comp = comp[(comp.modality == op) & (comp.eta > 0)]
    dep = pd.concat([pd.read_csv(os.path.join(REV_METRICS, n)) for n in os.listdir(REV_METRICS)
                     if n.startswith("deploy__")])
    clean = dep[dep.seed.isin(seeds) & (dep.pipeline == "imputed") & (dep.eta == 0) & (dep.modality == "gaussian_feature")]
    clean = clean.drop_duplicates(["seed", "dataset", "model"])[["seed", "dataset", "model", "eta", "mean_acc"]]
    comp = pd.concat([comp[["seed", "dataset", "model", "eta", "mean_acc"]],
                      clean[clean.dataset.isin(datasets) & clean.model.isin(comp.model.unique())]])
    by_seed = pd.read_csv("results/analysis/primary/cells_by_seed.csv")
    loss = {}
    for name, g in [(op, comp.groupby(["dataset", "model", "eta"], as_index=False).mean_acc.mean())] + [
            (c, by_seed[by_seed.seed.isin(seeds) & (by_seed.condition == c) & by_seed.dataset.isin(datasets)]
             .groupby(["dataset", "model", "eta"], as_index=False).mean_acc.mean()) for c in parts]:
        loss[name] = {k: y.sort_values("eta").pipe(lambda d: d.mean_acc.iloc[0] - trapezoid_scores(d)[0])
                      for k, y in g.groupby(["dataset", "model"])}
    keys = loss[op].keys()
    return float(np.median([loss[op][k] - loss[parts[0]][k] - loss[parts[1]][k] for k in keys]))


def check_sensitivity(tex):
    """Sensitivity runs and extra statistics (supplement Section S9, three sentences in the article)."""
    sens = load_json(SENS)
    supp = open(SUPP).read() if os.path.exists(SUPP) else ""
    t, c, b, g = sens["tabicl_settings"], sens["compound"], sens["bayes"], sens["gap_ci"]
    default_settings = load_json(os.path.join(REV_METRICS, "foundation_default_settings.json"))
    check("default-settings run used eight estimators and no row cap",
          default_settings["n_estimators"] == 8 and default_settings["max_train"] is None, default_settings)
    check("supplement Section S9 is generated and wired", "tab:sens_settings" in supp and "tab:sens_bayes" in supp
          and "sup:sens" in supp and os.path.exists(SENS_TABLE))
    check("S9 cites Benavoli et al. for the Bayesian signed-rank test", "cite{benavoli2016should}" in supp)
    claim(tex, "The MRR rank of TabICL and its clean accuracy",
          all(abs(v["main"]["tabicl_rank"] - v["default"]["tabicl_rank"]) < 1e-12
              for k, v in t["ranks"].items() if k.startswith("mrr")), t["ranks"])
    claim(tex, "its balanced-AUDC rank under MNAR masking rises from",
          t["ranks"]["audc_bal|mnar_imputed"]["default"]["tabicl_rank"] < t["ranks"]["audc_bal|mnar_imputed"]["main"]["tabicl_rank"])
    claim(tex, "so the absolute scores of the main run are conservative for TabICL",
          all(v["default"]["tabicl_rank"] <= v["main"]["tabicl_rank"] + 1e-12 for k, v in t["ranks"].items()
              if k.startswith("audc_bal")))
    claim(tex, "The MRR rank of TabICL and its clean accuracy (within", t["clean"]["max_abs_diff"] < 0.02, t["clean"]["max_abs_diff"])
    claim(tex, "lie below zero under every operator", g["all_upper_below_zero"] and all(
        v["lo"] <= v["median"] <= v["hi"] for v in g["conditions"].values()), g)
    claim(tex, "lost less accuracy than the two single operators together",
          all(v < 0 for v in c["median_additivity_acc"].values()), c["median_additivity_acc"])
    claim(tex, "and kept the MRR order of the noise component",
          c["tau_mrr_dominant_min"] > 0.9 and c["tau_mrr_mcar_min"] < c["tau_mrr_dominant_min"],
          (c["tau_mrr_dominant_min"], c["tau_mrr_mcar_min"]))
    for op, parts in {"label_mcar": ["symmetric_label", "mcar_imputed"], "gauss_mcar": ["gaussian_feature", "mcar_imputed"]}.items():
        ind = _independent_compound_loss(op, parts, c["datasets"])
        check(f"compound additivity reproduces from the raw files ({op})",
              abs(ind - c["ops"][op]["additivity"]["acc"]["median"]) < 1e-9, ind)
    check("compound block is complete (2 operators x 6 models x 6 datasets x 10 tiers x 3 seeds)",
          sum(len(pd.read_csv(os.path.join(REV_METRICS, f"compound__seed{s}.csv"))) for s in c["seeds"]) == 2160)
    check("Bayesian posterior probabilities sum to one", all(abs(v["p_left"] + v["p_rope"] + v["p_right"] - 1) < 1e-9
                                                              and abs(v["decision_left"] + v["decision_rope"] + v["decision_right"] - 1) < 1e-9
                                                              for v in b["conditions"].values()))
    check("S9: the best family never wins the Bayesian decision with probability above 0.95",
          b["n_best_prob_over_95"] == 0 and b["max_decision_best"] < 0.95, b["max_decision_best"])
    check("S9: practical equivalence is the likeliest decision in the number of conditions stated",
          f"practical-equivalence region has the highest posterior probability in {['zero','one','two','three','four','five'][b['n_rope_decision_largest']]} of the five" in supp,
          b["n_rope_decision_largest"])
    check("S9: median gap between the first two families is below one percentage point under every condition",
          all(abs(v["median_diff"]) < 0.01 for v in b["conditions"].values()))
    check("S9: improves-or-equal balanced-AUDC statement and ROPE sentence are in the supplement",
          "improves or stays equal under every operator" in supp and "ROPE $\\pm0.01$" in supp)
    check("S9: every bootstrap interval lies below zero as the supplement states", g["all_upper_below_zero"])
    check("S9: compound loss sub-additive in at least the stated share of pairs",
          f"in at least {round(100 * c['sub_additive_share_min']):.0f}" in supp, c["sub_additive_share_min"])


# --------------------------------------------------------------------------- corruption engine
def check_principal_gaussian_substitution():
    """The primary Gaussian cells are the continuous-only cells on the six nominal-attribute datasets (same
    dataset, model, seed and eta), the all-column cells elsewhere, and the clean fits are shared."""
    from gaussian_principal import cat_datasets, load_cont
    by = pd.read_csv("results/analysis/primary/cells_by_seed.csv")
    g = by[(by.condition == "gaussian_feature") & by.dataset.isin(cat_datasets()) & (by.eta > 0)]
    cont = load_cont("train_cont").rename(columns={"mean_acc": "acc_cont"})
    m = g.merge(cont[["seed", "dataset", "model", "eta", "acc_cont"]], on=["seed", "dataset", "model", "eta"])
    check("principal Gaussian cells on the nominal-attribute datasets are the continuous-only cells",
          len(m) == len(g) == 6 * 6 * 10 * 3 and float((m.mean_acc - m.acc_cont).abs().max()) < 1e-12, len(m))
    allcol = pd.read_csv("results/analysis/primary/cells_gaussian_allcol_by_seed.csv")
    d = g.merge(allcol, on=["seed", "dataset", "model", "eta"], suffixes=("", "_all"))
    check("all-column stress-test cells kept apart and different from the principal ones",
          len(d) == len(g) and float((d.mean_acc - d.mean_acc_all).abs().max()) > 0.01, len(d))


def check_operators(tex):
    """The operator audit (experiments/operator_audit.py) and the sentences that report it."""
    ops = load_json(os.path.join(PRIM, "operator_audit.json"))
    check("operator audit passes (rates, targets, uniformity, flat blocks)", ops["all_pass"])
    fb, nest = ops["flat_blocks"], ops["nesting"]
    claim(tex, "which is what an operator that corrupted nothing would produce",
          fb is not None and fb["suspect"] == [] and fb["all_at_majority_rate"] == [] and fb["n_blocks"] == 300
          and sorted(fb["no_continuous_attributes"]) == [f"gaussian_feature/amazon_access/seed{s}" for s in (42, 43, 44)], fb)
    claim(tex, "there the corrupted examples are coupled in the first fold only, and their new labels in none",
          nest["symmetric_multiclass_first_fold_set"] == 1.0 and nest["symmetric_multiclass_later_folds_set"] == 0.0
          and nest["symmetric_multiclass_first_fold"] == 0.0 and nest["symmetric_multiclass_later_folds"] == 0.0, nest)
    claim(tex, "The coupling of the tiers described above was confirmed fold by fold",
          all(nest[k] == 1.0 for k in ["symmetric_binary_all_folds", "pairflip", "minority", "gaussian", "mcar"]), nest)
    claim(tex, "every pair-flip moves a label to the successor class", ops["pairflip"]["wrong_target"] == 0)
    claim(tex, "minority-to-majority noise never alters the other class", ops["minority"]["other_class_touched"] == 0)
    claim(tex, "on the two exactly balanced datasets, madelon and hill\\_valley",
          ops["minority"]["tie_datasets"] == ["hill_valley", "madelon"], ops["minority"]["tie_datasets"])
    claim(tex, "a column that is constant there is left unchanged",
          all(abs(r) < 1e-12 for r in ops["gaussian"]["constant_noise_sd_over_eta"]),
          ops["gaussian"]["constant_noise_sd_over_eta"][:3])


# --------------------------------------------------------------------------- harness reproduction
def check_harness():
    sys.path.insert(0, os.path.dirname(__file__))
    from run_extended import run_cell
    b = pd.read_csv("results/metrics/benchmark_results.csv")
    cells = [("vehicle", "cart", "symmetric_label", 0.2), ("phoneme", "randomforest", "mcar_missing", 0.3),
             ("churn", "xgboost", "gaussian_feature", 0.1), ("qsar-biodeg", "lightgbm", "asymmetric_label", 0.4),
             ("spambase", "catboost", "mcar_missing", 0.2), ("vehicle", "mlp", "symmetric_label", 0.3)]
    worst = 0.0
    for d, m, mod, e in cells:
        r = run_cell(("base", 42, d, m, mod, e))
        s = b[(b.dataset == d) & (b.model == m) & (b.modality == mod) & (b.eta.round(2) == e)].mean_acc.iloc[0]
        worst = max(worst, abs(r["mean_acc"] - s))
    check("extended harness reproduces seed-42 cells (one per family) within 1e-15", worst <= 1e-15, worst)


# --------------------------------------------------------------------------- document
def entries(path):
    out = {}
    for chunk in re.split(r"(?=^@)", open(path).read(), flags=re.M):
        m = re.match(r"@\w+\{([^,]+),", chunk.strip())
        if m:
            out.setdefault(m.group(1).strip(), chunk.strip())
    return out


def authors(entry):
    m = re.search(r"author=\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", entry)
    return m.group(1) if m else ""


def check_structure(body, supp_body):
    """Reviewer-requested layout: every appendix follows the reference list, the derivations sit in
    Appendix A of the article, and the supplement points to them instead of repeating them."""
    bib, app = body.find("\\bibliography{"), body.find("\\begin{appendices}")
    check("appendices come after the bibliography", 0 <= bib < app, (bib, app))
    appendix = body[app:] if app >= 0 else ""
    sections = re.findall(r"\\section\{([^}]*)\}\\label\{([^}]*)\}", appendix)
    check("the only appendix is Appendix A, the derivations",
          sections == [("Mathematical derivations", "app:math")], sections)
    labels = re.findall(r"\\begin\{appproposition\}[^\n]*\\label\{(prop:a_[a-z_]+)\}", appendix)
    check("Appendix A holds the four label-noise propositions", len(set(labels)) == 4, labels)
    check("Appendix A carries a proof for each proposition", appendix.count("\\begin{proof}") == 4, appendix.count("\\begin{proof}"))
    main_text = body[:app] if app >= 0 else body
    check("main text cites the interior-knee propositions in the supplement",
          "Supplementary Proposition~S1" in main_text and "Supplementary Proposition~S2" in main_text)
    cited = set(re.findall(r"\\ref\{(prop:a_[a-z_]+)\}", main_text))
    check("main text refers to Appendix A propositions by \\ref and every target exists",
          bool(cited) and cited <= set(labels), cited - set(labels))
    check("supplement states its propositions", supp_body.count("\\begin{supproposition}") == 4, supp_body.count("\\begin{supproposition}"))
    sup_head = re.findall(r"\\section\{([^}]*)\}", supp_body)
    check("supplement keeps ten sections, S1 first, the Wilcoxon tests in S8, sensitivity in S9, cleaning in S10",
          len(sup_head) == 10 and sup_head[0].startswith("Properties") and sup_head[7] == "Pairwise Wilcoxon tests"
          and sup_head[8].startswith("Sensitivity runs") and sup_head[9].startswith("Data cleaning"), sup_head)
    check("main text points to Supplementary Section S8 for the Wilcoxon tests",
          "Wilcoxon tests in Supplementary Section~S8" in main_text)
    check_pdf_structure()


def check_pdf_structure():
    """The typeset article: Appendix A follows the reference list, and the whole PDF stays under the
    reviewer's 45-page limit."""
    pdf = os.path.join(ROOT, "TabStress.pdf")
    try:
        text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        check("pdftotext available for the appendix-position check", False, str(err))
        return
    refs = text.find("\nReferences")
    head_a = re.search(r"^Appendix A\s+Mathematical derivations", text, re.M)
    check("typeset article: the 'Appendix A' heading follows the reference list",
          bool(head_a) and 0 <= refs < head_a.start(), (refs, bool(head_a)))
    n_pages = text.count("\f")
    check("typeset article has fewer than 45 pages", 0 < n_pages < 45, n_pages)


def check_document(tex):
    body = re.sub(r"(?m)^\s*%.*$", "", tex)
    supp_body = re.sub(r"(?m)^\s*%.*$", "", open(SUPP).read()) if os.path.exists(SUPP) else ""
    sem_e, cor_e = entries("paper/references.bib"), entries("literature/papers_2025_2026.bib")
    rev_e = entries(REVISED_BIB) if os.path.exists(REVISED_BIB) else {}
    sem_e = {**sem_e, **rev_e}
    out_e = entries(os.path.join(ROOT, "TabStress.bib"))
    cited = {k.strip() for g in re.findall(r"\\cite[pt]?\{([^}]*)\}", body + supp_body) for k in g.split(",")}
    check("every citation from corpus or seminal set", cited <= (set(sem_e) | set(cor_e)), str(cited - set(sem_e) - set(cor_e)))
    check("bibliography holds exactly the cited keys", set(out_e) == cited, str(set(out_e) ^ cited))
    bad = []
    for k, e in out_e.items():
        if k in rev_e:
            continue
        if k in sem_e and authors(e) != authors(sem_e[k]):
            bad.append(k)
        elif k not in sem_e and k in cor_e:
            if len(authors(e).split(" and ")) != len([a for a in authors(cor_e[k]).split(",") if a.strip()]):
                bad.append(k)
    check("author lists preserved", not bad, str(bad))
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body + supp_body)
    main_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    check("main text keeps at most six figures", len(set(main_figs)) <= 6, main_figs)
    missing = [f for f in figs if not os.path.exists(os.path.join(ROOT, "figures", f))]
    check("all figures exist", not missing, str(missing))
    check("all 20 seed-averaged degradation figures embedded", len({f for f in figs if f.startswith("curves_")}) == 20)
    check("10 critical-difference diagrams embedded",
          len({f for f in figs if re.match(r"cd_(mrr|audc)_", f)}) == 10)
    check("superseded seed-42 figures not embedded",
          not [f for f in figs if f.startswith("degradation_curve_") or f.startswith("cd_diagram_")])
    check("single file (no \\input)", not re.search(r"^\s*\\input\{", body + supp_body, re.M))
    check_structure(body, supp_body)
    check("no markdown rules", not re.search(r"^---\s*$", body, re.M))
    prose = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", body, flags=re.S)
    check("no raw | in math (program.sty)", not re.search(r"\$[^$]*(?<![-\\])\|(?![-])[^$]*\$", prose))
    outside = [m.group(0) for line in body.splitlines() for m in re.finditer(r"\\times10\^\{-?\d+\}", line)
               if line[:m.start()].count("$") % 2 == 0]
    check("scientific notation only inside math mode", not outside, str(outside[:3]))
    hyphen_minus = [m.group(0) for line in body.splitlines() if not line.lstrip().startswith("\\")
                    for m in re.finditer(r"(?<![\w$-])-\d+\.\d+", line) if line[:m.start()].count("$") % 2 == 0]
    check("negative numbers typeset as math minus", not hyphen_minus, str(hyphen_minus[:3]))
    low = body.lower()
    hits = [w for w in BANNED if w in low]
    check("banned vocabulary absent", not hits, str(hits))
    for stale in ["inflection boundary", "39,600", "zero data corruption", "Pristine Benchmark Bias",
                  "chosen from the suites of Grinsztajn", "FICO", "MAR and MNAR operators (fourteen", "all but volkert, Bioresponse", "\\in [0, 1]$ (retention"]:
        check(f"stale claim absent: {stale}", stale not in tex)
    template_vocab = [
        "quantum", "abilene", "geant", "germany50", "simulated annealing",
        "allele", "genetic behaviour", "genetic behavior", "network reliability",
        "nodes vs reliability", "nodes vs. reliability", "working example of noiserank",
        "working example of tabstress", "breast-w", "credit-g", "heart-statlog", "diabetes", "blood-transfusion"
    ]
    template_hits = [w for w in template_vocab if w in low]
    check("banned template vocabulary absent", not template_hits, str(template_hits))
    for stale in ["zenodo.10842918", "largest controlled tabular corruption benchmark", "first systematic",
                  "systematic meta-analysis", "Analytical Breakdown Point", "directly provides empirical evidence",
                  "isolated, ad-hoc noise levels", "Real-Time Implementation", "utilized solely for grammar",
                  "breakdown threshold", "None treats the shape", "2{,}773", "Practical Guidelines",
                  "provably interior", "Results Analysis and Discussions", "Non-Parametric Algorithmic Breakdown"]:
        check(f"retracted phrase absent: {stale}", stale not in tex)
    nat = json.load(open("results/analysis/audit_primary/extras.json"))["native"]
    worse = [m for m in ("cart", "randomforest", "xgboost", "lightgbm", "catboost")
             if nat[m]["p_audc_holm"] < 0.05 and nat[m]["median_delta_audc"] < 0]
    better = [m for m in ("cart", "randomforest", "xgboost", "lightgbm", "catboost")
              if nat[m]["p_audc_holm"] < 0.05 and nat[m]["median_delta_audc"] > 0]
    check("native: no family significantly better, CART and CatBoost significantly worse",
          not better and worse == ["cart", "catboost"], f"better={better} worse={worse}")
    deg = json.load(open("results/analysis/audit_primary/fix_extras.json"))["degenerate"]
    check("degenerate predictors confined to amazon_access and disclosed",
          deg["datasets"] == ["amazon_access"] and "amazon\\_access, on which two models are degenerate" in body, deg["pairs"])
    check_principal_gaussian_substitution()
    flat = json.load(open("results/analysis/audit_primary/audit_numbers.json"))["flat"]
    check("flat trajectories confined to amazon_access; all six families flat under Gaussian noise (no continuous attribute)",
          flat["datasets"] == ["amazon_access"] and flat["count"] == 9 and "every family under Gaussian noise" in body, flat)
    check("MRR leaders unchanged without the degenerate dataset", deg["leader_changes"] == 0, deg["leader_changes"])
    inv = json.load(open("results/analysis/audit_primary/audit_numbers.json"))["inventory"]
    check("amazon_access and ozone_level share the printed majority rate",
          f"{inv['amazon_access']['majority_rate']:.2f}" == f"{inv['ozone_level']['majority_rate']:.2f}")
    check("AI-use disclosure present in the methods", "\\paragraph{Use of AI tools.} During the preparation of this manuscript, automated language assistance tools were utilized" in tex)
    abstract = tex[tex.index("\\abstract{") + 10: tex.index("\\keywords")]
    abstract = re.sub(r"\\[a-zA-Z]+", " ", re.sub(r"\$[^$]*\$", "X", re.sub(r"(?m)^%.*$", "", abstract)))
    n_words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", abstract))
    check("abstract at most 250 words", n_words <= 250, n_words)
    keywords = tex[tex.index("\\keywords{") + 10:].split("}", 1)[0].split(",")
    check("at most six keywords", len(keywords) <= 6, len(keywords))
    import importlib.metadata
    for pkg, exp_ver in [("scikit-learn", "1.9.1"), ("xgboost", "3.4.1"), ("lightgbm", "4.7.0"), ("catboost", "1.2.10"), ("autorank", "1.3.0")]:
        try:
            installed = importlib.metadata.version(pkg)
            check(f"installed {pkg} matches pinned version ({exp_ver})", installed == exp_ver, f"got {installed}")
        except Exception as err:
            check(f"installed {pkg} matches pinned version ({exp_ver})", False, str(err))


# --------------------------------------------------------------------------- originality
def prose_words(tex):
    tex = tex[tex.find("\\begin{document}"):] if "\\begin{document}" in tex else tex
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    tex = re.sub(r"\\begin\{(table\*?|tabular\*?|tikzpicture|algorithmic|equation|gather)\}.*?\\end\{\1\}", " ",
                 tex, flags=re.S)
    tex = re.sub(r"\$[^$\n]*\$", " ", tex)
    tex = re.sub(r"\\(cite|ref|label|includegraphics|eqref)\*?(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", tex.lower())


def check_originality(tex, n=8):
    words = prose_words(tex)
    check("prose extraction sane (> 6,000 words)", len(words) > 6000, str(len(words)))
    grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    src = pd.read_csv("literature/papers_2025_2026.csv")
    texts = (src.title.fillna("") + " " + src.abstract.fillna("")).tolist()
    texts += pd.read_csv("literature/granular_gaps_and_future_scopes.csv").sentence.fillna("").tolist()
    hits = set()
    for txt in texts:
        w = re.findall(r"[a-z]+(?:'[a-z]+)?", txt.lower())
        hits |= grams & {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    check(f"no {n}-word run copied from 2,773 abstracts or 1,234 gap sentences", not hits, str(sorted(hits)[:5]))


def main():
    tex = open(TEX).read()
    prim, extra = load_json(os.path.join(PRIM, "audit_numbers.json")), load_json(os.path.join(PRIM, "extras.json"))
    orig = load_json("results/analysis/audit/audit_numbers.json")
    check_build(tex)
    check_counts(extra)
    check_original_patch()
    final = extra.get("seeds") is not None and extra.get("budget", {}).get("complete")
    if final:
        t = check_independent(prim, extra)
        check_claims(tex, prim, extra, orig, t)
    else:
        check("analysis complete (seeds and library defaults)", False, "provisional data; claims not checked")
    check_operators(tex)
    check_foundation(tex)
    check_sensitivity(tex)
    if "--skip-harness" not in sys.argv:
        check_harness()
    check_document(tex)
    check_originality(tex)
    print(f"passed {passes}, failed {len(failures)}")
    for f in failures:
        print("  FAIL", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
