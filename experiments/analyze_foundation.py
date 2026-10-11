"""Pretrained foundation model (TabICL) against the six classical families under Protocol B, imputed pipeline.

Reuses the machinery of analyze_revision (trajectory metrics, Friedman/Iman-Davenport, Holm, seed-mean) with the
model pool extended to seven. Writes results/analysis/revision/foundation.json and tab_foundation.tex.
Only seeds and datasets for which TabICL is complete enter the comparison, for all seven models alike.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))
import analyze_revision as ar                                           # noqa: E402
from forensic_audit import holm                                         # noqa: E402

CLASSICAL = list(ar.MODELS)
FOUNDATION = "tabicl"
POOL = CLASSICAL + [FOUNDATION]
OPERATORS = ["gaussian_feature", "mcar_missing", "mar_missing", "mnar_missing"]
N_TIERS = 11
N_FOLDS = 5
UNSUPPORTED = ["Bioresponse"]      # 1,776 columns: infeasible on the 8 GB machine (foundation_failures.txt)
CONDITIONS = ["gaussian_feature", "mcar_imputed", "mar_imputed", "mnar_imputed"]
COND_LABEL = {"gaussian_feature": "Gaussian", "mcar_imputed": "MCAR", "mar_imputed": "MAR", "mnar_imputed": "MNAR"}
SCORES = ["audc_bal", "audc", "mrr"]
SCORE_LABEL = {"audc_bal": "AUDC$_{\\mathrm{bal}}$", "audc": "AUDC", "mrr": "MRR"}
NICE = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
        "mlp": "MLP", "tabicl": "TabICL"}
OUT = ar.OUT


def foundation_rows(fnd):
    """Analysable TabICL rows and the seeds that are complete.

    Resumed runs are de-duplicated. A seed counts only when it holds every planned cell (all supported datasets x
    four operators x eleven tiers), so a partially written seed file (seeds 43 and 44 are appended while this
    runs) never enters the analysis. Mirrors analyze_revision.mechanism_rows."""
    from src.data_loader import BENCHMARK_SUITE
    fnd = fnd[fnd.modality.isin(OPERATORS) & ~fnd.dataset.isin(UNSUPPORTED)]
    fnd = fnd.drop_duplicates(["seed", "dataset", "model", "modality", "eta"], keep="last")
    planned = len([d for d in BENCHMARK_SUITE if d not in UNSUPPORTED]) * len(OPERATORS) * N_TIERS
    counts = fnd.groupby("seed").size()
    seeds = sorted(int(s) for s, n in counts.items() if n == planned)
    return fnd[fnd.seed.isin(seeds)], seeds


def load_trajectories(bases):
    """Seed-mean Protocol B trajectories of TabICL and the six classical families on the same datasets, seeds,
    folds and corruption draws (imputed pipeline, four operators)."""
    fnd = ar.load_block("foundation")
    if fnd is None:
        raise SystemExit("no foundation results")
    fnd, seeds = foundation_rows(fnd)
    if not seeds:
        raise SystemExit("no complete TabICL seed yet")
    dep = ar.load_block("deploy")
    dep = dep[(dep.pipeline == "imputed") & dep.model.isin(CLASSICAL) & dep.modality.isin(OPERATORS)
              & dep.seed.isin(seeds) & dep.dataset.isin(fnd.dataset.unique())]
    dep = dep.drop_duplicates(["seed", "dataset", "model", "modality", "eta"], keep="last")
    datasets = sorted(fnd.dataset.unique())
    expected = len(seeds) * len(datasets) * len(CLASSICAL) * len(OPERATORS) * N_TIERS
    if len(dep) != expected:
        raise SystemExit(f"classical deploy rows incomplete for the TabICL seeds: {len(dep)} of {expected}")
    raw = ar.deploy_conditions(pd.concat([dep, fnd], ignore_index=True))
    raw = raw[raw.condition.isin(CONDITIONS)]
    cells = ar.seed_mean(raw, ["condition", "dataset", "model"])
    traj = ar.trajectories(cells, ["condition", "dataset", "model"], bases)
    return traj, seeds, datasets, dict(rows=int(len(fnd)), fits=int(N_FOLDS * fnd.drop_duplicates(
        ["seed", "dataset", "model", "pipeline"]).shape[0]))


def rank_vector(traj, models, metric, cond):
    pv = traj[(traj.condition == cond) & traj.model.isin(models)].pivot(index="dataset", columns="model",
                                                                          values=metric)[models]
    R, res = ar.friedman_block(pv)
    return {m: float(R[m]) for m in models}, res


def block_tests(traj):
    """Friedman/Iman-Davenport per (score, operator) for the 7-model pool, Holm over the 12 tests of the block."""
    rows = []
    for metric in SCORES:
        for cond in CONDITIONS:
            ranks, res = rank_vector(traj, POOL, metric, cond)
            six, _ = rank_vector(traj, CLASSICAL, metric, cond)
            rows.append(dict(metric=metric, condition=cond, F_ID=res["F_ID"], p=res["p_ID"], CD=res["CD"],
                             ranks=ranks, ranks_six=six))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    return df


def summarize_blocks(df):
    out = {}
    for r in df.itertuples():
        lead7, last7 = min(r.ranks, key=r.ranks.get), max(r.ranks, key=r.ranks.get)
        lead6, last6 = min(r.ranks_six, key=r.ranks_six.get), max(r.ranks_six, key=r.ranks_six.get)
        within = [r.ranks[m] for m in CLASSICAL]       # ranking of the six families inside the 7-model pool
        tau = float(stats.kendalltau([r.ranks_six[m] for m in CLASSICAL], within)[0])
        out[f"{r.metric}|{r.condition}"] = dict(
            ranks=r.ranks, ranks_six=r.ranks_six, F_ID=float(r.F_ID), p=float(r.p), p_holm=float(r.p_holm),
            CD=float(r.CD), tabicl_rank=r.ranks[FOUNDATION],
            tabicl_position=int(1 + sum(v < r.ranks[FOUNDATION] for v in r.ranks.values())),
            leader7=lead7, last7=last7, leader6=lead6, last6=last6,
            leader_changes=lead7 != lead6, last_changes=last7 != last6, tau_six_with_vs_without=tau)
    return out


def clean_accuracy(traj):
    clean = traj[traj.condition == "gaussian_feature"].pivot(index="dataset", columns="model", values="acc0")[POOL]
    R, res = ar.friedman_block(clean)
    best = clean.idxmax(axis=1)
    return {"mean_acc": {m: float(clean[m].mean()) for m in POOL},
            "median_acc": {m: float(clean[m].median()) for m in POOL},
            "mean_rank": {m: float(R[m]) for m in POOL},
            "friedman_p": float(res["p_ID"]),
            "best_classical_mean_acc": float(clean[CLASSICAL].mean().max()),
            "tabicl_best_count": int((best == FOUNDATION).sum()), "n_datasets": int(len(clean)),
            "tabicl_best_datasets": sorted(best[best == FOUNDATION].index),
            "tabicl_minus_best_classical_median": float((clean[FOUNDATION] - clean[CLASSICAL].max(axis=1)).median()),
            "per_dataset": clean.round(4).to_dict(orient="index")}


def clean_vs_robust(traj, models):
    """Share of datasets where the clean-accuracy winner is also the MRR winner / balanced-AUDC winner."""
    out = {}
    for cond in CONDITIONS:
        sub = traj[(traj.condition == cond) & traj.model.isin(models)]
        rec = {}
        for metric in ["mrr", "audc_bal"]:
            rec[metric] = float(np.mean([
                g.set_index("model").acc0.idxmax() == g.set_index("model")[metric].idxmax()
                for _, g in sub.groupby("dataset")]))
        out[cond] = rec
    out["mean"] = {m: float(np.mean([out[c][m] for c in CONDITIONS])) for m in ["mrr", "audc_bal"]}
    return out


def tabicl_gap(traj):
    """Per operator: median over datasets of TabICL's score minus the best classical family's score."""
    out = {}
    for cond in CONDITIONS:
        sub = traj[traj.condition == cond]
        rec = {}
        for metric in ["audc", "audc_bal", "mrr"]:
            pv = sub.pivot(index="dataset", columns="model", values=metric)
            gap = pv[FOUNDATION] - pv[CLASSICAL].max(axis=1)
            rec[metric] = dict(median_gap=float(gap.median()), share_tabicl_higher=float((gap > 0).mean()),
                               wilcoxon_p=float(stats.wilcoxon(gap, zero_method="pratt").pvalue)
                               if (gap != 0).any() else 1.0)
        out[cond] = rec
    return out


def conclusion_changes(blocks):
    keys = list(blocks)
    lists = dict(leader_changes=[k for k in keys if blocks[k]["leader_changes"]],
                 last_changes=[k for k in keys if blocks[k]["last_changes"]],
                 tabicl_leader=[k for k in keys if blocks[k]["leader7"] == FOUNDATION],
                 tabicl_last=[k for k in keys if blocks[k]["last7"] == FOUNDATION])
    counts = {f"n_{name}": len(v) for name, v in lists.items()}
    # blocks whose leader among the six classical families changes although TabICL does not take the lead
    counts["n_leader_changes_without_tabicl"] = len(set(lists["leader_changes"]) - set(lists["tabicl_leader"]))
    return dict(n_blocks=len(keys), **lists, **counts,
                tau_min=float(min(blocks[k]["tau_six_with_vs_without"] for k in keys)),
                tau_median=float(np.median([blocks[k]["tau_six_with_vs_without"] for k in keys])),
                n_significant_holm=int(sum(blocks[k]["p_holm"] < ar.ALPHA for k in keys)))


def headline(blocks, gap, clean):
    """Counts and ranges quoted in the text, per score, over the four operators."""
    out = {}
    for metric in SCORES:
        keys = [f"{metric}|{c}" for c in CONDITIONS]
        pos = [blocks[k]["tabicl_position"] for k in keys]
        out[metric] = dict(
            n_tabicl_leader=int(sum(blocks[k]["leader7"] == FOUNDATION for k in keys)),
            n_tabicl_last=int(sum(blocks[k]["last7"] == FOUNDATION for k in keys)),
            n_tabicl_top2=int(sum(p <= 2 for p in pos)), n_tabicl_bottom2=int(sum(p >= len(POOL) - 1 for p in pos)),
            position_min=int(min(pos)), position_max=int(max(pos)),
            rank_min=float(min(blocks[k]["tabicl_rank"] for k in keys)),
            rank_max=float(max(blocks[k]["tabicl_rank"] for k in keys)),
            gap_min=float(min(gap[c][metric]["median_gap"] for c in CONDITIONS)),
            gap_max=float(max(gap[c][metric]["median_gap"] for c in CONDITIONS)),
            share_higher_max=float(max(gap[c][metric]["share_tabicl_higher"] for c in CONDITIONS)))
    out["masking_abs_gap_max"] = float(max(abs(gap[c][m]["median_gap"]) for c in ("mcar_imputed", "mar_imputed")
                                           for m in ("audc_bal", "audc")))
    out["clean_position"] = int(1 + sum(v < clean["mean_rank"][FOUNDATION] for v in clean["mean_rank"].values()))
    out["clean_mean_acc_highest"] = bool(clean["mean_acc"][FOUNDATION] == max(clean["mean_acc"].values()))
    return out


def signed(x):
    """Signed three-decimal value; a difference that rounds to zero carries no sign."""
    return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"


def cell(name, ranks):
    text = f"{NICE[name]} ({ranks[name]:.2f})"
    return text


def latex(out, path):
    """Compact table: TabICL's mean rank among the seven and the 7-model leader, per score and operator."""
    clean = out["clean_accuracy"]
    cr = clean["mean_rank"]
    lead = min(cr, key=cr.get)
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{A pretrained tabular foundation model (TabICL) among the six classical families under "
             "Protocol~B (inference robustness; clean in-context training set, corrupted evaluation, imputed pipeline, "
             f"{'seed ' if len(out['seeds']) == 1 else 'seeds '}{', '.join(map(str, out['seeds']))}, "
             f"$N={out['n_datasets']}$ datasets). Mean Friedman rank of TabICL among the seven models and the "
             "model with the lowest mean rank (rank in parentheses); adj.\\ $p$: Iman--Davenport $p$-value "
             "of the seven-model comparison, Holm-adjusted over the twelve tests of the block; $\\Delta$: median "
             "over datasets of TabICL's score minus that of the best classical family. Secondary analysis; TabICL was run under a constrained configuration and the table does not generalise to foundation models as a class.}",
             "\\label{tab:foundation}", "\\begin{tabular}{@{}llrlrr@{}}", "\\toprule",
             "Score & Operator & TabICL rank & Leader of seven & adj.\\ $p$ & $\\Delta$ \\\\", "\\midrule",
             f"Clean accuracy & -- & {cr[FOUNDATION]:.2f} & {cell(lead, cr)} & -- & "
             f"${signed(clean['tabicl_minus_best_classical_median'])}$ \\\\"]
    for metric in SCORES:
        lines.append("\\midrule")
        for cond in CONDITIONS:
            b = out["blocks"][f"{metric}|{cond}"]
            gap = out["gap"][cond][metric]["median_gap"]
            lines.append(f"{SCORE_LABEL[metric] if cond == CONDITIONS[0] else ''} & {COND_LABEL[cond]} & "
                         f"{b['tabicl_rank']:.2f} & {cell(b['leader7'], b['ranks'])} & {ar.latex_p(b['p_holm'])} & "
                         f"${signed(gap)}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    bases = ar.dataset_baselines()
    traj, seeds, datasets, counts = load_trajectories(bases)
    df = block_tests(traj)
    blocks = summarize_blocks(df)
    clean = clean_accuracy(traj)
    gap = tabicl_gap(traj)
    settings_path = os.path.join(ar.REV_METRICS, "foundation_settings.json")
    out = {"seeds": seeds, "n_seeds": len(seeds), "datasets": datasets, "n_datasets": len(datasets), "pool": POOL,
           "excluded_datasets": UNSUPPORTED, "accounting": counts,
           "settings": json.load(open(settings_path)) if os.path.exists(settings_path) else None,
           "clean_accuracy": clean, "blocks": blocks, "headline": headline(blocks, gap, clean),
           "conclusion_changes": conclusion_changes(blocks),
           "clean_vs_robust_7": clean_vs_robust(traj, POOL),
           "clean_vs_robust_6": clean_vs_robust(traj, CLASSICAL),
           "gap": gap,
           "median_scores": {c: {m: {s: float(traj[(traj.condition == c) & (traj.model == m)][s].median())
                                      for s in SCORES + ["acc0", "acc_end"]} for m in POOL} for c in CONDITIONS}}
    traj.to_csv(os.path.join(OUT, "trajectories_foundation.csv"), index=False)
    json.dump(ar.json_safe(out), open(os.path.join(OUT, "foundation.json"), "w"), indent=1)
    latex(out, os.path.join(OUT, "tab_foundation.tex"))
    brief = {k: out[k] for k in ["seeds", "n_datasets", "conclusion_changes", "clean_vs_robust_7", "clean_vs_robust_6",
                                 "gap"]}
    brief["clean"] = {k: out["clean_accuracy"][k] for k in ["mean_acc", "mean_rank", "tabicl_best_count",
                                                            "tabicl_minus_best_classical_median"]}
    print(json.dumps(ar.json_safe(brief), indent=1))


if __name__ == "__main__":
    main()
