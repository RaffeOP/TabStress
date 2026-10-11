"""LaTeX tables of the revised manuscript, from results/analysis/revision/revision.json and the audit outputs.

  tab_inventory_main.tex    simplified corpus table (revision point 33)
  tab_accounting.tex        derivation of every cell and fit count (point 9)
  tab_search_space.tex      search space of the tuned configurations (Supplementary Table S1)
  tab_protocol.tex          Protocol A versus Protocol B (points 4, 5)
  tab_protocol_b_tests.tex  Friedman tests of every Protocol B condition (supplement)
  tab_accounting.tex        also the TabICL evaluation count when foundation.json exists
  tab_sensitivity.tex       configuration, mechanism and artifact sensitivity (points 19, 20, 22, 39)
  tab_gaussian_principal.tex  principal Gaussian experiment, continuous attributes only (point 21)
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REV = "results/analysis/revision"
NICE = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "catboost": "CatBoost", "mlp": "MLP"}
COND = {"gaussian_feature": "Gaussian", "type_aware_feature": "Type-aware", "mcar_imputed": "MCAR, imputed",
        "mcar_native": "MCAR, native", "mar_imputed": "MAR, imputed", "mar_native": "MAR, native",
        "mnar_imputed": "MNAR, imputed", "mnar_native": "MNAR, native", "symmetric_label": "Symmetric label",
        "gaussian_feature_B": "Gaussian (Protocol B)", "class_conditional": "Class-conditional"}
DOMAINS = [("Science", ["Bioresponse", "qsar-biodeg", "eye_movements", "gesture_phase", "ozone_level"]),
           ("Industrial", ["steel-plates-fault", "vehicle", "electricity", "pol", "miniboone"]),
           ("Business, web", ["bank-marketing", "churn", "amazon_access", "nomao", "spambase"]),
           ("Synthetic, other", ["madelon", "hill_valley", "jungle_chess", "phoneme", "volkert"])]


def write(name, lines):
    with open(os.path.join(REV, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def esc(name):
    return name.replace("_", "\\_")


def p_txt(p):
    if p is None:
        return "--"
    if p >= 1e-3:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def inventory(audit):
    from experiments.run_extended import data
    from experiments.run_revision import discrete_columns
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{Benchmark corpus. $N$: rows (after the one-off stratified cap of 10{,}000); $D$: encoded "
            "columns; $K$: classes; $\\pi_{\\max}$: majority-class rate; type: share of integer-coded columns "
            "(real-valued when 0\\%, discrete when 100\\%). OpenML identifiers and preprocessing details are in "
            "Supplementary Table~S2.}", "\\label{tab:inventory}", "\\begin{tabular}{@{}llrrrrl@{}}", "\\toprule",
            "Dataset & Domain & $N$ & $D$ & $K$ & $\\pi_{\\max}$ & Type \\\\", "\\midrule"]
    for domain, names in DOMAINS:
        for d in names:
            inv = audit["inventory"][d]
            X, _, _ = data(d)
            share = discrete_columns(X).mean()
            kind = "real" if share == 0 else ("discrete" if share == 1 else f"mixed ({100 * share:.0f}\\%)")
            rows.append(f"{esc(d)} & {domain} & {inv['N_eff']:,} & {inv['D']} & {inv['K']} & "
                        f"{inv['majority_rate']:.2f} & {kind} \\\\".replace(",", "{,}", 1))
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_inventory_main.tex", rows)


def accounting(acc, found=None):
    def n(v):
        return f"{v:,}".replace(",", "{,}")
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{Derivation of the cell and fit counts. A cell is one dataset, model, operator, intensity and "
            "seed evaluated by five-fold cross-validation (five model fits under Protocol~A; under Protocol~B one fit per fold serves all tiers). An evaluation scores one fitted model on one evaluation fold at one tier, so a cell comprises five evaluations; the rows below count cells. The primary grid reuses "
            "each clean cell in all five conditions and the imputed MLP rows in the native condition.}",
            "\\label{tab:accounting}", "\\begin{tabular}{@{}p{3.3cm}p{5.2cm}r@{}}", "\\toprule",
            "Block & Factors & Cells \\\\", "\\midrule",
            f"Original run, seed 42 & $20\\times6\\times4$ operators $\\times\\,11$ tiers & {n(acc['original_seed42'])} \\\\",
            f"Seeds 43, 44 & $2\\times20\\times6\\times(1\\text{{ clean}}+4\\times10)$ & {n(acc['base_seeds_43_44'])} \\\\",
            f"Minority-to-majority & $3\\times14$ binary datasets $\\times\\,6\\times10$ & {n(acc['cc_binary'])} \\\\",
            f"Native handling & $3\\times20\\times5$ tree families $\\times\\,10$ & {n(acc['native_nan'])} \\\\",
            f"Library defaults & $2$ libraries $\\times\\,20\\times(1+4\\times10)$, seed 42 & {n(acc['library_defaults'])} \\\\",
            "\\midrule",
            f"Protocol A total & cells / model fits & {n(acc['total_cells'])} / {n(acc['total_fits'])} \\\\",
            f"\\quad of which primary & $3\\times(120\\text{{ clean}}+5{{,}}800\\text{{ noisy}})$ & {n(acc['primary_distinct'])} \\\\",
            f"\\quad pair-flip on binary & test--retest control, $3\\times14\\times6\\times10$ & {n(acc['pairflip_binary_test_retest'])} \\\\",
            f"\\quad duplicate clean & seed-42 clean cells of three extra operators & {n(acc['duplicate_clean_seed42'])} \\\\",
            f"Primary grid rows & $20\\times6\\times5\\times11\\times3$ & {n(acc['primary_grid_cells'])} \\\\"]
    if "deploy_rows" in acc:
        rows.append(f"Protocol B & cells / model fits & "
                    f"{n(acc['deploy_rows'])} / {n(acc['deploy_fits'])} \\\\")
    if found:
        rows.append(f"TabICL (B) & cells / model fits, {seed_span(found['seeds'])}, "
                    f"{found['n_datasets']} datasets & {n(found['accounting']['rows'])} / "
                    f"{n(found['accounting']['fits'])} \\\\")
    if "train_mech_rows" in acc:
        rows.append(f"MAR, MNAR (A) & cells, {seed_span(acc['train_mech_seeds'])} & {n(acc['train_mech_rows'])} \\\\")
    if "tuned_rows" in acc:
        rows.append(f"Tuned configurations & cells, seed 42 & {n(acc['tuned_rows'])} \\\\")
    for key, label in [("deploy_cat_rows", "Categorical (B)"), ("train_cat_rows", "Categorical (A)")]:
        if key in acc:
            rows.append(f"{label} & cells, seeds 42--44 & {n(acc[key])} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_accounting.tex", rows)


def search_space():
    from experiments.run_revision import SEARCH_SPACE, TUNED_TRIALS
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            f"\\caption{{Search space of the tuned configurations. For each dataset and family, {TUNED_TRIALS} "
            "configurations are drawn uniformly (seed 42) and compared by three-fold cross-validated accuracy on the "
            "clean training part of the first outer fold; parameters not listed keep the matched-budget value.}",
            "\\label{tab:search}", "\\begin{tabular}{@{}lp{9.5cm}@{}}", "\\toprule", "Family & Values \\\\", "\\midrule"]
    for m, space in SEARCH_SPACE.items():
        txt = "; ".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} $\\in$ \\{{{', '.join(map(str, v))}\\}}"
                        for k, v in space.items())
        rows.append(f"{NICE[m]} & {txt} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_search_space.tex", rows)


def comparison_rows(comp, label_a, label_b):
    rows = []
    for cond, rec in comp.items():
        for metric, mname in [("audc_bal", "AUDC$_{\\mathrm{bal}}$"), ("audc", "AUDC"), ("mrr", "MRR")]:
            r = rec[metric]
            rows.append(f"{COND.get(cond, cond) if metric == 'audc_bal' else ''} & {mname} & "
                        f"${r['median_delta']:+.4f}$ & {100 * r['share_b_lower']:.0f} & {p_txt(r['p_wilcoxon'])} & "
                        f"{NICE[r['best_a']]} / {NICE[r['best_b']]} & ${r['tau']:.2f}$ \\\\")
    return rows


def seed_span(seeds):
    """[42] -> "seed 42", [42, 43, 44] -> "seeds 42--44", [42, 44] -> "seeds 42, 44"."""
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"seeds {seeds[0]}--{seeds[-1]}"
    return "seeds " + ", ".join(map(str, seeds))


def protocol(rev):
    dep = rev["deploy"]
    mech_scope = seed_span(rev["mechanisms"]["seeds"]) if (rev.get("mechanisms") or {}).get("complete") else "seed 42"
    rows = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
            "\\caption{Training robustness (Protocol~A, corrupted training and test data) against inference robustness (Protocol~B, clean training, corrupted test data) "
            "on matched conditions. $\\Delta$: median over the 120 dataset--model pairs of the score under B minus "
            "the score under A (negative values indicate performance degradation under deployment corruption); "
            "B$<$A: share of pairs with a lower score under B; $p$: Wilcoxon signed-rank test of "
            "the paired differences; leaders: model with the lowest mean rank under A / B; $\\tau$: Kendall "
            "correlation of the two mean-rank vectors (lower values signify ranking reordering across protocols). "
            "Upper block: seeds 42--44, twenty datasets; lower block: " + mech_scope + ", "
            "the same twenty datasets, for both protocols.}",
            "\\label{tab:protocol}", "\\begin{tabular}{@{}llrrrlr@{}}", "\\toprule",
            "Condition & Score & $\\Delta$ (B$-$A) & B$<$A (\\%) & $p$ & Leader A / B & $\\tau$ \\\\", "\\midrule"]
    rows += comparison_rows(dep["comparison"], "A", "B")
    mech = rev.get("mechanisms") or {}
    if mech.get("complete"):
        rows.append("\\midrule")
        rows += comparison_rows(mech["comparison"], "A", "B")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("tab_protocol.tex", rows)
    tests = ["\\begin{table}[p]", "\\centering", "\\footnotesize",
             "\\caption{Protocol~B (clean training, corrupted evaluation, seeds 42--44): Iman--Davenport tests of "
             "every condition and score, Holm-adjusted over the whole block; best and worst model by mean rank; "
             "W: model pairs separated by Holm-adjusted Wilcoxon tests (of 15).}",
             "\\label{tab:protocolb_tests}", "\\begin{tabular}{@{}llllrr@{}}", "\\toprule",
             "Condition & Score & Best & Worst & adj.\\ $p$ & W \\\\", "\\midrule"]
    for key, t in sorted(dep["tests"].items(), key=lambda kv: (kv[0].split("|")[1], kv[0].split("|")[0])):
        metric, cond = key.split("|")
        tests.append(f"{COND.get(cond, cond)} & {metric.replace('_bal', '$_{\\mathrm{bal}}$').upper().replace('$_{\\MATHRM{BAL}}$', '$_{\\mathrm{bal}}$')} & "
                     f"{NICE[t['best']]} & {NICE[t['worst']]} & {p_txt(t['p_holm'])} & {t['n_wilcoxon_sig']} \\\\")
    tests += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_protocol_b_tests.tex", tests)


def categorical(rev):
    cat = rev.get("categorical") or {}
    if not cat.get("complete"):
        return
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{Categorical corruption on the " + str(cat["n_datasets"]) + " datasets with nominal attributes "
            "(seeds 42--44). Categorical-aware: Gaussian noise on continuous attributes and whole-attribute "
            "resampling of nominal attributes; categorical-only: nominal attributes resampled, continuous attributes "
            "clean. $\\Delta$: median over dataset--model pairs of balanced AUDC minus that of the plain Gaussian "
            "operator on the same folds; $p$: Wilcoxon signed-rank test.}",
            "\\label{tab:categorical}", "\\begin{tabular}{@{}llrrrll@{}}", "\\toprule",
            "Protocol & Operator & AUDC$_{\\mathrm{bal}}$ & $\\Delta$ vs.\\ Gaussian & $p$ & Best & Worst \\\\",
            "\\midrule"]
    for proto in ["A", "B"]:
        rec = cat[proto]
        for op, name in [("cat_aware_feature", "categorical-aware"), ("cat_only_feature", "categorical-only")]:
            r = rec[op]
            rows.append(f"{proto if op == 'cat_aware_feature' else ''} & {name} & {r['audc_bal_median']:.3f} & "
                        f"${r['vs_gaussian_median_delta']:+.4f}$ & {p_txt(r['vs_gaussian_p'])} & "
                        f"{NICE[r['best_audc_bal']]} & {NICE[r['worst_audc_bal']]} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_categorical.tex", rows)


def gaussian_principal(rev):
    gc = rev.get("gauss_cont") or {}
    if not gc.get("complete"):
        return
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{The principal Gaussian condition against its all-column variant, " + str(gc["n_datasets"]) +
            " datasets, seeds 42--44. Principal: Gaussian noise on the continuous attributes only (nominal "
            "attributes left clean on the six datasets that have them; identical to the all-column operator on the "
            "other fourteen), the condition used throughout the article. All-column: the same noise on every encoded "
            "column, including label-encoded nominal codes; a mathematical stress test of representation resilience, "
            "not a realistic categorical error process. Omnibus $p$-values are unadjusted and belong to the "
            "sensitivity family. $\\tau$ compares the mean-rank vectors of the two variants.}",
            "\\label{tab:gaussian_principal}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{@{}llrrllrrrl@{}}", "\\toprule",
            "Protocol & Variant & AUDC$_{\\mathrm{bal}}$ $F$ & $p$ & Best & Worst & MRR $F$ & $p$ & "
            "Kendall $\\tau$ & Same best/worst \\\\",
            "\\midrule"]
    labels = {"A": "A (training)", "B": "B (inference)"}
    for proto in ["A", "B"]:
        r = gc[proto]
        same = f"{'yes' if r['same_best'] else 'no'} / {'yes' if r['same_worst'] else 'no'}"
        for variant, rec, tail in [("Principal", r, f"{r['tau_vs_all_column_audc_bal']:.2f} & {same}"),
                                   ("All-column", r["all_column"], "-- & --")]:
            rows.append(f"{labels[proto] if variant == 'Principal' else ''} & {variant} & "
                        f"{rec['audc_bal']['F']:.2f} & {p_txt(rec['audc_bal']['p'])} & "
                        f"{NICE[rec['audc_bal']['best']]} & {NICE[rec['audc_bal']['worst']]} & "
                        f"{rec['mrr']['F']:.2f} & {p_txt(rec['mrr']['p'])} & {tail} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    write("tab_gaussian_principal.tex", rows)


def sensitivity(rev):
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{Sensitivity of the cross-model ranking. Each row compares the mean-rank vectors of a "
            "variant with the reference analysis over the fifteen confirmatory blocks (score $\\times$ condition) "
            "or the stated blocks: blocks with the same best and worst model, the smallest Kendall $\\tau$, and the "
            "number of omnibus tests that remain significant after Holm adjustment.}",
            "\\label{tab:sensitivity}", "\\begin{tabular}{@{}p{4.6cm}rrrr@{}}", "\\toprule",
            "Variant & Same best & Same worst & min.\\ $\\tau$ & Significant \\\\", "\\midrule"]
    art = rev["artifacts"]
    for key, label in [("cleaned", "Cleaned spambase, churn, MiniBooNE"),
                       ("excluded", "Flagged datasets excluded ($N=16$)")]:
        if key in art:
            a = art[key]
            rows.append(f"{label} & {a['same_best']}/{a['n_blocks']} & {a['same_worst']}/{a['n_blocks']} & "
                        f"${a['tau_min']:.2f}$ & {a['n_sig_holm']}/{a['n_blocks']} \\\\")
    tuned = rev.get("tuned") or {}
    if tuned.get("complete"):
        comp = tuned["comparison"]
        same_b = sum(comp[c][m]["best_a"] == comp[c][m]["best_b"] for c in comp for m in comp[c])
        same_w = sum(comp[c][m]["worst_a"] == comp[c][m]["worst_b"] for c in comp for m in comp[c])
        n = sum(len(comp[c]) for c in comp)
        tau = min(comp[c][m]["tau"] for c in comp for m in comp[c])
        rows.append(f"Tuned instead of matched budget (seed 42) & {same_b}/{n} & {same_w}/{n} & ${tau:.2f}$ & -- \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_sensitivity.tex", rows)


def tabicl_configs():
    path = os.path.join(REV, "sensitivity_extra.json")
    if not os.path.exists(path):
        return
    st = json.load(open(path))["tabicl_settings"]
    names = {"audc_bal": "AUDC$_{\\mathrm{bal}}$", "mrr": "MRR"}
    ops = [("gaussian_feature", "Gaussian"), ("mcar_imputed", "MCAR"), ("mar_imputed", "MAR"),
           ("mnar_imputed", "MNAR")]
    rows = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
            "\\caption{TabICL, secondary analysis (inference robustness, Protocol~B, seed 42, "
            f"{st['n_datasets']} datasets): constrained main configuration (4 estimators, in-context set capped at "
            "3,000 rows) against a default-setting sensitivity run (8 estimators, no cap). The two runs use "
            "different compute budgets and are not an equal-budget comparison; the default run covers a subset of "
            "datasets and one seed. Mean rank of TabICL among the seven models (position in parentheses). "
            "Unchanged: TabICL's MRR rank under all four operators and its clean accuracy. "
            "Changed: its balanced-AUDC rank under MNAR masking. "
            "No claim extends to foundation models as a class.}",
            "\\label{tab:tabicl_configs}", "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
            "Score & Operator & Main (constrained) & Default \\\\", "\\midrule"]
    for score in ("mrr", "audc_bal"):
        for i, (op, label) in enumerate(ops):
            r = st["ranks"][f"{score}|{op}"]
            first = names[score] if i == 0 else ""
            rows.append(f"{first} & {label} & {r['main']['tabicl_rank']:.2f} ({r['main']['position']}) & "
                        f"{r['default']['tabicl_rank']:.2f} ({r['default']['position']}) \\\\")
        rows.append("\\midrule")
    c = st["clean"]
    rows.append(f"Clean accuracy (mean) & -- & {c['mean_main']:.3f} & {c['mean_default']:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_tabicl_configs.tex", rows)


def main():
    rev = json.load(open(os.path.join(REV, "revision.json")))
    audit = json.load(open("results/analysis/audit_primary/audit_numbers.json"))
    inventory(audit)
    fpath = os.path.join(REV, "foundation.json")
    accounting(rev["accounting"], json.load(open(fpath)) if os.path.exists(fpath) else None)
    search_space()
    if rev.get("deploy"):
        protocol(rev)
    sensitivity(rev)
    categorical(rev)
    gaussian_principal(rev)
    tabicl_configs()


if __name__ == "__main__":
    main()
