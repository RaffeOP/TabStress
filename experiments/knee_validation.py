"""Synthetic validation of the knee estimators (revision point 25).

Trajectories with a known knee are sampled on the eleven TabStress tiers, perturbed by Gaussian noise
of the size observed for seed-averaged tiers, and passed to three estimators:
  spline     reference: argmax of the curvature of the interpolating natural cubic spline (500-point grid)
  spline_tier the reference snapped to the nearest sampled tier (its effective resolution)
  discrete   argmax over interior tiers of |A_{t-1} - 2 A_t + A_{t+1}| (second difference, no interpolation)
Writes results/analysis/revision/knee_validation.json and figure knee_validation.png.
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.breakdown_math import compute_critical_breakdown_threshold

TIERS = np.round(np.arange(0, 0.501, 0.05), 2)
OUT = "results/analysis/revision"
REPLICATES = 500
NOISE_SDS = [0.0, 0.0025, 0.005, 0.01, 0.02]
KNEES = {"early": 0.10, "middle": 0.25, "late": 0.40}
ACC0 = 0.85
SLOPE_BEFORE, SLOPE_AFTER = 0.10, 1.00
KNEE_WIDTH = 0.02
TIER_STEP = 0.05


def hinge(eta, knee):
    """Smoothed hinge: slope SLOPE_BEFORE below the knee, SLOPE_AFTER above it (softplus of width KNEE_WIDTH)."""
    soft = KNEE_WIDTH * np.logaddexp(0.0, (eta - knee) / KNEE_WIDTH)
    return ACC0 - SLOPE_BEFORE * eta - (SLOPE_AFTER - SLOPE_BEFORE) * soft


def shapes():
    out = {f"{name}_knee": (lambda e, k=k: hinge(e, k), k) for name, k in KNEES.items()}
    out["smooth_linear"] = (lambda e: ACC0 - 0.4 * e, None)
    out["smooth_exponential"] = (lambda e: 0.5 + (ACC0 - 0.5) * np.exp(-4.0 * e), None)
    out["flat"] = (lambda e: np.full_like(e, ACC0), None)
    return out


def estimators(acc):
    res = compute_critical_breakdown_threshold(TIERS, acc)
    spline = res["eta_star"] if res["max_curvature"] > 0 else np.nan
    second = np.abs(acc[:-2] - 2 * acc[1:-1] + acc[2:])
    return {"spline": spline,
            "spline_tier": np.round(spline / TIER_STEP) * TIER_STEP if np.isfinite(spline) else np.nan,
            "discrete": TIERS[1 + int(np.argmax(second))],
            "kappa_max": res["max_curvature"]}


def summarise(estimates, truth):
    est = np.array(estimates, dtype=float)
    est = est[np.isfinite(est)]
    if len(est) == 0:
        return {"n": 0, "undefined_share": 1.0}
    out = {"n": int(len(est)), "median": float(np.median(est)), "iqr": [float(np.percentile(est, 25)),
                                                                      float(np.percentile(est, 75))]}
    if truth is not None:
        err = est - truth
        out.update(bias=float(np.mean(err)), mae=float(np.mean(np.abs(err))),
                   within_one_tier=float(np.mean(np.abs(err) <= TIER_STEP + 1e-9)),
                   within_half_tier=float(np.mean(np.abs(err) <= TIER_STEP / 2 + 1e-9)))
    else:
        out["interior_share"] = float(np.mean((est > 0.05) & (est < 0.45)))
    out["undefined_share"] = 1.0 - len(est) / len(estimates)
    return out


def run():
    rng = np.random.default_rng(2026)
    report = {}
    for name, (fn, truth) in shapes().items():
        clean = fn(TIERS)
        report[name] = {"true_knee": truth}
        for sd in NOISE_SDS:
            draws = [estimators(clean + rng.normal(0, sd, len(TIERS))) for _ in range(REPLICATES if sd else 1)]
            block = {k: summarise([d[k] for d in draws], truth) for k in ["spline", "spline_tier", "discrete"]}
            block["kappa_max_median"] = float(np.median([d["kappa_max"] for d in draws]))
            report[name][f"sd_{sd}"] = block
    return report


def headline(report):
    """Numbers quoted in the paper: recovery at the noise level of a seed-averaged tier (sd 0.005)."""
    key = "sd_0.005"
    knee_names = [f"{n}_knee" for n in KNEES]
    h = {"noise_sd": 0.005, "replicates": REPLICATES}
    for est in ["spline", "spline_tier", "discrete"]:
        h[est] = {n: report[n][key][est] for n in knee_names}
        h[est + "_within_one_tier_min"] = min(report[n][key][est]["within_one_tier"] for n in knee_names)
        h[est + "_mae_max"] = max(report[n][key][est]["mae"] for n in knee_names)
    h["noiseless_spline_error"] = {n: report[n]["sd_0.0"]["spline"]["mae"] for n in knee_names}
    h["no_knee_kappa_ratio"] = (report["smooth_linear"][key]["kappa_max_median"]
                                / report["middle_knee"][key]["kappa_max_median"])
    h["high_noise"] = {n: report[n]["sd_0.02"]["spline"]["within_one_tier"] for n in knee_names}
    h["no_knee_interior_share"] = {n: report[n][key]["spline"]["interior_share"]
                                   for n in ["smooth_linear", "smooth_exponential", "flat"]}
    return h


def figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    grid = np.linspace(0, 0.5, 200)
    for name, (fn, truth) in shapes().items():
        axes[0].plot(grid, fn(grid), label=name.replace("_", " "))
        if truth is not None:
            axes[0].axvline(truth, ls=":", lw=0.8, color="grey")
    axes[0].set(xlabel=r"corruption intensity $\eta$", ylabel="accuracy", title="(a) Synthetic trajectories")
    axes[0].legend(fontsize=7)
    for est, marker in [("spline", "o"), ("spline_tier", "s"), ("discrete", "^")]:
        for name in KNEES:
            vals = [report[f"{name}_knee"][f"sd_{sd}"][est]["within_one_tier"] for sd in NOISE_SDS]
            axes[1].plot(NOISE_SDS, vals, marker=marker, label=f"{est}, {name}", lw=1)
    axes[1].set(xlabel="noise SD added to each tier", ylabel="share within one tier of true knee",
                title="(b) Knee recovery", ylim=(0, 1.05))
    axes[1].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig("springer_project/figures/knee_validation.png", dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rep = run()
    json.dump({"detail": rep, "headline": headline(rep)}, open(os.path.join(OUT, "knee_validation.json"), "w"),
              indent=1, default=float)
    figure(rep)
    print(json.dumps(headline(rep), indent=1, default=float))
