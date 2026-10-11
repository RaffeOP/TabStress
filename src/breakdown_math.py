import numpy as np
from scipy.interpolate import CubicSpline

def compute_d_rate(acc_clean: float, acc_noisy: float, delta_eta: float = 0.20) -> float:
    return float((acc_clean - acc_noisy) / delta_eta)

def compute_audc(noise_levels: np.ndarray, accuracies: np.ndarray) -> float:
    sort_idx = np.argsort(noise_levels)
    x = noise_levels[sort_idx]
    y = accuracies[sort_idx]
    eta_max = float(x[-1] - x[0])
    if eta_max <= 0.0:
        return float(y[0])
    area = np.trapezoid(y, x)
    return float(area / eta_max)

def compute_mrr(noise_levels: np.ndarray, accuracies: np.ndarray) -> float:
    audc_val = compute_audc(noise_levels, accuracies)
    sort_idx = np.argsort(noise_levels)
    base_acc = float(accuracies[sort_idx][0])
    if base_acc <= 0.0:
        return 0.0
    return float(audc_val / base_acc)

def compute_critical_breakdown_threshold(noise_levels: np.ndarray, accuracies: np.ndarray, smoothing: float = 0.001) -> dict:
    noise_levels = np.asarray(noise_levels, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    sort_idx = np.argsort(noise_levels)
    x = noise_levels[sort_idx]
    y = accuracies[sort_idx]

    # Natural cubic spline enforces C^2 continuity and zero-curvature boundary conditions S''(x_0) = S''(x_n) = 0
    # Eliminating Runge oscillation and preventing boundary knot collapse
    cs = CubicSpline(x, y, bc_type="natural")

    x_dense = np.linspace(x.min(), x.max(), 500)
    y_dense = cs(x_dense)
    d1_dense = cs(x_dense, 1)
    d2_dense = cs(x_dense, 2)

    # Differential geometry curvature: kappa(eta) = |S''(eta)| / (1 + (S'(eta))^2)^(3/2)
    curv_dense = np.abs(d2_dense) / np.power(1.0 + np.square(d1_dense), 1.5)
    max_idx = np.argmax(curv_dense)
    eta_star = float(x_dense[max_idx])
    max_curvature = float(curv_dense[max_idx])

    idx_clean = np.argmin(np.abs(x - 0.0))
    idx_20 = np.argmin(np.abs(x - 0.20))
    d_rate = compute_d_rate(y[idx_clean], y[idx_20], delta_eta=0.20)
    audc_val = compute_audc(x, y)
    mrr_val = compute_mrr(x, y)

    return {
        "eta_star": eta_star,
        "max_curvature": max_curvature,
        "d_rate": d_rate,
        "audc": audc_val,
        "mrr": mrr_val,
        "x_dense": x_dense,
        "y_dense": y_dense,
        "d1_dense": d1_dense,
        "d2_dense": d2_dense
    }
