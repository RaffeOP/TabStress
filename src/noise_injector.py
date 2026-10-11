"""
TabStress: Multi-Modal Synthetic Noise Injection Engine (MSNIE)
Implements 4 parameterized noise modalities across feature tensors and label vectors:
1. Symmetric (uniform) label noise
2. Asymmetric (confusion matrix) label noise
3. Additive Gaussian feature jitter
4. Missing Completely At Random (MCAR) feature masking
"""

import numpy as np

class MSNIE:
    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def inject_symmetric_label_noise(self, y: np.ndarray, eta: float, num_classes: int = None) -> np.ndarray:
        """
        Flips a fraction eta of labels to any other class with uniform probability:
        P(y_tilde = j | y = i) = (1 - eta) if j == i else eta / (K - 1)
        """
        if eta <= 0.0:
            return y.copy()
        
        y_noisy = y.copy()
        n_samples = len(y)
        classes = np.unique(y) if num_classes is None else np.arange(num_classes)
        k = len(classes)
        if k <= 1:
            return y_noisy

        # Select indices to corrupt
        corrupt_mask = self.rng.random(n_samples) < eta
        corrupt_indices = np.where(corrupt_mask)[0]

        for idx in corrupt_indices:
            orig_c = y[idx]
            other_classes = [c for c in classes if c != orig_c]
            if other_classes:
                y_noisy[idx] = self.rng.choice(other_classes)

        return y_noisy

    def inject_asymmetric_label_noise(self, y: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Transitions labels according to an explicit class confusion transition matrix T:
        T_ij = P(y_tilde = j | y = i)
        """
        y_noisy = y.copy()
        n_samples = len(y)
        k = transition_matrix.shape[0]

        for i in range(k):
            indices = np.where(y == i)[0]
            if len(indices) > 0:
                probs = transition_matrix[i]
                probs = probs / np.sum(probs)
                y_noisy[indices] = self.rng.choice(k, size=len(indices), p=probs)

        return y_noisy

    def inject_asymmetric_label_noise_eta(self, y: np.ndarray, eta: float, num_classes: int = None) -> np.ndarray:
        """
        Transitions labels according to standard asymmetric pair-flip matrix parameterized by eta:
        T_ii = 1 - eta, T_i,(i+1)%K = eta
        """
        if eta <= 0.0:
            return y.copy()
        classes = np.unique(y) if num_classes is None else np.arange(num_classes)
        k = len(classes)
        if k <= 1:
            return y.copy()
        
        # Build pair-flip transition matrix
        T = np.eye(k) * (1.0 - eta)
        for i in range(k):
            T[i, (i + 1) % k] += eta
        
        return self.inject_asymmetric_label_noise(y, T)

    @staticmethod
    def minority_class(y: np.ndarray):
        """The less frequent of two classes. On an exact tie the class with the larger label is
        returned: argmin and argmax of equal counts both select the first class, which would make
        minority and majority the same class and the corruption a no-op."""
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("minority-to-majority noise is defined for binary labels only")
        return classes[1] if counts[0] == counts[1] else classes[np.argmin(counts)]

    def inject_minority_to_majority_noise(self, y: np.ndarray, eta: float, minority=None) -> np.ndarray:
        """
        One-directional class-conditional noise for binary tasks: each label of the
        minority class is relabelled as the other class with probability eta; the other
        class is never touched. T = [[1, 0], [eta, 1 - eta]] with rows ordered (majority, minority).
        minority: the class to corrupt. Callers fix it once per dataset (minority_class of the full
        label vector) so that every fold corrupts the same class; if None it is taken from y.
        """
        if eta <= 0.0:
            return y.copy()
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("minority-to-majority noise is defined for binary labels only")
        minority = self.minority_class(y) if minority is None else minority
        if minority not in classes:
            raise ValueError(f"class {minority!r} is not present in y")
        majority = classes[classes != minority][0]
        y_noisy = y.copy()
        draws = self.rng.random(len(y))
        y_noisy[(y == minority) & (draws < eta)] = majority
        return y_noisy

    def inject_gaussian_feature_noise(self, X: np.ndarray, gamma: float) -> np.ndarray:
        """
        Injects zero-mean additive Gaussian perturbations scaled by feature variance:
        x_tilde_ij = x_ij + epsilon_ij, epsilon_ij ~ N(0, (gamma * sigma_j)^2)
        """
        if gamma <= 0.0:
            return X.copy()
        
        X_noisy = X.copy().astype(np.float64)
        # A column that is constant in X has sigma_j = 0 and receives no noise, as the definition
        # requires. (An earlier version substituted sigma_j = 1 there, which turned constant columns
        # into pure-noise features.) One normal draw is consumed per entry whatever its scale, so the
        # noise on every other column is unchanged.
        sigmas = np.nanstd(X_noisy, axis=0)

        noise = self.rng.normal(loc=0.0, scale=gamma * sigmas, size=X_noisy.shape)
        return X_noisy + noise

    def inject_mcar_missingness(self, X: np.ndarray, h: float) -> np.ndarray:
        """
        Replaces a fraction h of matrix entries with NaN (Missing Completely At Random):
        P(x_tilde_ij = NaN) = h
        """
        if h <= 0.0:
            return X.copy()
        
        X_corrupt = X.copy().astype(np.float64)
        mask = self.rng.random(X_corrupt.shape) < h
        X_corrupt[mask] = np.nan
        return X_corrupt

    def inject_mar_missingness(self, X: np.ndarray, h: float, driver: int) -> np.ndarray:
        """
        Missing At Random: the probability that x_ij is masked grows with the rank of the fully
        observed driver column d in row i, P(x_ij = NaN) = 2 h r_i / (n + 1) with r_i the rank of x_id.
        The mean masking rate is h and the maximum 2h <= 1 for h <= 0.5; column d is never masked.
        """
        if h <= 0.0:
            return X.copy()
        X_corrupt = X.copy().astype(np.float64)
        ranks = _average_ranks(X_corrupt[:, driver])
        p_row = 2.0 * h * ranks / (len(ranks) + 1.0)
        mask = self.rng.random(X_corrupt.shape) < p_row[:, None]
        mask[:, driver] = False
        X_corrupt[mask] = np.nan
        return X_corrupt

    def inject_mnar_missingness(self, X: np.ndarray, h: float) -> np.ndarray:
        """
        Missing Not At Random (self-masking): the probability that x_ij is masked grows with the rank
        of x_ij within column j, P(x_ij = NaN) = 2 h r_ij / (n + 1). Large values go missing more often,
        so the missingness depends on the unobserved value itself; the mean rate is h.
        """
        if h <= 0.0:
            return X.copy()
        X_corrupt = X.copy().astype(np.float64)
        n = X_corrupt.shape[0]
        ranks = np.column_stack([_average_ranks(X_corrupt[:, j]) for j in range(X_corrupt.shape[1])])
        mask = self.rng.random(X_corrupt.shape) < 2.0 * h * ranks / (n + 1.0)
        X_corrupt[mask] = np.nan
        return X_corrupt

    def inject_type_aware_feature_noise(self, X: np.ndarray, gamma: float, discrete: np.ndarray,
                                        reference: np.ndarray) -> np.ndarray:
        """
        Type-aware feature corruption: continuous columns receive the Gaussian operator with scale
        gamma * sigma_j; each entry of a discrete (integer-coded: one-hot, ordinal, count) column is,
        with probability gamma, replaced by a draw from that column's empirical marginal in
        `reference` (the clean training data), so corrupted values stay valid category codes.
        """
        if gamma <= 0.0:
            return X.copy()
        X_noisy = self.inject_gaussian_feature_noise(X, gamma)
        X_noisy[:, discrete] = X[:, discrete]
        for j in np.where(discrete)[0]:
            hit = self.rng.random(X.shape[0]) < gamma
            X_noisy[hit, j] = self.rng.choice(reference[:, j], size=int(hit.sum()))
        return X_noisy


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (scipy.stats.rankdata semantics)."""
    from scipy.stats import rankdata
    return rankdata(values, method="average")


def resample_categorical_groups(rng, X_out: np.ndarray, X: np.ndarray, eta: float, groups, reference: np.ndarray):
    """Replace, with probability eta per row and attribute, the encoded value of each categorical attribute
    (all its one-hot columns jointly, or its ordinal code) by the encoding of a row drawn from `reference`,
    so that a corrupted entry is always a valid category and one-hot groups stay one-hot."""
    for cols in groups:
        hit = np.where(rng.random(X.shape[0]) < eta)[0]
        if len(hit):
            donors = rng.integers(0, reference.shape[0], size=len(hit))
            X_out[np.ix_(hit, cols)] = reference[np.ix_(donors, cols)]
    return X_out


def categorical_aware_noise(injector: "MSNIE", X: np.ndarray, eta: float, groups, reference: np.ndarray,
                            continuous: bool = True) -> np.ndarray:
    """Categorical-aware feature corruption. Continuous attributes receive the Gaussian operator (if
    `continuous`), categorical attributes are resampled as a whole from the clean training marginal."""
    if eta <= 0.0:
        return X.copy()
    cat_cols = np.concatenate(groups) if groups else np.array([], dtype=int)
    out = injector.inject_gaussian_feature_noise(X, eta) if continuous else X.copy().astype(np.float64)
    out[:, cat_cols] = X[:, cat_cols]
    return resample_categorical_groups(injector.rng, out, X, eta, groups, reference)


def continuous_gaussian_noise(injector: "MSNIE", X: np.ndarray, eta: float, groups) -> np.ndarray:
    """Gaussian operator restricted to continuous attributes; nominal attributes are left clean."""
    if eta <= 0.0:
        return X.copy()
    cat_cols = np.concatenate(groups) if groups else np.array([], dtype=int)
    out = injector.inject_gaussian_feature_noise(X, eta)
    out[:, cat_cols] = X[:, cat_cols]
    return out
