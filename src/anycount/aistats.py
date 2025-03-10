import numpy as np
from fn import F, _
from fn.iters import iterate, dropwhile, head
from scipy.optimize import brentq
from scipy.special import gammaln


def log_bayes_factor(N, phi, Lam_0):
    return (
        phi * np.log(phi)
        - (phi + N) * np.log(phi + Lam_0)
        + gammaln(phi + N)
        - gammaln(phi)
        + Lam_0
    )


def confidence_sequence(N, phi, alpha=0.05):
    """
    Given N(t), the number of counts in [0, t], provides a 1-alpha confidence interval on

    𝔼[N(t)] = Λ(t) = ∫₀ᵗ λ(s) ds
    """

    def root(x):
        return log_bayes_factor(N, phi, x) + np.log(alpha)

    upper_bound = head(dropwhile(lambda x: root(x) < 0, iterate(_ + N, N)))
    upper = brentq(root, N, upper_bound)
    if root(0) < 0:
        lower = 0
    else:
        lower = brentq(root, 0, N)
    return lower, upper


def A_confidence_sequence(N_A, N_B, phi, alpha=0.05):
    K = (
        -np.log(alpha)
        - 2 * phi * np.log(phi)
        + (phi + N_B) * np.log(phi + N_B)
        - N_B
        + 2 * gammaln(phi)
        - gammaln(phi + N_A)
        - gammaln(phi + N_B)
    )

    def root(Lambda_A):
        return Lambda_A - (phi + N_A) * np.log(phi + Lambda_A) - K

    upper_bound = head(dropwhile(lambda x: root(x) < 0, iterate(_ + N_A, N_A)))
    lower_bound = 0
    upper = brentq(root, N_A, upper_bound)
    lower = brentq(root, lower_bound, N_A) if N_A > 0 else 0
    return lower, upper


def B_confidence_sequence(N_A, N_B, phi, alpha=0.05):
    K = (
        -np.log(alpha)
        - 2 * phi * np.log(phi)
        + (phi + N_A) * np.log(phi + N_A)
        - N_A
        + 2 * gammaln(phi)
        - gammaln(phi + N_A)
        - gammaln(phi + N_B)
    )

    def root(x):
        return x - (phi + N_B) * np.log(phi + x) - K

    upper_bound = head(dropwhile(lambda x: root(x) < 0, iterate(_ + N_B, N_B)))
    lower_bound = 0
    upper = brentq(root, N_B, upper_bound)
    lower = brentq(root, lower_bound, N_B) if N_B > 0 else 0
    return lower, upper


def AB_confidence_sequence(N_A, N_B, phi, alpha=0.05):
    def g(w, v):
        term1 = v
        term2 = (phi + N_B) * np.log(phi + 0.5 * (v + w))
        term3 = (phi + N_A) * np.log(phi + 0.5 * (v - w))
        return term1 - term2 - term3

    def h(w):
        term1 = 0.5 * (N_A + N_B) - phi
        term2 = np.sqrt(
            (0.25 * N_A**2)
            + (0.5 * N_A * N_B)
            + (N_A * phi)
            + (N_A * w)
            + (0.25 * N_B**2)
            + (N_B * phi)
            - (N_B * w)
            + (phi**2)
            + (w**2)
        )
        return term1 + term2

    K = (
        np.log(1 / alpha)
        - 2 * phi * np.log(phi)
        + 2 * gammaln(phi)
        - gammaln(phi + N_A)
        - gammaln(phi + N_B)
    )

    def root(x):
        return g(x, h(x)) - K

    step = np.abs(np.abs(N_B - N_A), 1)
    upper_bound = head(
        dropwhile(lambda x: root(x) < 0, iterate(_ + step, N_B - N_A))
    )
    lower_bound = head(
        dropwhile(lambda x: root(x) < 0, iterate(_ - step, N_B - N_A))
    )
    upper = brentq(root, N_B - N_A, upper_bound)
    lower = brentq(root, lower_bound, N_B - N_A)
    return lower, upper


def AB_log_sequential_p_value(N_A, N_B, phi):
    Lambda_estimate = (N_A + N_B) / 2
    return -log_bayes_factor(N_A, phi, Lambda_estimate) - log_bayes_factor(
        N_B, phi, Lambda_estimate
    )


def AB_sequential_p_value(N_A, N_B, phi):
    return np.exp(AB_log_sequential_p_value(N_A, N_B, phi))


def compute_stats(N_A, N_B, phi, alpha=0.05):
    return AB_confidence_sequence(N_A, N_B, phi, alpha) | {
        "AB_sequential_p_value": AB_sequential_p_value(N_A, N_B, phi)
    }
