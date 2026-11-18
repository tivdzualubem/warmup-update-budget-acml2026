"""
Calibration and evaluation metrics for pretrained baselines.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        if bin_mask.sum() > 0:
            bin_acc = accuracies[bin_mask].mean()
            bin_conf = confidences[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(labels)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def multiclass_nll(labels: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels])))


def multiclass_brier(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def temperature_scaling(labels: np.ndarray, logits: np.ndarray):
    def nll_loss(T):
        T = max(T, 1e-8)
        scaled_logits = logits / T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    optimal_T = result.x

    scaled_logits = logits / optimal_T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return scaled_probs, float(optimal_T)
