import copy
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize_scalar


def compute_spectral_norm_power_iteration(weight: torch.Tensor, num_iters: int = 20) -> float:
    if weight.dim() != 2:
        weight = weight.reshape(weight.size(0), -1)

    with torch.no_grad():
        u = torch.randn(weight.size(0), device=weight.device)
        u = u / (u.norm() + 1e-8)

        for _ in range(num_iters):
            v = weight.t() @ u
            v = v / (v.norm() + 1e-8)
            u = weight @ v
            u = u / (u.norm() + 1e-8)

        sigma = (u @ (weight @ v)).item()

    return abs(sigma)


def compute_model_norms(model: nn.Module, exclude_embedding: bool = True):
    norms = []
    for name, module in model.named_modules():
        if exclude_embedding and "embedding" in name.lower():
            continue
        if isinstance(module, nn.Linear):
            weight = module.weight.data.detach().cpu()
            norm = compute_spectral_norm_power_iteration(weight, num_iters=20)
            norms.append(norm)

    product = np.prod(norms) if norms else 1.0
    sum_norms = np.sum(norms) if norms else 0.0

    return {
        "norms": norms,
        "product": float(product),
        "sum": float(sum_norms),
        "count": len(norms)
    }


def compute_rademacher_bound(norms, n_samples: int, d_model: int) -> float:
    product = norms["product"]
    bound = (2.0 / np.sqrt(n_samples)) * product * np.sqrt(d_model)
    return float(bound)


def _classification_error(model: nn.Module, data_loader, device: str) -> float:
    model.eval()
    total = 0
    wrong = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            wrong += (preds != labels).sum().item()
            total += labels.size(0)

    return wrong / total


def compute_pacbayes_bound(model: nn.Module, data_loader, device: str = "cpu",
                           sigma: float = 0.01, delta: float = 0.05,
                           posterior_samples: int = 3):
    model = model.to(device)
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    original_state = copy.deepcopy(model.state_dict())

    squared_norm = 0.0
    num_params = 0
    for p in params:
        squared_norm += (p.detach() ** 2).sum().item()
        num_params += p.numel()

    kl = squared_norm / (2.0 * sigma ** 2)

    risks = []
    for _ in range(posterior_samples):
        sampled_state = {}
        for name, tensor in original_state.items():
            if tensor.dtype.is_floating_point:
                noise = torch.randn_like(tensor) * sigma
                sampled_state[name] = tensor + noise
            else:
                sampled_state[name] = tensor.clone()

        model.load_state_dict(sampled_state)
        risks.append(_classification_error(model, data_loader, device))

    model.load_state_dict(original_state)

    empirical_gibbs_risk = float(np.mean(risks))
    n_samples = len(data_loader.dataset)
    complexity = (kl + np.log((2.0 * np.sqrt(n_samples)) / delta)) / (2.0 * max(1, n_samples - 1))
    bound = empirical_gibbs_risk + np.sqrt(complexity)

    return {
        "bound": float(bound),
        "kl": float(kl),
        "empirical_gibbs_risk": float(empirical_gibbs_risk),
        "sigma": float(sigma),
        "posterior_samples": int(posterior_samples),
        "num_params": int(num_params)
    }


def compute_margin_bound(model: nn.Module, data_loader, device: str = "cpu", gamma: float = 0.1):
    model = model.to(device)
    model.eval()

    classifier_weight = model.classifier.weight.detach()
    weight_norm = torch.norm(classifier_weight, p="fro").item()

    total = 0
    margin_violations = 0
    max_feature_norm = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            features = model.encode(inputs)
            logits = model.classifier(features)

            correct_scores = logits[torch.arange(len(labels), device=device), labels]
            competitor_scores = logits.clone()
            competitor_scores[torch.arange(len(labels), device=device), labels] = -float("inf")
            max_wrong_scores = competitor_scores.max(dim=1)[0]

            margins = correct_scores - max_wrong_scores
            margin_violations += (margins <= gamma).sum().item()
            total += labels.size(0)

            batch_feature_norm = torch.norm(features, p=2, dim=1).max().item()
            max_feature_norm = max(max_feature_norm, batch_feature_norm)

    margin_error = margin_violations / total
    complexity_term = (2.0 * weight_norm * max_feature_norm) / (gamma * np.sqrt(total))
    bound = margin_error + complexity_term

    return {
        "bound": float(bound),
        "margin_error": float(margin_error),
        "weight_norm": float(weight_norm),
        "max_feature_norm": float(max_feature_norm),
        "gamma": float(gamma)
    }


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
        scaled_logits = logits / max(T, 1e-8)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        log_probs = np.log(probs[np.arange(len(labels)), labels])
        return -np.mean(log_probs)

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def mean_ci95(values):
    values = np.array(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    n = len(values)
    half_width = 1.96 * std / np.sqrt(max(n, 1))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width)
    }
