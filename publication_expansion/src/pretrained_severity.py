"""First-transition LR severity diagnostic for pretrained experiments."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

from publication_expansion.src.anchor import (
    batch_fingerprint,
    gradient_fingerprint,
    model_fingerprint,
    optimizer_non_lr_fingerprint,
)
from publication_expansion.src.pretrained_job_runner import (
    NUM_EPOCHS,
    TASK_CONFIG,
    atomic_json,
    forward_batch,
    linear_warmup_decay_lr,
    prepare_task,
    set_seed,
)
from publication_expansion.src.pretrained_model_adapter import (
    build_pretrained_adamw,
)
from publication_expansion.src.scratch_data_contract import (
    make_dataloader,
)


MULTIPLIERS = (
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
)

WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0


def severity_candidates(
    *,
    intended_lr: float,
    peak_lr: float,
):
    candidates = [
        {
            "label": f"{multiplier:g}x_intended",
            "multiplier":
                float(multiplier),
            "executed_lr":
                float(
                    intended_lr
                    * multiplier
                ),
            "natural_late_schedule":
                False,
        }
        for multiplier in MULTIPLIERS
    ]

    candidates.append(
        {
            "label":
                "natural_late_schedule_peak",
            "multiplier":
                float(
                    peak_lr
                    / intended_lr
                ),
            "executed_lr":
                float(peak_lr),
            "natural_late_schedule":
                True,
        }
    )

    return candidates


def clone_gradients(model):
    return {
        name:
            None
            if parameter.grad is None
            else parameter.grad.detach()
            .clone()
        for name, parameter
        in model.named_parameters()
    }


def restore_gradients(
    model,
    gradients,
):
    for name, parameter in (
        model.named_parameters()
    ):
        grad = gradients[name]

        parameter.grad = (
            None
            if grad is None
            else grad.to(
                parameter.device
            ).clone()
        )


def update_ratio(
    model,
    before_params,
) -> float:
    update_sq = 0.0
    parameter_sq = 0.0

    with torch.no_grad():
        for name, parameter in (
            model.named_parameters()
        ):
            if (
                not parameter.requires_grad
                or name not in before_params
            ):
                continue

            before = before_params[
                name
            ].to(parameter.device)

            diff = (
                parameter.detach()
                - before
            )

            update_sq += float(
                torch.sum(
                    diff * diff
                ).item()
            )

            parameter_sq += float(
                torch.sum(
                    before * before
                ).item()
            )

    return float(
        math.sqrt(update_sq)
        / (
            math.sqrt(parameter_sq)
            + 1e-12
        )
    )


def run_severity(args) -> int:
    if args.dataset not in TASK_CONFIG:
        raise ValueError(
            "unsupported dataset"
        )

    if args.seed not in {
        42,
        123,
        456,
        789,
        2026,
    }:
        raise ValueError(
            "seed not in frozen seed set"
        )

    cfg = TASK_CONFIG[
        args.dataset
    ]

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    result_dir = (
        Path(args.output_root)
        / args.dataset
        / f"seed{args.seed}"
    )

    result_file = (
        result_dir / "result.json"
    )

    if (
        result_file.exists()
        and not args.force
    ):
        print(
            "SKIP completed severity case:",
            args.dataset,
            args.seed,
        )
        return 0

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    prepared, model = prepare_task(
        args.dataset
    )

    model = model.to(device)

    loader = make_dataloader(
        prepared["train"],
        batch_size=cfg[
            "train_batch_size"
        ],
        shuffle=True,
        experiment_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    total_steps = (
        len(loader)
        * NUM_EPOCHS
    )

    warmup_steps = max(
        1,
        math.ceil(
            total_steps
            * WARMUP_RATIO
        ),
    )

    peak_lr = float(
        cfg["peak_lr"]
    )

    intended_lr = (
        linear_warmup_decay_lr(
            1,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    )

    optimizer = (
        build_pretrained_adamw(
            model,
            initial_lr=peak_lr,
        )
    )

    batch = next(iter(loader))

    optimizer.zero_grad(
        set_to_none=True
    )

    pre_model_hash = (
        model_fingerprint(model)
    )

    pre_optimizer_hash = (
        optimizer_non_lr_fingerprint(
            optimizer
        )
    )

    batch_hash = (
        batch_fingerprint(batch)
    )

    logits, _, loss = (
        forward_batch(
            task=cfg["task"],
            model=model,
            batch=batch,
            device=device,
        )
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "non-finite diagnostic loss"
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP,
    )

    gradient_hash = (
        gradient_fingerprint(model)
    )

    gradients = clone_gradients(
        model
    )

    initial_model_state = (
        copy.deepcopy(
            model.state_dict()
        )
    )

    initial_optimizer_state = (
        copy.deepcopy(
            optimizer.state_dict()
        )
    )

    before_params = {
        name:
            parameter.detach()
            .clone()
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    trials = []

    for candidate in (
        severity_candidates(
            intended_lr=intended_lr,
            peak_lr=peak_lr,
        )
    ):
        model.load_state_dict(
            initial_model_state
        )

        optimizer.load_state_dict(
            initial_optimizer_state
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        restore_gradients(
            model,
            gradients,
        )

        executed_lr = float(
            candidate[
                "executed_lr"
            ]
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = (
                executed_lr
            )

        optimizer.step()

        ratio = update_ratio(
            model,
            before_params,
        )

        trials.append(
            {
                **candidate,
                "update_ratio":
                    float(ratio),
            }
        )

    baseline = trials[0][
        "update_ratio"
    ]

    for trial in trials:
        trial[
            "update_ratio_relative_to_1x"
        ] = float(
            trial["update_ratio"]
            / (
                baseline
                + 1e-30
            )
        )

    result = {
        "schema_version": 1,
        "diagnostic":
            "first_transition_lr_severity",
        "dataset":
            args.dataset,
        "task":
            cfg["task"],
        "seed":
            int(args.seed),
        "device":
            str(device),
        "peak_lr":
            peak_lr,
        "total_steps":
            int(total_steps),
        "warmup_steps":
            int(warmup_steps),
        "intended_step1_lr":
            float(intended_lr),
        "natural_late_multiplier":
            float(
                peak_lr
                / intended_lr
            ),
        "gradient_clip":
            GRAD_CLIP,
        "same_transition_anchor":
            {
                "model_state":
                    pre_model_hash,
                "optimizer_non_lr":
                    pre_optimizer_hash,
                "first_batch":
                    batch_hash,
                "clipped_gradient":
                    gradient_hash,
                "loss":
                    float(
                        loss.item()
                    ),
            },
        "trials":
            trials,
    }

    atomic_json(
        result_file,
        result,
    )

    print(
        "SEVERITY:",
        args.dataset,
        "seed",
        args.seed,
    )

    print(
        "intended step1 LR:",
        intended_lr,
    )

    print(
        "natural late multiplier:",
        result[
            "natural_late_multiplier"
        ],
    )

    for trial in trials:
        print(
            f"{trial['label']:<28} "
            f"lr={trial['executed_lr']:.6e} "
            f"ratio={trial['update_ratio']:.6e} "
            f"relative="
            f"{trial['update_ratio_relative_to_1x']:.3f}x"
        )

    return 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            TASK_CONFIG
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--output-root",
        default=(
            "publication_expansion/"
            "results/pretrained_severity"
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main():
    args = (
        build_parser()
        .parse_args()
    )

    raise SystemExit(
        run_severity(args)
    )


if __name__ == "__main__":
    main()
