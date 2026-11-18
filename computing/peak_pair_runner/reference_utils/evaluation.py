"""Sample-weighted logit cross-entropy, with no probability clipping."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

EVALUATOR_VERSION = 'logit_ce_sample_weighted_v2'

@dataclass
class CrossEntropyAccumulator:
    total_loss: float = 0.0
    num_examples: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
            raise ValueError('Expected classification logits [N,C] and labels [N].')
        if len(labels) == 0:
            return
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits.')
        # Double precision also avoids avoidable accumulation/softmax underflow.
        per_example = F.cross_entropy(logits.to(torch.float64), labels, reduction='none')
        if not torch.isfinite(per_example).all():
            raise FloatingPointError('Nonfinite logit cross-entropy.')
        self.total_loss = math.fsum((self.total_loss, float(per_example.sum().item())))
        self.num_examples += len(labels)

    @property
    def mean(self) -> float:
        if self.num_examples == 0:
            raise ValueError('Cannot evaluate an empty dataset.')
        return self.total_loss / self.num_examples
