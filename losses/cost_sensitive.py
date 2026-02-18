"""
Cost-Sensitive Loss: Penaliza FN lambda vezes mais que FP.

Uso:
    criterion = CostSensitiveCrossEntropy(lambda_risk=20)
    loss = criterion(logits, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveCrossEntropy(nn.Module):
    """
    Cross-Entropy com pesos baseados em custo clinico.

    Args:
        lambda_risk: razao de custo C_FN / C_FP (padrao: 20)
        reduction: 'mean', 'sum', ou 'none'

    Pesos:
        - Classe 0 (negativo/saudavel): peso = 1.0
        - Classe 1 (positivo/doenca): peso = lambda_risk

    Isso faz com que erros na classe positiva (FN) sejam
    penalizados lambda vezes mais que erros na classe negativa (FP).
    """

    def __init__(self, lambda_risk=20.0, reduction='mean'):
        super().__init__()
        self.lambda_risk = lambda_risk
        self.reduction = reduction
        self.register_buffer('weights', torch.tensor([1.0, float(lambda_risk)]))

    def forward(self, logits, labels):
        return F.cross_entropy(
            logits, labels,
            weight=self.weights.to(logits.device),
            reduction=self.reduction
        )

    def __repr__(self):
        return f"CostSensitiveCrossEntropy(lambda={self.lambda_risk})"


def cost_sensitive_cross_entropy(logits, labels, lambda_risk=20.0, reduction='mean'):
    """Versao funcional."""
    weights = torch.tensor([1.0, float(lambda_risk)], device=logits.device)
    return F.cross_entropy(logits, labels, weight=weights, reduction=reduction)
