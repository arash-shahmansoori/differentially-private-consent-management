import torch

from abc import ABC, abstractmethod


class Cost(ABC):
    """The interface for creating the different target cost for adversarial attacks.

    Notations:
        `B = number of batches`;
        `n_attributes = number of sensitive attributes`
    """

    @abstractmethod
    def get_cost(self, feat, target, kappa=0.0):
        """Implement the method to compute the cost according to the selection of target label.

        Args:
            - feat :math:`(B, n_attributes)`: The logits of the sensitive attribute classifier.
            - target: :math:`(B)`: The sensitive attributes (e.g., gender indices).
            - kappa (float): The parameter kappa (also written as 'confidence') in the paper. (Default: 0)
        """


class TargetedCostFGSM(Cost):
    """Cost for the targeted FGSM.

    Attributes:
        criterion: The cost function.
    """

    def __init__(self, criterion):

        self.criterion = criterion

    def get_cost(self, feat, target):
        """Compute the cost.

        Args:
            feat: The input features.
            target: The target labels.

        Return:
            cost: The cost.
        """

        cost = -self.criterion(feat, target)

        return cost


class NonTargetedCostFGSM(Cost):
    """The the non-targeted FGSM.

    Attributes:
        criterion: The cost function.
    """

    def __init__(self, criterion):

        self.criterion = criterion

    def get_cost(self, feat, target):
        """Compute the cost.

        Args:
            feat: The input features.
            target: The target labels.

        Return:
            cost: The cost.
        """

        cost = self.criterion(feat, target)

        return cost


class TargetedCostCW(Cost):
    """Cost for the targeted CW.

    Attributes:
        device: device to send the parameters to.
    """

    def __init__(self, device):

        self.device = device

    def get_cost(self, feat, target, kappa):
        """Compute the cost.

        Args:
            feat: The input features.
            target: The target labels.
            kappa: The confidence parameter.

        Return:
            The cost.
        """

        feat = feat.to("cpu")
        target = target.to("cpu")

        one_hot_labels = torch.eye(len(feat[0]))[target]

        # Get the largest logit
        j = torch.masked_select(feat, one_hot_labels.bool())

        # Get the second largest logit
        i, _ = torch.max((1 - one_hot_labels) * feat, dim=1)

        return torch.clamp((i - j), min=-kappa).sum().to(self.device)


class NonTargetedCostCW(Cost):
    """Cost for the non-targeted CW.

    Attributes:
        device: device to send the parameters to.
    """

    def __init__(self, device):

        self.device = device

    def get_cost(self, feat, target, kappa):
        """Compute the cost.

        Args:
            feat: The input features.
            target: The target labels.
            kappa: The confidence parameter.

        Return:
            The cost.
        """

        feat = feat.to("cpu")
        target = target.to("cpu")

        one_hot_labels = torch.eye(len(feat[0]))[target]

        # Get the largest logit
        j = torch.masked_select(feat, one_hot_labels.bool())

        # Get the second largest logit
        i, _ = torch.max((1 - one_hot_labels) * feat, dim=1)

        return torch.clamp((j - i), min=-kappa).sum().to(self.device)


class NonTargetedCostCWDP(Cost):
    """Cost for the non-targeted CW.

    Attributes:
        device: device to send the parameters to.
    """

    def __init__(self, device):

        self.device = device

    def get_cost(self, feat, target, kappa):
        """Compute the cost.

        Args:
            feat: The input features.
            target: The target labels.
            kappa: The confidence parameter.

        Return:
            The cost.
        """

        feat = feat.to("cpu")
        target = target.to("cpu")

        one_hot_labels = torch.eye(len(feat[0]))[target]

        logits_second = torch.matmul((1 - one_hot_labels), feat)

        # Get the largest logit
        j = torch.masked_select(feat, one_hot_labels.bool())
        # Get the second largest logit
        i, _ = torch.max(logits_second, dim=1)

        return torch.clamp((j - i), min=-kappa).sum().to(self.device)
