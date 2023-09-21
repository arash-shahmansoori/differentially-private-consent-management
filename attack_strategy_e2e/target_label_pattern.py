import torch

from abc import ABC, abstractmethod


class TargetLabel(ABC):
    """The interface for creating the different target label for adversarial attacks.

    Notations:
        `B = number of batches`;
        `e_dim = latent dimension`;
    """

    @abstractmethod
    def get_target_label(self, *args, **kwargs):
        """Implement the method to compute the different target label."""


class NonTargetLabelStrategy(TargetLabel):
    """The non-target label strategy.

    Attributes:
        kwargs: The keyword arguments
    """

    def __init__(self, device):

        self.device = device

    def get_target_label(self, **kwargs):
        """
        Args:
            kwargs => The keyword arguments including:
                target: :math:`(B)`: The sensitive attributes (e.g., gender indices).

        Return:
            The target.
        """

        return kwargs["target"].to(self.device)


class LeastLikelyTargetLabelStrategy(TargetLabel):
    """Compute least likely label strategy.

    Attributes:
        kwargs => The keyword arguments including:
            model: The model.
            device: The device to send the parameters to.
            labels: The provided labels. (Default: None)
            kth_min (int): The hyper-parameter to obtain the k-th value by:
            `torch.kthvalue(..., kth_min)`. (Default: 1)
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.device = kwargs["device"]
        self.labels = kwargs["labels"]
        self.kth_min = kwargs["kth_min"]

    @torch.no_grad()
    def get_target_label(self, **kwargs):
        """Obtain the target label.

        Args:
            kwargs => The keyword arguments including:
                z: input features.

        Return:
            The target labels.
        """

        z = kwargs["z"]

        _, feat_out = self.model(z)

        feat_out.to("cpu")

        if self.labels is None:
            _, self.labels = torch.max(feat_out, dim=1)
        n_classses = feat_out.shape[-1]

        target_labels = torch.zeros_like(self.labels)

        for counter in range(self.labels.shape[0]):

            l = list(range(n_classses))
            l.remove(self.labels[counter])
            _, t = torch.kthvalue(feat_out[counter][l], self.kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)


class RandomTargetLabelStrategy(TargetLabel):
    """Compute random target label strategy.

    Attributes:
        model (nn.Module): The model to attack.
        device: The device to send the parameters to.
        labels: The provided labels. (Default: None)
    """

    def __init__(self, model, **kwargs):

        self.model = model
        self.device = kwargs["device"]
        self.labels = kwargs["labels"]

    @torch.no_grad()
    def get_target_label(self, **kwargs):
        """Obtain the target label.

        Args:
            kwargs => The keyword arguments including:
                z: input features.

        Return:
            The target labels.
        """

        z = kwargs["z"]

        _, feat_out = self.model(z)

        if self.labels is None:
            _, self.labels = torch.max(feat_out, dim=1)
        n_classses = feat_out.shape[-1]

        target_labels = torch.zeros_like(self.labels)
        for counter in range(self.labels.shape[0]):
            l = list(range(n_classses))
            l.remove(self.labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)
