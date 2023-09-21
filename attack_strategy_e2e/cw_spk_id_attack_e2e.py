import torch
import torch.nn as nn


from torch.optim import Adam


class CWSpkIDE2E:
    r"""CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Notations:
        `B = number of batches`;
        `latent_dim = latent dimension`;
        `dim_emb = embedding dimension`

    Distance Measure : L2

    Attributes:
        model (nn.Module): model to attack.

        device: device to send the parameters to.

        target_strategy: strategy to create the target label for adversarial attack.
        cost_strategy: strategy to compute the cost according to targeted/non-targeted scenario.


        hparams including:
            c (float): c in the paper. parameter for box-constraint.
                :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
            kappa (float): kappa (also written as 'confidence') in the paper.
                :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
            num_steps (int): number of steps.
            lr_cls (float): learning rate of the Adam optimizer.
    """

    def __init__(
        self,
        model,
        device,
        target_strategy,
        cost_strategy,
        hparams,
    ):
        self.model = model

        self.device = device

        self.target_strategy = target_strategy
        self.cost_strategy = cost_strategy

        self.c = hparams.c
        self.kappa = hparams.kappa
        self.num_steps = hparams.num_steps
        self.lr_cls = hparams.lr_cls

    def __call__(self, x, labels):
        """The forward method to compute the adversarial features.

        Args:
            - x: :math:`(B, seg_length, dim)`: The input samples.
            - labels: :math:`(B)`: The speaker IDs.

        Return:
            best_adv :math:`(B, seg_length, dim)`: The adversarial input samples.
            x :math:`(B, seg_length, dim)`: The normal input samples.
        """

        wx = x.detach()
        wx.requires_grad = True

        best_adv = x.clone().detach()
        best_L2 = 1e10 * torch.ones((len(x))).to(self.device)
        prev_cost = 1e10
        dim = len(x.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = Adam([wx], lr=self.lr_cls)

        for step in range(self.num_steps):
            # Get adversarial features
            adv_x = wx

            # Calculate loss
            current_L2 = MSELoss(
                Flatten(adv_x.view(x.shape[0], -1)), Flatten(x.view(x.shape[0], -1))
            ).sum(dim=1)

            L2_loss = current_L2.sum()

            _, feat_out, _, _ = self.model(adv_x)

            f_loss = self.cost_strategy.get_cost(feat_out, labels, self.kappa)

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial features
            _, pre = torch.max(feat_out.detach(), 1)
            correct = (pre == labels).float()

            # filter out features that get either correct predictions or non-decreasing loss,
            # i.e., only features that are both misclassified and loss-decreasing are left
            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv = mask * adv_x.detach() + (1 - mask) * best_adv

            # Early stop when loss does not converge.
            if (
                step % max(self.num_steps // 10, 1) == 0
            ):  # max(.,1) To prevent MODULO BY ZERO error in the next step.
                if cost.item() > prev_cost:
                    return best_adv, x
                prev_cost = cost.item()

        return best_adv, x
