import torch
import torch.nn as nn


from torch.optim import Adam


class CWSpkIDV2:
    r"""CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Notations:
        `B = number of batches`;
        `latent_dim = latent dimension`;
        `dim_emb = embedding dimension`

    Distance Measure : L2

    Attributes:
        shadow_model (nn.Module): shadow model to attack.

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
        shadow_model,
        device,
        target_strategy,
        cost_strategy,
        hparams,
    ):
        self.shadow_model = shadow_model

        self.device = device

        self.target_strategy = target_strategy
        self.cost_strategy = cost_strategy

        self.c = hparams.c
        self.kappa = hparams.kappa
        self.num_steps = hparams.num_steps
        self.lr_cls = hparams.lr_cls

        # Set the models to the evaluation mode
        self.shadow_model.eval()

    def __call__(self, feats, labels):
        """The forward method to compute the adversarial features.

        Args:
            - feats: :math:`(B, dim_emb)`: The features.
            - labels: :math:`(B)`: The speaker IDs.

        Return:
            best_adv_z :math:`(B, latent_dim)`: The adversarial features.
            z :math:`(B, latent_dim)`: The normal features.
        """

        z = self.shadow_model.encoder(feats)

        z = z.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        target_labels = self.target_strategy.get_target_label(z=z, target=labels)

        w = z.detach()

        w.requires_grad = True

        best_adv_z = z.clone().detach()
        best_L2 = 1e10 * torch.ones((len(z))).to(self.device)
        prev_cost = 1e10
        dim = len(feats.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = Adam([w], lr=self.lr_cls)

        for step in range(self.num_steps):
            # Get adversarial features
            adv_z = w

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_z), Flatten(z)).sum(dim=1)
            L2_loss = current_L2.sum()

            feat_out = self.shadow_model.classifier(adv_z)

            f_loss = self.cost_strategy.get_cost(
                feat_out, target_labels.to(self.device), self.kappa
            )

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
            best_adv_z = mask * adv_z.detach() + (1 - mask) * best_adv_z

            # Early stop when loss does not converge.
            if (
                step % max(self.num_steps // 10, 1) == 0
            ):  # max(.,1) To prevent MODULO BY ZERO error in the next step.
                if cost.item() > prev_cost:
                    return best_adv_z, z
                prev_cost = cost.item()

        return best_adv_z, z, feat_out
