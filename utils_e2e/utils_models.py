import torch
import torch.nn as nn
import abc
import torch.nn.functional as F


from opacus.layers import DPLSTM


class DvectorInterface(nn.Module, metaclass=abc.ABCMeta):
    """d-vector interface."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            and hasattr(subclass, "seg_len")
            or NotImplemented
        )

    @abc.abstractmethod
    def forward(self, inputs):
        """Forward a batch through network.

        Args:
            inputs: (batch, seg_len, mel_dim)

        Returns:
            embeds: (batch, emb_dim)
        """
        raise NotImplementedError

    # @torch.jit.export
    def embed_utterance(self, utterance):
        """Embed an utterance by segmentation and averaging

        Args:
            utterance: (uttr_len, mel_dim) or (1, uttr_len, mel_dim)


        Returns:
            embed: (emb_dim)
        """
        assert utterance.ndim == 2 or (utterance.ndim == 3 and utterance.size(0) == 1)

        if utterance.ndim == 3:
            utterance = utterance.squeeze(0)

        if utterance.size(1) <= self.seg_len:
            embed = self.forward(utterance.unsqueeze(0)).squeeze(0)
        else:
            segments = utterance.unfold(0, self.seg_len, self.seg_len // 2)
            embeds = self.forward(segments)
            embed = embeds.mean(dim=0)
            embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True))

        return embed

    # @torch.jit.export
    def embed_utterances(self, utterances):
        """Embed utterances by averaging the embeddings of utterances

        Args:
            utterances: [(uttr_len, mel_dim), ...]

        Returns:
            embed: (emb_dim)
        """
        embeds = torch.stack([self.embed_utterance(uttr) for uttr in utterances])
        embed = embeds.mean(dim=0)
        return embed.div(embed.norm(p=2, dim=-1, keepdim=True))


class CustomGroupNorm(nn.Module):
    r"""
    ## Group Normalization Layer
    """

    def __init__(
        self,
        groups: int,
        channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        * `groups` is the number of groups the features are divided into
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        assert (
            channels % groups == 0
        ), "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels

        self.eps = eps
        self.affine = affine
        # Create parameters for $\gamma$ and $\beta$ for scale and shift
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, xp: torch.Tensor = None):
        """
        `x` is a tensor of shape `[batch_size, channels, *]` for public/private data.
        `xp` is a tensor of shape `[batch_size, channels, *]` for public data.
        `*` denotes any number of (possibly 0) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        # Keep the original shape
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]
        # Sanity check to make sure the number of features is the same
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, groups, n]`
        x = x.view(batch_size, self.groups, -1)

        if xp != None:
            x_statistics = xp.view(batch_size, self.groups, -1)
        else:
            x_statistics = x.view(batch_size, self.groups, -1)

        # Calculate the mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x_{(i_N, i_G)}]$
        # mean = x.mean(dim=[-1], keepdim=True)
        mean = x_statistics.mean(dim=[-1], keepdim=True)
        # Calculate the squared mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x^2_{(i_N, i_G)}]$
        # mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        mean_x2 = (x_statistics**2).mean(dim=[-1], keepdim=True)
        # Variance for each sample and feature group
        # $Var[x_{(i_N, i_G)}] = \mathbb{E}[x^2_{(i_N, i_G)}] - \mathbb{E}[x_{(i_N, i_G)}]^2$
        # var = mean_x2 - mean ** 2
        var = mean_x2 - mean**2

        # Normalize
        # $$\hat{x}_{(i_N, i_G)} =
        # \frac{x_{(i_N, i_G)} - \mathbb{E}[x_{(i_N, i_G)}]}{\sqrt{Var[x_{(i_N, i_G)}] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if xp != None:
            xp_norm = (x_statistics - mean) / torch.sqrt(var + self.eps)
            # Scale and shift channel-wise
            # $$y_{i_C} =\gamma_{i_C} \hat{x}_{i_C} + \beta_{i_C}$$
            if self.affine:
                x_norm = x_norm.view(batch_size, self.channels, -1)
                x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
            # Reshape to original and return
            return x_norm.view(x_shape), xp_norm.view(x_shape)

        else:
            xp_norm = x_norm

            # Reshape to original and return
            return x_norm.view(x_shape)


class DvecAttentivePooledClsE2E(DvectorInterface):
    """End-to-end LSTM-based d-vector with attentive pooling and classification."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        self.gnorm = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.linear_0 = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.gnorm_0 = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)

        self.linear_1 = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.linear_2 = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward a batch through network."""
        out, _ = self.lstm(x)

        embeds = torch.tanh(self.embedding(out))

        embeds = self.gnorm(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

        feat_0 = self.relu(self.linear_0(embeds))

        feat_0 = self.gnorm_0(feat_0)
        feat_1 = self.relu(self.linear_1(feat_0))

        feat_out = self.linear_2(feat_1)
        out = self.softmax(feat_out)

        return out, feat_out, feat_1, embeds


class DvecAttentivePooledClsTransformerE2E(DvectorInterface):
    """End-to-end Transformer-based d-vector with attentive pooling and classification."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.prenet = nn.Linear(40, 80)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=80,
            nhead=2,
            dim_feedforward=256,
        )

        self.embedding = nn.Linear(80, 80)

        self.gnorm = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.linear = nn.Linear(80, 1)

        self.linear_0 = nn.Linear(80, 40)
        self.gnorm_0 = nn.GroupNorm(self.args.gp_norm_cls, 40)

        self.linear_1 = nn.Linear(40, 40)
        self.linear_2 = nn.Linear(40, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward a batch through network."""

        # out_prenet: (batch_size, seg_length, d_model)
        out_prenet = self.prenet(x)

        # out_prenet: (seg_length, batch_size, d_model)
        out_prenet = out_prenet.permute(1, 0, 2)

        # out_enc: (seg_length, batch_size, d_model)
        out_enc = self.encoder(out_prenet)

        # out_enc: (batch_size, seg_length, d_model)
        out_enc = out_enc.transpose(0, 1)

        # Mean pooling
        embeds = out_enc.mean(dim=1)
        # embeds = out_enc[:, -1, :].view(out_enc.shape[0], out_enc.shape[2])

        # # Attentive pooling
        # embeds = torch.tanh(self.embedding(out_enc))
        # embeds = self.gnorm(embeds)
        # attn_weights = F.softmax(self.linear(embeds), dim=1)
        # embeds = torch.sum(embeds * attn_weights, dim=1)
        # embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

        feat_0 = self.relu(self.linear_0(embeds))
        feat_0 = self.gnorm_0(feat_0)

        feat_1 = self.relu(self.linear_1(feat_0))

        feat_out = self.linear_2(feat_1)
        out = self.softmax(feat_out)

        return out, feat_out, feat_1, embeds


class DvecAttentivePooledClsE2E_v2(DvectorInterface):
    """End-to-end LSTM-based d-vector with attentive pooling and classification."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        self.gnorm = CustomGroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.linear_0 = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        self.gnorm_0 = CustomGroupNorm(8, self.args.latent_dim)

        self.linear_1 = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.linear_2 = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x, xp):
        """Forward a batch through network."""
        out, _ = self.lstm(x)
        outp, _ = self.lstm(xp).detach()

        embeds = torch.tanh(self.embedding(out))
        embedsp = torch.tanh(self.embedding(outp)).detach()

        embeds, embedsp = self.gnorm(embeds, embedsp)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

        attn_weightsp = F.softmax(self.linear(embedsp), dim=1).detach()
        embedsp = torch.sum(embedsp * attn_weightsp, dim=1)
        embedsp.div(embedsp.norm(p=2, dim=-1, keepdim=True))

        embeds = embeds / torch.norm(embeds, dim=1).view(embeds.size()[0], 1)
        embedsp = embedsp / torch.norm(embedsp, dim=1).view(embedsp.size()[0], 1)

        feat_0 = self.relu(self.linear_0(embeds))
        feat_0_p = self.relu(self.linear_0(embedsp)).detach()

        feat_0_n, _ = self.gnorm_0(feat_0, feat_0_p)
        feat_1 = self.relu(self.linear_1(feat_0_n))

        feat_out = self.linear_2(feat_1)
        out = self.softmax(feat_out)

        return out, feat_out, feat_1, embeds


class DvecAttentivePooledClsE2E_v3(DvectorInterface):
    """End-to-end LSTM-based d-vector with attentive pooling and classification."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.lstm = DPLSTM(
            self.args.feature_dim,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        self.gnorm = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        # self.layer_norm = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])
        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.linear_0 = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        # self.gnorm_0 = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)

        self.linear_1 = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        # self.gnorm_1 = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)

        self.linear_2 = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward a batch through network."""
        out, _ = self.lstm(x)

        embeds = torch.tanh(self.embedding(out))

        # embeds = self.layer_norm(embeds)
        embeds = self.gnorm(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)

        embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

        feat_0 = self.relu(self.linear_0(embeds))
        # feat_0 = self.gnorm_0(feat_0)

        feat_1 = self.relu(self.linear_1(feat_0))
        # feat_1 = self.gnorm_1(feat_1)

        feat_out = self.linear_2(feat_1)
        out = self.softmax(feat_out)

        return out, feat_out, feat_1, embeds


class ShadowDvecAttentivePooledClsE2E(DvectorInterface):
    """End-to-end LSTM-based d-vector with attentive pooling and classification."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.lstm = nn.LSTM(
            self.args.feature_dim,
            self.args.dim_cell // 2,
            self.args.num_layers,
            batch_first=True,
        )

        self.embedding = nn.Linear(self.args.dim_cell // 2, self.args.dim_emb)

        self.gnorm = nn.GroupNorm(self.args.gp_norm_dvector, self.args.seg_len)
        # self.layer_norm = nn.LayerNorm([self.args.seg_len, self.args.dim_emb])
        self.linear = nn.Linear(self.args.dim_emb, 1)

        self.linear_0 = nn.Linear(self.args.dim_emb, self.args.latent_dim)
        # self.gnorm_0 = nn.GroupNorm(self.args.gp_norm_cls, self.args.latent_dim)

        self.linear_1 = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        self.linear_2 = nn.Linear(self.args.latent_dim, self.args.n_speakers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward a batch through network."""
        out, _ = self.lstm(x)

        embeds = torch.tanh(self.embedding(out))

        # embeds = self.layer_norm(embeds)
        embeds = self.gnorm(embeds)

        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)

        embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

        feat_0 = self.relu(self.linear_0(embeds))
        # feat_0 = self.gnorm_0(feat_0)

        feat_1 = self.relu(self.linear_1(feat_0))

        feat_out = self.linear_2(feat_1)
        out = self.softmax(feat_out)

        return out, feat_out, feat_1, embeds


class MembInfAttk(nn.Module):
    def __init__(self, args):
        super(MembInfAttk, self).__init__()
        self.args = args

        self.input = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.hidden = nn.Linear(self.args.latent_dim, self.args.e_dim)

        self.out = nn.Linear(self.args.e_dim, 2)

        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward a batch through network."""

        feat = self.relu(self.input(z))

        feat = self.relu(self.hidden(feat))

        feat_out = self.out(feat)

        return feat_out, feat


class MembInfAttk_v2(nn.Module):
    def __init__(self, args):
        super(MembInfAttk_v2, self).__init__()
        self.args = args

        self.input = nn.Linear(self.args.dim_emb, self.args.latent_dim)

        self.hidden = nn.Linear(self.args.latent_dim, self.args.e_dim)

        self.out = nn.Linear(self.args.e_dim, 2)

        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward a batch through network."""

        feat = self.relu(self.input(z))

        feat = self.relu(self.hidden(feat))

        feat_out = self.out(feat)

        return feat_out, feat


class MembInfAttk_v3(nn.Module):
    def __init__(self, args):
        super(MembInfAttk_v3, self).__init__()
        self.args = args

        self.input = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        self.hidden = nn.Linear(self.args.latent_dim, self.args.e_dim)

        self.out = nn.Linear(self.args.e_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, z):
        """Forward a batch through network."""

        feat = self.relu(self.input(z))

        feat = self.relu(self.hidden(feat))

        feat_out = self.out(feat)

        return feat_out, feat
