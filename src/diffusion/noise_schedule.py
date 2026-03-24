"""Noise schedules and transition matrices for discrete graph diffusion.

Implements the forward process components of DiGress (Vignac et al., 2023):
predefined noise schedules that map continuous time t ∈ [0, 1] to noise levels,
and transition matrices Qt / Qt_bar that define how graph features are corrupted
at each diffusion step.

Three transition strategies are provided:
- DiscreteUniformTransition: uniform noise over all classes
- MarginalUniformTransition: noise proportional to the data marginal distribution
- AbsorbingStateTransition: all mass absorbs into a designated mask/absorbing token

``PredefinedNoiseScheduleDiscrete`` provides the per-step β and cumulative ᾱ
lookup tables used by ``DiscreteDenoisingDiffusion``.

Reference:
    Vignac et al., "DiGress: Discrete Denoising Diffusion for Graph Generation",
    ICLR 2023. https://openreview.net/forum?id=UaAD-Nu86WX
"""

import torch
from src import utils
from src.diffusion import diffusion_utils


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """Per-step β and cumulative ᾱ lookup tables for discrete graph diffusion.

    Precomputes and stores:
    - ``betas``: the per-step noise level β_t for t = 1 … T
    - ``alphas_bar``: the cumulative product ᾱ_t = ∏_{s=1}^{t} (1 − β_s)

    Both tables are indexed by integer timestep and are accessed via
    ``forward`` (returns β_t) and ``get_alpha_bar`` (returns ᾱ_t).
    ``DiscreteDenoisingDiffusion`` uses this schedule to obtain the noise
    parameters needed to construct the transition matrices Qt and Qt_bar.
    """

    def __init__(self, noise_schedule: str, timesteps: int) -> None:
        """Precompute β and ᾱ lookup tables for the given schedule.

        Args:
            noise_schedule: Schedule type. Supports ``'cosine'`` and
                ``'custom'``.
            timesteps: Total number of diffusion steps T. Both lookup
                tables have length T.

        Raises:
            NotImplementedError: If ``noise_schedule`` is not recognised.
        """
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the per-step noise level β_t for the requested timestep(s).

        Exactly one of ``t_normalized`` or ``t_int`` must be provided.

        Args:
            t_normalized: Continuous time in [0, 1]. Converted to an integer
                timestep by rounding to the nearest multiple of 1/T.
            t_int: Integer timestep(s) in {0, …, T} as a float or long tensor.

        Returns:
            β_t values with the same leading shape as the provided time tensor.
        """
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the cumulative noise retention ᾱ_t = ∏_{s=1}^{t} (1 − β_s).

        ᾱ_t measures how much signal survives after t diffusion steps; it
        approaches 0 as t → T, meaning the graph is fully noised. Used to
        construct the marginal transition matrix Qt_bar.

        Exactly one of ``t_normalized`` or ``t_int`` must be provided.

        Args:
            t_normalized: Continuous time in [0, 1].
            t_int: Integer timestep(s) in {0, …, T} as a float or long tensor.

        Returns:
            ᾱ_t values with the same leading shape as the provided time tensor,
            placed on the same device as ``t_int``.
        """
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class DiscreteUniformTransition:
    """Transition matrices that add uniform noise over all discrete classes.

    Implements the uniform absorbing transition from DiGress:

        Qt = (1 − β_t) · I + β_t / K · 𝟏𝟏ᵀ

    where K is the number of classes and 𝟏𝟏ᵀ / K is a matrix whose rows
    are all equal to the uniform distribution. Each row of Qt is a valid
    probability distribution, so multiplying a one-hot vector by Qt mixes
    it with the uniform distribution with weight β_t.

    The marginal transition Qt_bar from step 0 to step t becomes:

        Qt_bar = ᾱ_t · I + (1 − ᾱ_t) / K · 𝟏𝟏ᵀ

    Separate matrices are maintained for node features (X), edge features (E),
    and global conditioning (y).
    """

    def __init__(self, x_classes: int, e_classes: int, y_classes: int) -> None:
        """Precompute the uniform stationary matrices u_x, u_e, u_y.

        Args:
            x_classes: Number of node feature classes (K_X).
            e_classes: Number of edge feature classes (K_E).
            y_classes: Number of global label classes (K_y).
        """
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device) -> utils.PlaceHolder:
        """Return the one-step transition matrices Qt for a batch of timesteps.

        Constructs Qt = (1 − β_t) · I + β_t · u for each of X, E, and y,
        where u is the uniform stationary matrix and I is the identity.
        Each row of Qt sums to 1 and is a mixture of staying (weight 1 − β_t)
        and jumping to a uniform class (weight β_t).

        Args:
            beta_t: Per-sample noise levels of shape (bs,), with values in
                [0, 1]. Corresponds to β_t in the DiGress forward process.
            device: Target device for all returned matrices.

        Returns:
            A PlaceHolder with:
            - ``X``: node transition matrices of shape (bs, K_X, K_X)
            - ``E``: edge transition matrices of shape (bs, K_E, K_E)
            - ``y``: label transition matrices of shape (bs, K_y, K_y)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device: torch.device) -> utils.PlaceHolder:
        """Return the marginal transition matrices Qt_bar from step 0 to step t.

        Constructs Qt_bar = ᾱ_t · I + (1 − ᾱ_t) · u for each of X, E, and y.
        As ᾱ_t → 0, Qt_bar approaches the uniform stationary distribution,
        meaning the original data is completely forgotten. As ᾱ_t → 1 (t → 0),
        Qt_bar approaches the identity.

        Args:
            alpha_bar_t: Cumulative signal retention ᾱ_t of shape (bs,), with
                values in [0, 1]. Computed as ∏_{s=1}^{t} (1 − β_s).
            device: Target device for all returned matrices.

        Returns:
            A PlaceHolder with:
            - ``X``: marginal node matrices of shape (bs, K_X, K_X)
            - ``E``: marginal edge matrices of shape (bs, K_E, K_E)
            - ``y``: marginal label matrices of shape (bs, K_y, K_y)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalUniformTransition:
    """Transition matrices that add noise proportional to the data marginals.

    A data-aware alternative to DiscreteUniformTransition. Instead of mixing
    with the uniform distribution, the stationary distribution used for noise
    is the empirical marginal of each feature type in the training data:

        Qt = (1 − β_t) · I + β_t · m

    where m is a matrix whose rows are all equal to the data marginal
    distribution. This encourages the noised samples to resemble the marginal
    statistics of the training set, which can improve sample quality.

    The marginal transition Qt_bar from step 0 to step t is:

        Qt_bar = ᾱ_t · I + (1 − ᾱ_t) · m

    Global label transitions (y) still use a uniform stationary distribution
    because no label marginals are provided.
    """

    def __init__(
        self,
        x_marginals: torch.Tensor,
        e_marginals: torch.Tensor,
        y_classes: int,
    ) -> None:
        """Precompute the marginal stationary matrices u_x, u_e, u_y.

        Args:
            x_marginals: Empirical node-class distribution of shape (K_X,),
                summing to 1. Each row of u_x will equal this distribution.
            e_marginals: Empirical edge-class distribution of shape (K_E,),
                summing to 1. Each row of u_e will equal this distribution.
            y_classes: Number of global label classes. Uses uniform noise.
        """
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device) -> utils.PlaceHolder:
        """Return the one-step marginal-stationary transition matrices Qt.

        Constructs Qt = (1 − β_t) · I + β_t · u for each of X, E, and y,
        where u for X and E uses the empirical data marginals as rows.

        Args:
            beta_t: Per-sample noise levels of shape (bs,), with values in
                [0, 1].
            device: Target device for all returned matrices.

        Returns:
            A PlaceHolder with:
            - ``X``: node transition matrices of shape (bs, K_X, K_X)
            - ``E``: edge transition matrices of shape (bs, K_E, K_E)
            - ``y``: label transition matrices of shape (bs, K_y, K_y)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device: torch.device) -> utils.PlaceHolder:
        """Return the marginal transition matrices Qt_bar from step 0 to step t.

        Constructs Qt_bar = ᾱ_t · I + (1 − ᾱ_t) · u for each of X, E, and y,
        where u for X and E uses the empirical data marginals as rows.

        Args:
            alpha_bar_t: Cumulative signal retention ᾱ_t of shape (bs,),
                with values in [0, 1].
            device: Target device for all returned matrices.

        Returns:
            A PlaceHolder with:
            - ``X``: marginal node matrices of shape (bs, K_X, K_X)
            - ``E``: marginal edge matrices of shape (bs, K_E, K_E)
            - ``y``: marginal label matrices of shape (bs, K_y, K_y)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class AbsorbingStateTransition:
    """Transition matrices that route all probability mass to an absorbing state.

    Implements the absorbing (mask) diffusion from Austin et al. (2021).
    All transitions converge to a designated absorbing token ``abs_state``
    (typically a [MASK] token index):

        Qt = (1 − β_t) · I + β_t · u_abs

    where u_abs is a matrix in which every row is the one-hot vector for
    ``abs_state``. Once a feature is absorbed it stays absorbed, so the
    absorbing state acts as an irreversible mask rather than random noise.

    Unlike DiscreteUniformTransition, this class does *not* accept a device
    argument in get_Qt / get_Qt_bar — callers must move tensors to the
    correct device beforehand.
    """

    def __init__(
        self,
        abs_state: int,
        x_classes: int,
        e_classes: int,
        y_classes: int,
    ) -> None:
        """Precompute the absorbing-state matrices u_x, u_e, u_y.

        Args:
            abs_state: Index of the absorbing (mask) token. Must be a valid
                class index for all three feature types.
            x_classes: Number of node feature classes.
            e_classes: Number of edge feature classes.
            y_classes: Number of global label classes.
        """
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.u_x = torch.zeros(1, self.X_classes, self.X_classes)
        self.u_x[:, :, abs_state] = 1

        self.u_e = torch.zeros(1, self.E_classes, self.E_classes)
        self.u_e[:, :, abs_state] = 1

        self.u_y = torch.zeros(1, self.y_classes, self.y_classes)
        self.u_y[:, :, abs_state] = 1

    def get_Qt(
        self, beta_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the one-step absorbing transition matrices Qt.

        Constructs Qt = (1 − β_t) · I + β_t · u_abs for each of X, E, and y.
        Each row of Qt places weight β_t on the absorbing state and weight
        1 − β_t on retaining the current class.

        Args:
            beta_t: Per-sample noise levels of shape (bs,), with values in
                [0, 1]. Must already be on the correct device.

        Returns:
            A 3-tuple (q_x, q_e, q_y) of transition matrices with shapes
            (bs, K_X, K_X), (bs, K_E, K_E), and (bs, K_y, K_y) respectively.
        """
        beta_t = beta_t.unsqueeze(1)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes).unsqueeze(0)
        return q_x, q_e, q_y

    def get_Qt_bar(
        self, alpha_bar_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the marginal absorbing transition matrices Qt_bar from step 0 to t.

        Constructs Qt_bar = ᾱ_t · I + (1 − ᾱ_t) · u_abs for each feature type.
        As ᾱ_t → 0, each class deterministically maps to the absorbing state.

        Args:
            alpha_bar_t: Cumulative signal retention ᾱ_t of shape (bs,),
                with values in [0, 1]. Must already be on the correct device.

        Returns:
            A 3-tuple (q_x, q_e, q_y) of marginal transition matrices with
            shapes (bs, K_X, K_X), (bs, K_E, K_E), and (bs, K_y, K_y).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        q_x = alpha_bar_t * torch.eye(self.X_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return q_x, q_e, q_y
