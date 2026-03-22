"""Discrete denoising diffusion probabilistic model for graph generation.

Implements the DiGress model (Vignac et al., 2022) — a discrete DDPM that
operates on categorical node and edge features of molecular graphs. Noise is
applied via Markov transition matrices Q_t rather than Gaussian perturbations,
and the denoising network is a graph transformer trained to predict x_0 from
a noisy graph z_t.

This module defines :class:`DiscreteDenoisingDiffusion`, the central
PyTorch Lightning module used for training, validation, testing, and
unconditional graph sampling.
"""

import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.noise_schedule import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
)
from metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from metrics.train_metrics import TrainLossDiscrete
from models.transformer_model import GraphTransformer
from src import utils
from src.diffusion import diffusion_utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    """Discrete denoising diffusion model for graph generation (DiGress).

    Implements the full training, evaluation, and sampling pipeline from
    Vignac et al. (2022) "DiGress: Discrete Denoising diffusion for graph
    generation". A graph transformer denoises noisy graphs z_t back to
    clean graphs x_0 by learning the reverse of a discrete Markov process
    defined by transition matrices Q_t.

    The ELBO loss decomposes into:
      - KL prior: KL(q(z_T | x) || p(z_T)) — should be near zero
      - Diffusion loss L_t: sum of KL terms at intermediate timesteps
      - Reconstruction loss L_0: -log p(x | z_0)
      - Node count log-probability: -log p(N)

    Attributes:
        T: Total number of diffusion timesteps.
        Xdim: Input node feature dimension.
        Edim: Input edge feature dimension.
        ydim: Input global feature (conditioning) dimension.
        Xdim_output: Output/predicted node feature dimension.
        Edim_output: Output/predicted edge feature dimension.
        ydim_output: Output/predicted global feature dimension.
        model: The underlying GraphTransformer denoising network.
        noise_schedule: Maps timestep t to noise level β_t and ᾱ_t.
        transition_model: Builds transition matrices Q_t, Q̄_t from noise levels.
        limit_dist: Stationary distribution p(z_T) that the noisy graph converges to.
        best_val_nll: Tracks the best validation NLL seen so far.
    """

    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ) -> None:
        """Initialize the model, noise schedule, transition model, and metrics.

        Args:
            cfg: Hydra config object. Reads ``cfg.model``, ``cfg.train``, and
                ``cfg.general`` sub-configs.
            dataset_infos: Dataset metadata object exposing ``input_dims``,
                ``output_dims``, ``nodes_dist``, ``node_types``, and
                ``edge_types``.
            train_metrics: Metric aggregator called each training step to track
                per-class accuracy.
            sampling_metrics: Metrics evaluated on sampled molecules (e.g.
                validity, uniqueness) — called at the end of validation and test.
            visualization_tools: Optional tool for saving chain and molecule
                visualizations to disk. Pass ``None`` to skip visualization.
            extra_features: Callable that computes structural graph features
                (e.g. cycle counts, eigenvalues) appended to the network input.
            domain_features: Callable that computes domain-specific features
                (e.g. molecular fingerprints) appended to the network input.
        """
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps
        )

        if cfg.model.transition == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == "marginal":
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges"
            )
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=torch.ones(self.ydim_output) / self.ydim_output,
            )

        self.save_hyperparameters(ignore=["train_metrics", "sampling_metrics"])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i) -> dict[str, torch.Tensor] | None:
        """Run one training step: add noise, denoise, compute cross-entropy loss.

        Args:
            data: PyG ``Data`` batch with ``x``, ``edge_index``, ``edge_attr``,
                ``y``, and ``batch`` attributes.
            i: Global step index used to decide whether to log this step.

        Returns:
            Dict with key ``"loss"`` containing the scalar training loss, or
            ``None`` if the batch had no edges and was skipped.
        """
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Create the AdamW optimizer with AMSGrad and weight decay from config."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        """Cache the number of training iterations and log input dimensions."""
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        """Reset per-epoch loss and metric accumulators and start the epoch timer."""
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Log epoch-level cross-entropy losses, atom/bond accuracy, and GPU memory."""
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE']:.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE']:.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE']:.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        else:
            print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        """Reset all validation metric accumulators before the epoch begins."""
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i) -> dict[str, torch.Tensor]:
        """Compute the ELBO estimate for one validation batch.

        Args:
            data: PyG ``Data`` batch.
            i: Batch index within the epoch (unused, kept for Lightning API).

        Returns:
            Dict with key ``"loss"`` containing the batch-mean NLL estimate.
        """
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(
            pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False
        )
        return {"loss": nll}

    def on_validation_epoch_end(self) -> None:
        """Log aggregated val NLL and KL metrics; optionally run molecule sampling."""
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute(),
        ]
        self.log_dict(
            {
                "val/X_kl": metrics[1],
                "val/E_kl": metrics[2],
                "val/X_logp": metrics[3],
                "val/E_logp": metrics[4],
            }
        )

        self.print(
            f"Epoch {self.current_epoch}: Val NLL {metrics[0]:.2f} -- Val Atom type KL {metrics[1]:.2f} -- ",
            f"Val Edge type KL: {metrics[2]:.2f}",
        )

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(
                    self.sample_batch(
                        batch_id=ident,
                        batch_size=to_generate,
                        num_nodes=None,
                        save_final=to_save,
                        keep_chain=chains_save,
                        number_chain_steps=self.number_chain_steps,
                    )
                )
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(
                samples,
                self.name,
                self.current_epoch,
                val_counter=-1,
                test=False,
                local_rank=self.local_rank,
            )
            self.print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        """Reset all test metric accumulators before the epoch begins."""
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()

    def test_step(self, data, i) -> dict[str, torch.Tensor]:
        """Compute the ELBO estimate for one test batch.

        Args:
            data: PyG ``Data`` batch.
            i: Batch index within the epoch (unused, kept for Lightning API).

        Returns:
            Dict with key ``"loss"`` containing the batch-mean NLL estimate.
        """
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(
            pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True
        )
        return {"loss": nll}

    def on_test_epoch_end(self) -> None:
        """Log aggregated test NLL and KL metrics, save samples, and run sampling metrics."""
        metrics = [
            self.test_nll.compute(),
            self.test_X_kl.compute(),
            self.test_E_kl.compute(),
            self.test_X_logp.compute(),
            self.test_E_logp.compute(),
        ]
        self.log_dict(
            {
                "test/X_kl": metrics[1],
                "test/E_kl": metrics[2],
                "test/X_logp": metrics[3],
                "test/E_logp": metrics[4],
            }
        )

        self.print(
            f"Epoch {self.current_epoch}: Test NLL {metrics[0]:.2f} -- Test Atom type KL {metrics[1]:.2f} -- ",
            f"Test Edge type KL: {metrics[2]:.2f}",
        )

        test_nll = metrics[0]
        self.log("test/epoch_NLL", test_nll)

        self.print(f"Test loss: {test_nll:.4f}")

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        sample_id = 0
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{self.cfg.general.final_model_samples_to_generate}",
                end="",
                flush=True,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(
                self.sample_batch(
                    sample_id,
                    to_generate,
                    num_nodes=None,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                )
            )
            sample_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = "generated_samples1.txt"
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f"generated_samples{i}.txt"
            else:
                break
        with open(filename, "w") as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(
            samples,
            self.name,
            self.current_epoch,
            self.val_counter,
            test=True,
            local_rank=self.local_rank,
        )
        self.print("Done testing.")

    def kl_prior(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(q(z_T | x) || p(z_T)) for the prior loss term.

        This term measures how close the fully-noised distribution q(z_T | x)
        is to the stationary/prior distribution p(z_T) (uniform or marginal).
        In practice it is nearly zero when the noise schedule is well-calibrated,
        but tracking it helps detect bugs in the schedule or transition matrices.

        Corresponds to the KL prior term in the DiGress ELBO (Eq. 5).

        Args:
            X: Clean node feature one-hots of shape ``(bs, n, dx)``.
            E: Clean edge feature one-hots of shape ``(bs, n, n, de)``.
            node_mask: Boolean mask of shape ``(bs, n)`` marking valid nodes.

        Returns:
            Per-sample KL divergence of shape ``(bs,)``.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = (
            self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        )

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
        )

        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction="none"
        )

        return diffusion_utils.sum_except_batch(
            kl_distance_X
        ) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        pred,
        noisy_data: dict[str, torch.Tensor],
        node_mask: torch.Tensor,
        test: bool,
    ) -> torch.Tensor:
        """Compute the diffusion loss L_t — the KL between posterior distributions.

        For each sample in the batch, computes the KL divergence between the
        true denoising posterior q(z_s | z_t, x_0) and the predicted posterior
        p_θ(z_s | z_t), where s = t − 1. Scaled by T to match the ELBO
        (Vignac et al., 2022, Eq. 5).

        The posterior q(z_s | z_t, x_0) is computed analytically via Bayes'
        rule using the transition matrices Qt, Qsb, and Qtb. The model-predicted
        posterior uses softmax(pred) in place of x_0.

        Also accumulates per-class KL metrics into the appropriate
        ``val_X_kl``/``test_X_kl`` and ``val_E_kl``/``test_E_kl`` accumulators.

        Args:
            X: Clean node feature one-hots of shape ``(bs, n, dx)``.
            E: Clean edge feature one-hots of shape ``(bs, n, n, de)``.
            y: Clean global features of shape ``(bs, dy)``.
            pred: PlaceHolder with logit tensors ``pred.X`` ``(bs, n, dx)``,
                ``pred.E`` ``(bs, n, n, de)``, ``pred.y`` ``(bs, dy)``.
            noisy_data: Dict produced by :meth:`apply_noise` containing
                ``"X_t"``, ``"E_t"``, ``"y_t"``, ``"alpha_t_bar"``,
                ``"alpha_s_bar"``, and ``"beta_t"``.
            node_mask: Boolean mask of shape ``(bs, n)``.
            test: If ``True``, accumulates into test metrics; otherwise val.

        Returns:
            Per-sample diffusion loss of shape ``(bs,)``, scaled by T.
        """
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(
            X=X,
            E=E,
            y=y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(
            X=pred_probs_X,
            E=pred_probs_E,
            y=pred_probs_y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = (
            diffusion_utils.mask_distributions(
                true_X=prob_true.X,
                true_E=prob_true.E,
                pred_X=prob_pred.X,
                pred_E=prob_pred.E,
                node_mask=node_mask,
            )
        )
        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true.X, torch.log(prob_pred.X)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true.E, torch.log(prob_pred.E)
        )
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(
        self,
        t: torch.Tensor,
        X: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        """Compute the reconstruction log-probability log p_θ(x | z_0).

        Evaluates the L_0 term of the ELBO by: (1) sampling a slightly noisy
        z_0 from the one-step transition distribution q(z_0 | x_0) using β_0;
        (2) running the denoising network on z_0 at t=0; (3) returning the
        predicted probability distributions over x_0.

        The caller computes the actual log-likelihood as X * log(probX0),
        summed over nodes and classes.

        Args:
            t: Normalized timestep tensor of shape ``(bs, 1)``. Used only to
                derive β_0 via the noise schedule at t=0.
            X: Clean node feature one-hots of shape ``(bs, n, dx)``.
            E: Clean edge feature one-hots of shape ``(bs, n, n, de)``.
            node_mask: Boolean mask of shape ``(bs, n)``.

        Returns:
            PlaceHolder with fields ``X`` ``(bs, n, dx_out)``, ``E``
            ``(bs, n, n, de_out)``, and ``y`` containing the predicted
            probability distributions (after softmax) for each type.
        """
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(
            probX=probX0, probE=probE0, node_mask=node_mask
        )

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1).type_as(y0),
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(
            self.Edim_output
        ).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sample a random timestep and corrupt the graph via the forward process.

        Implements q(z_t | x_0) by: (1) sampling a uniform random timestep t;
        (2) computing the cumulative transition matrix Q̄_t = Q_1 · Q_2 · … · Q_t;
        (3) computing per-node and per-edge transition probabilities X @ Q̄_t
        and E @ Q̄_t; (4) sampling discrete noisy features z_t from those
        distributions.

        During training, t is sampled from {0, …, T}; during evaluation, t
        is sampled from {1, …, T} because L_0 is computed separately in
        :meth:`reconstruction_logp`.

        Args:
            X: Clean node feature one-hots of shape ``(bs, n, dx)``.
            E: Clean edge feature one-hots of shape ``(bs, n, n, de)``.
            y: Global/conditioning features of shape ``(bs, dy)``.
            node_mask: Boolean mask of shape ``(bs, n)``.

        Returns:
            Dict with keys: ``"t_int"`` (integer timestep), ``"t"`` (normalized
            float timestep), ``"beta_t"``, ``"alpha_s_bar"``, ``"alpha_t_bar"``
            (noise schedule values), ``"X_t"``, ``"E_t"``, ``"y_t"`` (noisy
            graph features), and ``"node_mask"``.
        """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device
        ).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_t_bar, device=self.device
        )  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }
        return noisy_data

    def compute_val_loss(
        self,
        pred,
        noisy_data: dict[str, torch.Tensor],
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        """Compute the ELBO estimate (negative log-likelihood lower bound).

        Assembles four terms that together estimate -log p(x):

        1. ``-log p(N)``: negative log-probability of the node count.
        2. ``KL prior``: KL(q(z_T | x) || p(z_T)), near zero for a good schedule.
        3. ``L_t``: sum of KL divergences over intermediate denoising steps.
        4. ``L_0``: reconstruction term -log p(x | z_0).

        NLL = -log_pN + kl_prior + L_t - L_0

        All four terms and the batch NLL are logged via ``self.log_dict``.

        Args:
            pred: PlaceHolder with logit tensors from the denoising network.
            noisy_data: Dict produced by :meth:`apply_noise`.
            X: Clean node feature one-hots of shape ``(bs, n, dx)``.
            E: Clean edge feature one-hots of shape ``(bs, n, n, de)``.
            y: Global features of shape ``(bs, dy)``.
            node_mask: Boolean mask of shape ``(bs, n)``.
            test: If ``True``, accumulates into test NLL metric; otherwise val.

        Returns:
            Scalar batch-mean NLL estimate.
        """
        t = noisy_data["t"]

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(
            E * prob0.E.log()
        )

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        self.log_dict(
            {
                "kl prior": kl_prior.mean(),
                "Estimator loss terms": loss_all_t.mean(),
                "log_pn": log_pN.mean(),
                "loss_term_0": loss_term_0,
                "batch_test_nll" if test else "val_nll": nll,
            }
        )
        return nll

    def forward(
        self,
        noisy_data: dict[str, torch.Tensor],
        extra_data,
        node_mask: torch.Tensor,
    ):
        """Run the denoising network on a noisy graph.

        Concatenates the noisy graph features with extra structural and domain
        features before passing them to the GraphTransformer.

        Args:
            noisy_data: Dict containing ``"X_t"`` ``(bs, n, dx)``, ``"E_t"``
                ``(bs, n, n, de)``, and ``"y_t"`` ``(bs, dy)``.
            extra_data: PlaceHolder with fields ``X``, ``E``, ``y`` containing
                the extra features computed by :meth:`compute_extra_data`.
            node_mask: Boolean mask of shape ``(bs, n)``.

        Returns:
            PlaceHolder with predicted logits ``X`` ``(bs, n, dx_out)``,
            ``E`` ``(bs, n, n, de_out)``, and ``y`` ``(bs, dy_out)``.
        """
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes: int | torch.Tensor | None = None,
    ) -> list:
        """Sample a batch of graphs by iterating the reverse diffusion process.

        Starts from noise z_T ~ p(z_T) and iteratively samples
        z_{T-1}, z_{T-2}, …, z_0 using :meth:`sample_p_zs_given_zt`.
        Optionally records intermediate chain states and visualizes them.

        Args:
            batch_id: Identifier for this batch, used in visualization file paths.
            batch_size: Number of graphs to generate simultaneously.
            keep_chain: Number of graphs (≤ batch_size) for which the full
                denoising chain is recorded and saved.
            number_chain_steps: How many evenly-spaced chain frames to record
                out of the T total steps. Must be less than T.
            save_final: Number of final graphs to pass to the visualization tool.
            num_nodes: Controls the size of generated graphs. If ``None``,
                samples node counts from the learned distribution. If an ``int``,
                all graphs have that many nodes. If a ``Tensor`` of shape
                ``(batch_size,)``, each graph gets the specified count.

        Returns:
            List of ``batch_size`` elements, each a list ``[atom_types, edge_types]``
            where ``atom_types`` has shape ``(n,)`` and ``edge_types`` has shape
            ``(n, n)`` with integer class indices (CPU tensors).
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    writer = self.logger.experiment if self.logger else None
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        writer=writer,
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing molecules...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            writer = self.logger.experiment if self.logger else None
            self.visualization_tools.visualize(
                result_path, molecule_list, save_final, writer=writer
            )
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        y_t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple:
        """Sample z_s ~ p_θ(z_s | z_t) for one reverse-diffusion step.

        Implements the reverse step from Vignac et al. (2022), Eq. 3. The
        denoising network predicts x_0 from z_t; the predicted x_0 is then
        used to compute the posterior p(z_s | z_t, x̂_0) analytically via:

            p(z_s | z_t) = Σ_{x_0} p(x_0 | z_t) · q(z_s | z_t, x_0)

        where q(z_s | z_t, x_0) is computed using the batched-over-x_0
        posterior utility. The final z_s is sampled from these probabilities.

        Args:
            s: Normalized previous timestep of shape ``(bs, 1)``, i.e. (t-1)/T.
            t: Normalized current timestep of shape ``(bs, 1)``, i.e. t/T.
            X_t: Noisy node features at time t of shape ``(bs, n, dx)``.
            E_t: Noisy edge features at time t of shape ``(bs, n, n, de)``.
            y_t: Noisy global features at time t of shape ``(bs, dy)``.
            node_mask: Boolean mask of shape ``(bs, n)``.

        Returns:
            Tuple ``(out_one_hot, out_discrete)`` where both are PlaceHolders
            with fields ``X`` ``(bs, n, dx_out)`` and ``E`` ``(bs, n, n, de_out)``.
            ``out_one_hot`` contains float one-hot representations; ``out_discrete``
            contains collapsed integer class indices (via ``mask(collapse=True)``).
        """
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
        )

        p_s_and_t_given_0_E = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )
        )
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(
            node_mask, collapse=True
        ).type_as(y_t)

    def compute_extra_data(self, noisy_data: dict[str, torch.Tensor]):
        """Compute extra structural and domain features and append the timestep.

        Called at every training step (after adding noise) and at every reverse
        step during sampling. The resulting features are concatenated to the
        noisy graph before being fed to the denoising transformer.

        The normalized timestep ``t`` is appended to the global feature vector
        ``y`` so the network is aware of the current noise level.

        Args:
            noisy_data: Dict containing the noisy graph features and ``"t"``
                (normalized timestep of shape ``(bs, 1)``).

        Returns:
            PlaceHolder with fields ``X``, ``E``, ``y`` containing the
            concatenated extra features (structural + domain + timestep).
        """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
