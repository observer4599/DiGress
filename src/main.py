"""Entry point for DiGress training and evaluation.

DiGress (Vignac et al., 2022) is a discrete denoising diffusion model for
graph generation. Noise is applied through categorical Markov transitions
rather than Gaussian perturbations, and a graph transformer learns to reverse
the process.

This module wires together dataset-specific components and the generic
diffusion model into a runnable experiment. The high-level flow is:

  1. A ``setup_*_components`` function loads the dataset and constructs the
     objects needed by ``DiscreteDenoisingDiffusion`` — metrics, feature
     extractors, and dataset statistics.
  2. ``build_trainer`` creates a PyTorch Lightning ``Trainer``, selecting the
     best available accelerator (CUDA > Apple MPS > CPU).
  3. ``main`` dispatches to ``trainer.fit`` for training or ``trainer.test``
     for checkpoint evaluation.

Supported datasets:

  - SPECTRE graph benchmarks: ``sbm``, ``comm20``, ``planar``.
  - Molecular graphs: ``qm9``, ``guacamol``, ``moses``.
"""

import dataclasses
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import pathlib
import warnings
from typing import Any

from loguru import logger
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics import TrainAbstractMetricsDiscrete
from model import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


@dataclasses.dataclass
class ModelComponents:
    """Dataset-specific objects consumed by DiscreteDenoisingDiffusion and the trainer.

    Bundles the data module alongside the model constructor arguments so
    callers track them as a unit. Use model_kwargs() to unpack the subset
    that DiscreteDenoisingDiffusion.__init__ accepts.

    Attributes:
        datamodule: Lightning data module providing train/val/test loaders.
        dataset_infos: Dataset statistics (node counts, edge types, input/output dims).
        train_metrics: Metric accumulator called each training step.
        sampling_metrics: Metrics evaluated on generated samples.
        visualization_tools: Saves sample visualizations to disk.
        extra_features: Computes graph-structural extra features.
        domain_features: Computes domain-specific features (e.g. molecular charge).
    """

    datamodule: Any
    dataset_infos: Any
    train_metrics: Any
    sampling_metrics: Any
    visualization_tools: Any
    extra_features: Any
    domain_features: Any

    def model_kwargs(self) -> dict[str, Any]:
        """Return all fields except ``datamodule`` as model constructor kwargs.

        ``DiscreteDenoisingDiffusion.__init__`` receives metrics, features, and
        dataset info directly. The ``datamodule`` is intentionally excluded
        because it is passed separately to ``Trainer.fit`` / ``Trainer.test``,
        not to the model constructor.
        """
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "datamodule"
        }


def get_resume(
    cfg: DictConfig,
    components: ModelComponents,
) -> tuple[DictConfig, DiscreteDenoisingDiffusion]:
    """Load a checkpoint for test-only mode without allowing config changes.

    Restores the full config saved in the checkpoint, then back-fills any
    keys that were added after the checkpoint was created.

    Args:
        cfg: Current Hydra config. Reads cfg.general.test_only and cfg.general.name.
        components: Dataset-specific model components forwarded to load_from_checkpoint.

    Returns:
        Tuple of (updated_cfg, loaded_model).
    """
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **components.model_kwargs())
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(
    cfg: DictConfig,
    components: ModelComponents,
) -> tuple[DictConfig, DiscreteDenoisingDiffusion]:
    """Load a checkpoint for training resumption, allowing config overrides.

    Loads the checkpoint config as the base, then overwrites every key with
    the current run config so that training hyperparameters can be adjusted
    without starting from scratch.

    Args:
        cfg: Current Hydra config. Reads cfg.general.resume (relative checkpoint path).
        components: Dataset-specific model components forwarded to load_from_checkpoint.

    Returns:
        Tuple of (updated_cfg, loaded_model).
    """
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **components.model_kwargs())
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"
    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_spectre_components(cfg: DictConfig) -> ModelComponents:
    """Build all dataset-specific components for SPECTRE graph datasets.

    Loads the data module, constructs the appropriate sampling metrics class
    (sbm, comm20, or planar), computes dataset statistics and input/output
    dims, and returns all components as a ModelComponents instance.

    Args:
        cfg: Hydra config. Reads cfg.dataset.name and cfg.model.extra_features.

    Returns:
        A fully populated ModelComponents instance.
    """
    from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
    from analysis.spectre_utils import (
        Comm20SamplingMetrics,
        PlanarSamplingMetrics,
        SBMSamplingMetrics,
    )
    from analysis.visualization import NonMolecularVisualization

    _METRICS = {
        "sbm": SBMSamplingMetrics,
        "comm20": Comm20SamplingMetrics,
        "planar": PlanarSamplingMetrics,
    }

    logger.info("Loading {} dataset...", cfg.dataset.name)
    datamodule = SpectreGraphDataModule(cfg)

    logger.info("Building sampling metrics (converting all splits to NetworkX)...")
    sampling_metrics = _METRICS[cfg.dataset.name](datamodule)
    logger.info("Sampling metrics ready.")

    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
    train_metrics = TrainAbstractMetricsDiscrete()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.extra_features is not None:
        logger.info("Computing extra features (type={})...", cfg.model.extra_features)
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    logger.info("Computing input/output dims...")
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    return ModelComponents(
        datamodule=datamodule,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )


def setup_molecular_components(cfg: DictConfig) -> ModelComponents:
    """Build all dataset-specific components for molecular graph datasets.

    Handles qm9, guacamol, and moses. Loads the appropriate data module and
    dataset info class, retrieves training SMILES for qm9, sets up molecular
    extra features, and returns all components as a ModelComponents instance.

    Args:
        cfg: Hydra config. Reads cfg.dataset.name and cfg.model.extra_features.

    Returns:
        A fully populated ModelComponents instance.

    Raises:
        ValueError: If cfg.dataset.name is not one of 'qm9', 'guacamol', 'moses'.
    """
    from metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
    from diffusion.extra_features_molecular import ExtraMolecularFeatures
    from analysis.visualization import MolecularVisualization

    dataset_name = cfg.dataset.name
    if dataset_name == "qm9":
        from datasets import qm9_dataset
        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        train_smiles = qm9_dataset.get_train_smiles(
            cfg=cfg,
            train_dataloader=datamodule.train_dataloader(),
            dataset_infos=dataset_infos,
            evaluate_dataset=False,
        )
    elif dataset_name == "guacamol":
        from datasets import guacamol_dataset
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
        train_smiles = None
    elif dataset_name == "moses":
        from datasets import moses_dataset
        datamodule = moses_dataset.MosesDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
        train_smiles = None
    else:
        raise ValueError(f"Unknown molecular dataset: {dataset_name!r}")

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    # Novelty is not evaluated during training.
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    return ModelComponents(
        datamodule=datamodule,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
    )


def build_callbacks(cfg: DictConfig) -> list[Callback]:
    """Build the PyTorch Lightning callback list from config.

    Creates up to three callbacks:
    - Two ModelCheckpoint callbacks when cfg.train.save_model is True: one
      saving the top-5 checkpoints by val/epoch_NLL and one always saving
      'last.ckpt'.
    - One EMA callback when cfg.train.ema_decay > 0.

    Args:
        cfg: Hydra config. Reads cfg.train.save_model, cfg.general.name,
            and cfg.train.ema_decay.

    Returns:
        List of callback objects to pass to pytorch_lightning.Trainer.
    """
    callbacks = []
    if cfg.train.save_model:
        best_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            monitor="val/epoch_NLL",
            save_top_k=5,
            mode="min",
            every_n_epochs=1,
        )
        last_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="last",
            every_n_epochs=1,
        )
        callbacks.extend([last_ckpt, best_ckpt])

    if cfg.train.ema_decay > 0:
        # TODO: utils.EMA is not currently defined — implement or import before enabling.
        callbacks.append(utils.EMA(decay=cfg.train.ema_decay))

    return callbacks


def build_trainer(cfg: DictConfig, callbacks: list) -> Trainer:
    """Build and configure the PyTorch Lightning Trainer.

    Selects the accelerator in priority order (CUDA GPU > Apple MPS > CPU),
    attaches a TensorBoard logger when wandb is not 'disabled', and wires up
    all training hyperparameters from the config.

    Args:
        cfg: Hydra config. Reads cfg.general and cfg.train sub-configs.
        callbacks: List of Lightning callbacks to attach to the trainer.

    Returns:
        A configured pytorch_lightning.Trainer instance.
    """
    use_cuda = cfg.general.gpus > 0 and torch.cuda.is_available()
    use_mps = not use_cuda and torch.backends.mps.is_available()

    if use_cuda:
        accelerator, devices = "gpu", cfg.general.gpus
        strategy = "ddp_find_unused_parameters_true"  # Required for loading old checkpoints.
    elif use_mps:
        accelerator, devices = "mps", 1
        strategy = "auto"
    else:
        accelerator, devices = "cpu", 1
        strategy = "auto"

    loggers = []
    if cfg.general.wandb != "disabled":
        loggers.append(TensorBoardLogger(
            save_dir="runs",
            name=f"graph_ddm_{cfg.dataset.name}",
            version=cfg.general.name,
        ))

    return Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if cfg.general.name != "debug" else 1,
        logger=loggers,
    )


def _evaluate_all_checkpoints(
    trainer: Trainer,
    model: DiscreteDenoisingDiffusion,
    datamodule: Any,
    cfg: DictConfig,
) -> None:
    """Run trainer.test on every checkpoint in the test_only directory.

    Skips the checkpoint already evaluated by the caller. Useful for
    evaluating all saved snapshots in a single run.

    Args:
        trainer: Configured Lightning Trainer.
        model: Already-loaded model instance.
        datamodule: Dataset data module providing the test split.
        cfg: Hydra config. Reads cfg.general.test_only for the directory path.
    """
    directory = pathlib.Path(cfg.general.test_only).parent
    logger.info("Evaluating all checkpoints in {}", directory)
    for ckpt_path in directory.glob("*.ckpt"):
        if str(ckpt_path) == cfg.general.test_only:
            continue
        logger.info("Loading checkpoint {}", ckpt_path)
        trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))


_SPECTRE_DATASETS = {"sbm", "comm20", "planar"}
_MOLECULAR_DATASETS = {"qm9", "guacamol", "moses"}


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for DiGress training and evaluation.

    Dispatches to the appropriate dataset setup function, optionally resumes
    from a checkpoint, then runs training or test-only evaluation.

    Args:
        cfg: Hydra config composed from configs/config.yaml and overrides.
    """
    dataset_name = cfg.dataset.name
    if dataset_name in _SPECTRE_DATASETS:
        components = setup_spectre_components(cfg)
    elif dataset_name in _MOLECULAR_DATASETS:
        components = setup_molecular_components(cfg)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name!r}")

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, components)
        os.chdir(cfg.general.test_only.split("checkpoints")[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, components)
        os.chdir(cfg.general.resume.split("checkpoints")[0])

    utils.create_folders(cfg)

    if cfg.general.name == "debug":
        logger.warning("Run name is 'debug' — fast_dev_run is active.")

    logger.info("Initialising model...")
    model = DiscreteDenoisingDiffusion(cfg=cfg, **components.model_kwargs())
    trainer = build_trainer(cfg, build_callbacks(cfg))

    if cfg.general.test_only:
        trainer.test(model, datamodule=components.datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            _evaluate_all_checkpoints(trainer, model, components.datamodule, cfg)
    else:
        logger.info("Starting training...")
        trainer.fit(model, datamodule=components.datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in {"debug", "test"}:
            trainer.test(model, datamodule=components.datamodule)


if __name__ == "__main__":
    main()
