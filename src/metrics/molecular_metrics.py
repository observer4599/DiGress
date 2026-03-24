import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, Metric
from torch.utils.tensorboard import SummaryWriter

from src.analysis.rdkit_functions import compute_molecular_metrics


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for atom_types, _ in molecules:
            self.n_dist[atom_types.shape[0]] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            assert (atom_types != -1).all(), (
                "Mask error, the molecules should already be masked at the right shape"
            )
            self.node_dist += torch.bincount(
                atom_types.long(), minlength=len(self.node_dist)
            ).float()

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.triu(torch.ones_like(edge_types), diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(
                edge_types, return_counts=True
            )
            self.edge_dist.scatter_add_(
                0, unique_edge_types.long(), counts.float()
            )

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            self.edgepernode_dist.scatter_add_(0, unique.long(), counts.float())

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred: torch.Tensor) -> None:
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred / pred.sum(), self.target_histogram)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, train_smiles):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims["X"])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims["E"])
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        n_target_dist = self._register_target_dist("n_target_dist", di.n_nodes, self.generated_n_dist.n_dist)
        node_target_dist = self._register_target_dist("node_target_dist", di.node_types, self.generated_node_dist.node_dist)
        edge_target_dist = self._register_target_dist("edge_target_dist", di.edge_types, self.generated_edge_dist.edge_dist)
        valency_target_dist = self._register_target_dist("valency_target_dist", di.valency_distribution, self.generated_valency_dist.edgepernode_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.dataset_info = di

    def _register_target_dist(self, name: str, raw: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Normalizes ``raw`` to a probability distribution, registers it as a buffer, and returns it."""
        dist = raw.type_as(ref)
        dist = dist / dist.sum()
        self.register_buffer(name, dist)
        return dist

    def _update_dist(
        self, dist_metric: Metric, mae_metric: HistogramsMAE, molecules: list
    ) -> torch.Tensor:
        """Updates a distribution metric and its MAE tracker, returning the normalized distribution."""
        dist_metric(molecules)
        result = dist_metric.compute()
        mae_metric(result)
        return result

    def forward(
        self,
        molecules: list,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        writer: SummaryWriter | None = None,
        global_step: int = 0,
    ) -> dict[str, float]:
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            molecules, self.train_smiles, self.dataset_info
        )

        if test and local_rank == 0:
            with open(r"final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    fp.write("%s\n" % smiles)
                print("All smiles saved")

        print("Starting custom metrics")
        generated_n_dist = self._update_dist(self.generated_n_dist, self.n_dist_mae, molecules)
        generated_node_dist = self._update_dist(self.generated_node_dist, self.node_dist_mae, molecules)
        generated_edge_dist = self._update_dist(self.generated_edge_dist, self.edge_dist_mae, molecules)
        generated_valency_dist = self._update_dist(self.generated_valency_dist, self.valency_dist_mae, molecules)

        to_log = self._build_to_log(
            generated_node_dist, generated_edge_dist, generated_valency_dist, stability, rdkit_metrics
        )

        if writer is not None:
            for key, val in to_log.items():
                writer.add_scalar(key, val, global_step)
            for tag, dist in [
                ("generated/n_dist", generated_n_dist),
                ("generated/node_dist", generated_node_dist),
                ("generated/edge_dist", generated_edge_dist),
                ("generated/valency_dist", generated_valency_dist),
            ]:
                writer.add_histogram(tag, dist, global_step)

        if local_rank == 0:
            print("Custom metrics computed.")
            with open(f"graphs/{name}/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt", "w") as f:
                f.writelines(rdkit_metrics[1])
            print("Stability metrics:", stability, "--", rdkit_metrics[0])
        return to_log

    def _build_to_log(
        self,
        generated_node_dist: torch.Tensor,
        generated_edge_dist: torch.Tensor,
        generated_valency_dist: torch.Tensor,
        stability: dict,
        rdkit_metrics: tuple,
    ) -> dict[str, float]:
        """Assembles the metric logging dict from generated distributions and RDKit outputs."""
        to_log = {}
        for i, atom_type in enumerate(self.dataset_info.atom_decoder):
            to_log[f"molecular_metrics/{atom_type}_dist"] = (generated_node_dist[i] - self.node_target_dist[i]).item()
        for j, bond_type in enumerate(["No bond", "Single", "Double", "Triple", "Aromatic"]):
            to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (generated_edge_dist[j] - self.edge_target_dist[j]).item()
        for valency in range(6):
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (generated_valency_dist[valency] - self.valency_target_dist[valency]).item()

        to_log["molecular_metrics/n_mae"] = self.n_dist_mae.compute()
        to_log["molecular_metrics/node_mae"] = self.node_dist_mae.compute()
        to_log["molecular_metrics/edge_mae"] = self.edge_dist_mae.compute()
        to_log["molecular_metrics/valency_mae"] = self.valency_dist_mae.compute()

        to_log["molecular_metrics/mol_stable"] = stability["mol_stable"]
        to_log["molecular_metrics/atm_stable"] = stability["atm_stable"]

        validity, relaxed_validity, uniqueness, novelty = rdkit_metrics[0]
        to_log["molecular_metrics/validity"] = validity
        to_log["molecular_metrics/relaxed_validity"] = relaxed_validity
        to_log["molecular_metrics/uniqueness"] = uniqueness
        to_log["molecular_metrics/novelty"] = novelty
        return to_log

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()
