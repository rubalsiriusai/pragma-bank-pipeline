"""AML GNN tests: transfer-graph build, handcrafted features, GraphSAGE, ablation plumbing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from pragmatiq.models.gnn import (
    AmlGNN,
    _train_val_test_mask,
)


class TestSplitDiscipline:
    """The ablation's fairness rests on a stratified train/val/test split that is
    disjoint and complete, so model selection (val) never touches the test set."""

    def test_masks_disjoint_complete_stratified(self) -> None:
        y = torch.tensor([0, 1] * 60 + [0] * 180)  # 300 nodes, 60 positives
        tr, va, te = _train_val_test_mask(len(y), y, seed=0)
        assert int((tr & va).sum()) == int((tr & te).sum()) == int((va & te).sum()) == 0
        assert int((tr | va | te).sum()) == len(y)  # every node assigned exactly once
        for name, m in (("train", tr), ("val", va), ("test", te)):
            assert int(y[m].sum()) > 0, f"{name} split has no positives (not stratified)"

requires_pyg = pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is None,
    reason="torch-geometric is required for the AML GNN model tests",
)




def _fake_embeddings(user_ids: list[str], dim: int = 16, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {u: rng.standard_normal(dim).astype(np.float32) for u in user_ids}






@requires_pyg
class TestAmlGNN:
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(20, 16)
        edge_index = torch.randint(0, 20, (2, 60))
        model = AmlGNN(16, hidden=32, n_layers=2)
        out = model(x, edge_index)
        assert out.shape == (20, 2)

    def test_three_layers(self) -> None:
        model = AmlGNN(8, hidden=16, n_layers=3)
        out = model(torch.randn(10, 8), torch.randint(0, 10, (2, 30)))
        assert out.shape == (10, 2)

    def test_rejects_bad_depth(self) -> None:
        with pytest.raises(ValueError):
            AmlGNN(8, n_layers=1)


class TestAblationPlumbing:


    def test_writeback_refuses_scale_downgrade(self, tmp_path: Path) -> None:
        from pragmatiq.models.gnn import write_aml_report

        big = ("# x\n\n## Results\n\n<!-- AML_ABLATION_RESULTS -->\n\nfull-scale table\n\n"
               "<sub>provenance: n_nodes=12000, n_edges=9, n_mules=9, seeds=[0], epochs=1, "
               "commit=abc</sub>\n\n## Next\n")
        readme = tmp_path / "README.md"
        readme.write_text(big)
        small = {
            "per_setup": {"a_isolated": {"mean": 0.7, "std": 0.0},
                          "b_gnn_pragma": {"mean": 0.7, "std": 0.0},
                          "c_gnn_handcrafted": {"mean": 0.7, "std": 0.0}},
            "raw": {"a_isolated": [0.7]},
            "verdict": {"b_beats_a": False, "c_beats_a": False, "b_beats_c": False, "pass": False},
            "n_nodes": 3000, "n_mules": 4, "n_edges": 12, "seeds": [0], "epochs": 1,
        }
        write_aml_report(small, readme_path=readme, notebook_path=None)
        assert "full-scale table" in readme.read_text(), "CI-scale must not clobber full-scale"


class TestGateMargin:
    """The gate-6 relational-recovery margin must be noise-aware, not a fixed 0.01."""

    def test_significant_margin_is_noise_aware(self) -> None:
        from pragmatiq.models.gnn import _significant_margin

        # Clearly separated, low cross-seed variance -> significant.
        assert _significant_margin([0.67, 0.66, 0.68], [0.60, 0.59, 0.61]) is True
        # Positive mean difference (~0.01) but swamped by cross-seed std (~0.06):
        # the old fixed-0.01 gate passes this; a noise-aware gate must reject it.
        assert _significant_margin([0.61, 0.55, 0.58], [0.59, 0.62, 0.50]) is False
        # Tiny margin below the floor even with zero variance -> rejected.
        assert _significant_margin([0.505, 0.505], [0.500, 0.500]) is False

    def test_markdown_cites_the_gated_noise_aware_flag(self) -> None:
        """The 'Relational recovery (gated)' line must print graph_recovers_signal
        (the noise-aware paired-seed test the gate uses), not c_beats_a (the fixed
        0.01 mean-margin reported flag), so the narrative matches the actual gate."""
        from pragmatiq.models.gnn import aml_results_markdown

        res = {
            "per_setup": {"a_isolated": {"mean": 0.60, "std": 0.05},
                          "b_gnn_pragma": {"mean": 0.61, "std": 0.05},
                          "c_gnn_handcrafted": {"mean": 0.62, "std": 0.05},
                          "d_lr_handcrafted": {"mean": 0.59, "std": 0.05}},
            "raw": {"a_isolated": [0.6]},
            # Divergent on purpose: the fixed-margin c_beats_a is True, but the
            # noise-aware gate (graph_recovers_signal) is False.
            "verdict": {"pass": False, "c_beats_a": True, "graph_recovers_signal": False,
                        "message_passing_adds": False, "b_beats_a": False, "b_beats_c": False,
                        "paper_ordering": False},
            "n_nodes": 100, "n_mules": 5, "n_edges": 40, "seeds": [0, 1, 2], "epochs": 10,
        }
        md = aml_results_markdown(res)
        assert "(c) > (a) = False" in md, "markdown must cite the gated noise-aware flag"
        assert "(c) > (a) = True" not in md


