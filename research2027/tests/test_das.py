import copy
from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
from torch import nn

from mockingbird2027.analysis.das import (
    OrthogonalSubspace, arbitrary_targets, binary_labels, build_discovery_split,
    erasure_decision, fit_alignment, interchange, interchange_scores,
    matched_partners, mean_ablation, random_subspace_controls,
    readout_parameters, select_candidate, validate_discovery_split,
)
from mockingbird2027.analysis.m4 import parameters_hash
from mockingbird2027.analysis.validity_splits import build_roles, cell_id
from mockingbird2027.data.manifest import ManifestValidationError
from test_m3 import _rows


def test_discovery_split_preserves_frozen_roles_cells_marginals_and_actor_coverage():
    rows = _rows(complete=True)
    original = build_roles(rows, 20260930)
    split = build_discovery_split(rows, original, 20261008)
    assert split == build_discovery_split(list(reversed(rows)), original, 20261008)
    assert Counter(item["role"] for item in split) == {"A_train": 480, "A_val": 240, "B": 360, "C": 360}
    before = {item["utterance_id"]: item["role"] for item in original}
    by_id = {row.utterance_id: row for row in rows}
    cells = defaultdict(set)
    for item in split:
        assert ("A" if item["role"].startswith("A_") else item["role"]) == before[item["utterance_id"]]
        cells[item["cell_id"]].add(item["role"])
    assert all(len(roles) == 1 for roles in cells.values())
    for role, cell_count in (("A_train", 8), ("A_val", 4)):
        selected = [by_id[item["utterance_id"]] for item in split if item["role"] == role]
        counts = Counter((row.emotion, row.intensity) for row in selected)
        assert set(counts.values()) == {cell_count * 4}
        assert len({row.speaker_id for row in selected}) == 24
    leaked = copy.deepcopy(split)
    train = next(item for item in leaked if item["role"] == "A_train")
    val = next(item for item in leaked if item["role"] == "A_val")
    train["role"], val["role"] = val["role"], train["role"]
    with pytest.raises(ManifestValidationError, match="cell crosses"):
        validate_discovery_split(rows, original, leaked)


def test_pairs_change_only_statement_and_retain_both_directions():
    rows = _rows(complete=True)
    partners = matched_partners(rows)
    np.testing.assert_array_equal(partners[partners], np.arange(len(rows)))
    for index, row in enumerate(rows):
        source = rows[partners[index]]
        assert row.utterance_id != source.utterance_id
        assert row.statement_id != source.statement_id
        assert (row.speaker_id, row.emotion, row.intensity, row.repetition) == (
            source.speaker_id, source.emotion, source.intensity, source.repetition,
        )
    with pytest.raises(ManifestValidationError, match="both|statements"):
        matched_partners(rows[1:])
    with pytest.raises(ManifestValidationError, match="unique"):
        matched_partners([*rows, rows[0]])


def test_arbitrary_mapping_is_cell_grouped_balanced_and_deterministic():
    rows = _rows(complete=True)
    partners = matched_partners(rows)
    targets, mapping = arbitrary_targets(rows, partners, 42)
    repeated, repeated_mapping = arbitrary_targets(rows, partners, 42)
    np.testing.assert_array_equal(targets, repeated)
    assert mapping == repeated_mapping
    assert Counter(mapping.values()) == {0: 180, 1: 180}
    actual_flips = targets ^ binary_labels(rows)[partners]
    assert all(actual_flips[index] == mapping[cell_id(row)] for index, row in enumerate(rows))
    np.testing.assert_array_equal(targets[partners], 1 - targets)


@pytest.mark.parametrize("rank", [1, 3, 7])
def test_orthogonality_and_equivalence_to_full_rotated_interchange(rank):
    torch.set_num_threads(1)
    alignment = OrthogonalSubspace(9, rank, 7)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=0.01)
    base = torch.tensor(np.random.default_rng(11).normal(size=(12, 9)), dtype=torch.float64)
    source = torch.flip(base, dims=[0])
    for _ in range(3):
        optimizer.zero_grad()
        interchange(base, source, alignment.basis).square().mean().backward()
        optimizer.step()
    q = alignment.basis.detach().numpy()
    np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=1e-12)
    full, _ = np.linalg.qr(q, mode="complete")
    rotated = base.numpy() @ full
    rotated[:, :rank] = (source.numpy() @ full)[:, :rank]
    expected = rotated @ full.T
    actual = interchange(base, source, alignment.basis).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(interchange(base, base, alignment.basis).detach(), base, atol=1e-12)
    with pytest.raises(ValueError, match="rank"):
        OrthogonalSubspace(9, 9, 7)


def test_mean_ablation_preserves_complement_sets_target_mean_and_is_idempotent():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(60, 10)) + 4
    q, _ = np.linalg.qr(rng.normal(size=(10, 4)), mode="reduced")
    mean = x.mean(axis=0)
    removal = mean_ablation(mean, q)
    erased = removal.transform(x)
    np.testing.assert_allclose(erased @ q, np.broadcast_to(mean @ q, (len(x), 4)), atol=1e-12)
    complement = np.eye(10) - q @ q.T
    np.testing.assert_allclose(erased @ complement, x @ complement, atol=1e-12)
    np.testing.assert_allclose(removal.transform(erased), erased, atol=1e-12)
    assert removal.removed_rank == 4
    centered = x - mean
    np.testing.assert_allclose(
        mean_ablation(centered.mean(axis=0), q).transform(centered),
        centered - (centered @ q) @ q.T, atol=1e-12,
    )
    with pytest.raises(ValueError, match="idempotent"):
        mean_ablation(mean, q * 2)


def test_das_optimizes_source_interchange_without_updating_the_readout():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    x = np.repeat(rng.normal(size=(50, 9)), 2, axis=0)
    x[:, 0] = np.tile([-2.0, 2.0], 50)
    v = np.repeat(rng.normal(size=(20, 9)), 2, axis=0)
    v[:, 0] = np.tile([-2.0, 2.0], 20)
    p, vp = np.arange(len(x)) ^ 1, np.arange(len(v)) ^ 1
    y, vy = (x[:, 0] > 0).astype(np.int64), (v[:, 0] > 0).astype(np.int64)
    model = nn.Linear(9, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0], model.weight[1, 0] = -1, 1
        model.bias.zero_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = parameters_hash(readout_parameters(model))
    settings = {"steps": 70, "learning_rate": 0.04, "evaluation_interval": 10}
    fit = fit_alignment(model, x, p, y[p], v, vp, vy[vp], 1, 17, settings)
    assert parameters_hash(readout_parameters(model)) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert fit.basis[0, 0] ** 2 > 0.8
    assert interchange_scores(model, v, vp, vy[vp], fit.basis)["accuracy"] == 1
    assert fit.records[-1]["train_cross_entropy"] < fit.records[0]["train_cross_entropy"]
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable behavioral"):
        fit_alignment(model, x, p, y[p], v, vp, vy[vp], 1, 17, settings)


def candidate(identity, dimension, seed, linear=0.6, rbf=0.7, iia=0.95, collateral=True):
    return {
        "candidate_id": identity, "dimension": dimension, "seed": seed,
        "linear_statement_f1": linear, "rbf_statement_f1": rbf,
        "interchange_accuracy": iia, "collateral_pass": collateral, "evaluation_role": "A_val",
    }


def test_candidate_selection_is_deterministic_uses_both_probes_and_reports_ineligibility():
    rows = [
        candidate("a", 1, 7, linear=0.3, rbf=0.9),
        candidate("b", 8, 9),
        candidate("c", 4, 9),
        candidate("d", 4, 7),
        candidate("damaging", 1, 1, linear=0.4, rbf=0.4, collateral=False),
    ]
    selected, status = select_candidate(rows, 0.9)
    assert selected["candidate_id"] == "d"
    assert status == "ELIGIBLE_A_CANDIDATE"
    assert select_candidate(list(reversed(rows)), 0.9) == (selected, status)
    assert select_candidate([candidate("bad", 1, 7, iia=0.3)], 0.9)[1] == "NO_ELIGIBLE_A_CANDIDATE_DIAGNOSTIC_ONLY"
    rows[0]["evaluation_role"] = "C"
    with pytest.raises(ValueError, match="only use A_val"):
        select_candidate(rows, 0.9)


def test_random_controls_keep_rank_and_never_rescale_to_fake_matching():
    x = np.random.default_rng(12).normal(size=(80, 10))
    q = np.eye(10)[:, :3]
    target = mean_ablation(x.mean(axis=0), q)
    settings = {"pool_size": 7, "count": 2, "seed": 31, "relative_rms_tolerance": 0.05}
    controls, records = random_subspace_controls(x, target, settings)
    assert len(records) == 7 and sum(row["selected"] for row in records) == 2
    selected = sorted(row["relative_rms_error"] for row in records if row["selected"])
    assert selected == sorted(row["relative_rms_error"] for row in records)[:2]
    for removal in controls.values():
        assert removal.removed_rank == 3
        np.testing.assert_allclose(removal.right.T @ removal.right, np.eye(3), atol=1e-12)
    assert random_subspace_controls(x, target, settings)[1] == records


def test_decision_requires_both_probes_both_metrics_precision_and_collateral():
    policy = {
        "before_min_macro_f1": 0.6, "minimum_target_f1_drop": 0.1,
        "after_max_macro_f1": 0.55, "after_max_accuracy": 0.55,
        "upper_confidence_ceiling": 0.55, "maximum_emotion_f1_drop": 0.1,
        "large_unrelated_f1_drop": 0.2, "maximum_large_unrelated_losses": 1,
    }
    rows = []
    for arm in ("original", "das_mean"):
        for probe, attribute in [
            *(("linear", name) for name in ("emotion", "speaker", "statement", "gender", "intensity")),
            ("rbf_svm", "statement"),
        ]:
            score = 0.5 if arm == "das_mean" and attribute == "statement" else 0.9
            rows.append({
                "arm": arm, "probe": probe, "attribute": attribute,
                "macro_f1": score, "accuracy": score,
                "macro_f1_upper_95": score + 0.02, "accuracy_upper_95": score + 0.02,
            })
    assert erasure_decision(rows, policy)["decision"] == "PASS_VALIDITY_ONLY"
    changed = copy.deepcopy(rows)
    changed[-1]["accuracy"] = 0.56
    assert erasure_decision(changed, policy)["decision"] == "FAIL_RESIDUAL_DECODABILITY"
    changed = copy.deepcopy(rows)
    changed[-1]["accuracy_upper_95"] = 0.56
    assert erasure_decision(changed, policy)["decision"] == "INCONCLUSIVE_PRECISION"
    changed = copy.deepcopy(rows)
    next(row for row in changed if row["arm"] == "das_mean" and row["attribute"] == "emotion")["macro_f1"] = 0.5
    assert erasure_decision(changed, policy)["decision"] == "FAIL_COLLATERAL_DAMAGE"
