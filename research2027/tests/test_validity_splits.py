from collections import Counter

import numpy as np
import pytest

from mockingbird2027.analysis.probes import ProbeConfig
from mockingbird2027.analysis.validity_splits import (
    build_probe_cv, build_roles, frozen_probe_cv, role_indices, validate_roles,
)
from mockingbird2027.data.manifest import ManifestValidationError
from test_m3 import _rows


def test_roles_are_exact_balanced_cell_disjoint_and_order_independent():
    rows = _rows(complete=True)
    assignments = build_roles(rows, 20260930)
    assert assignments == build_roles(list(reversed(rows)), 20260930)
    assert Counter(row["role"] for row in assignments) == {"A": 720, "B": 360, "C": 360}
    indices = role_indices(rows, assignments)
    assert not set(indices["A"]) & set(indices["B"])
    assert not set(indices["A"]) & set(indices["C"])
    assert not set(indices["B"]) & set(indices["C"])
    assert len(set(np.concatenate(list(indices.values())))) == 1440
    for role, count in (("A", 180), ("B", 90), ("C", 90)):
        assert len({row["cell_id"] for row in assignments if row["role"] == role}) == count
        assert len({rows[index].speaker_id for index in indices[role]}) == 24
        assert Counter(rows[index].statement_id for index in indices[role]) == {
            1: len(indices[role]) // 2, 2: len(indices[role]) // 2,
        }


def test_role_validation_rejects_cell_leakage_even_when_counts_match():
    rows = _rows(complete=True)
    assignments = build_roles(rows, 20260930)
    first = next(index for index, row in enumerate(assignments) if row["role"] == "A")
    second = next(index for index, row in enumerate(assignments) if row["role"] == "B")
    assignments[first]["role"], assignments[second]["role"] = "B", "A"
    with pytest.raises(ManifestValidationError, match="matched cell"):
        validate_roles(rows, assignments)


def test_frozen_probe_cv_uses_b_only_and_keeps_every_cell_and_class_intact():
    rows = _rows(complete=True)
    assignments = build_roles(rows, 20260930)
    config = ProbeConfig()
    records = build_probe_cv(rows, assignments, config)
    assert records == build_probe_cv(rows, assignments, config)
    folds = frozen_probe_cv(rows, assignments, records, 3)
    assert len(records) == 1800
    b_ids = {row["utterance_id"] for row in assignments if row["role"] == "B"}
    assert {row["utterance_id"] for row in records} == b_ids
    for group in folds.values():
        held_out = []
        for train, validation in group:
            assert (len(train), len(validation)) == (240, 120)
            assert not set(train) & set(validation)
            held_out.extend(validation)
        assert sorted(held_out) == list(range(360))


def test_cv_validation_rejects_cell_leakage_and_evaluation_role_ids():
    rows = _rows(complete=True)
    assignments = build_roles(rows, 20260930)
    records = build_probe_cv(rows, assignments, ProbeConfig())
    altered = [dict(row) for row in records]
    altered[0]["validation_fold"] = (altered[0]["validation_fold"] + 1) % 3
    with pytest.raises(ManifestValidationError, match="matched cell"):
        frozen_probe_cv(rows, assignments, altered, 3)
    altered = [dict(row) for row in records]
    altered[0]["utterance_id"] = next(row["utterance_id"] for row in assignments if row["role"] == "C")
    with pytest.raises(ManifestValidationError, match="B only"):
        frozen_probe_cv(rows, assignments, altered, 3)


def test_preparing_metadata_does_not_fit_any_representation_or_probe(monkeypatch):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from mockingbird2027.analysis.erasure import LEACE

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata preparation must not fit a model")

    for estimator in (PCA, LogisticRegression, StandardScaler, LEACE):
        monkeypatch.setattr(estimator, "fit", forbidden)
    rows = _rows(complete=True)
    assignments = build_roles(rows, 20260930)
    records = build_probe_cv(rows, assignments, ProbeConfig())
    assert len(frozen_probe_cv(rows, assignments, records, 3)) == 5
