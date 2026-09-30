import numpy as np
import pytest

from mockingbird2027.analysis.controls import (
    ControlMatchError,
    control_seed,
    matched_random_removal,
)
from mockingbird2027.analysis.erasure import (
    AffineRemoval,
    LEACE,
    MeanProjection,
    UnidentifiableErasure,
)
from mockingbird2027.analysis.probes import ProbeConfig, fit_fixed_probe, fit_probe


CONTROL_SETTINGS = {
    "power_bounds": [-4.0, 4.0],
    "bisection_steps": 80,
    "relative_rms_tolerance": 1e-5,
    "minimum_operator_difference": 0.001,
}


def synthetic(seed=7):
    generator = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), 200)
    mixing = generator.normal(size=(9, 9))
    features = generator.normal(size=(600, 9)) @ mixing
    features += np.eye(3)[labels] @ generator.normal(scale=2, size=(3, 9))
    features += np.arange(9) + 3.0
    return features, labels


@pytest.mark.parametrize("constructor", [MeanProjection, LEACE])
def test_multiclass_erasure_rank_covariance_mean_and_idempotence(constructor):
    features, labels = synthetic()
    original = features.copy()
    eraser = constructor().fit(features, labels)
    transformed = eraser.transform(features)
    assert eraser.removed_rank == 2
    np.testing.assert_array_equal(features, original)
    np.testing.assert_allclose(transformed.mean(axis=0), features.mean(axis=0), atol=1e-10)
    one_hot = np.eye(3)[labels]
    cross = (transformed - transformed.mean(axis=0)).T @ (one_hot - one_hot.mean(axis=0)) / len(features)
    np.testing.assert_allclose(cross, 0, atol=1e-9)
    np.testing.assert_allclose(eraser.transform(transformed), transformed, atol=1e-9)
    transform = eraser.transformation
    projection = np.eye(9) - transform.left @ transform.right.T
    assert np.linalg.matrix_rank(projection, tol=1e-7) == 7
    held_out = np.random.default_rng(44).normal(size=(15, 9))
    before = transform.left.copy()
    eraser.transform(held_out)
    np.testing.assert_array_equal(transform.left, before)


def test_leace_matches_closed_form_and_has_no_larger_training_mse_than_mp():
    features, labels = synthetic()
    leace = LEACE().fit(features, labels)
    mp = MeanProjection().fit(features, labels)
    assert leace.training_rms_ <= mp.training_rms_ + 1e-10
    centered = features - features.mean(axis=0)
    one_hot = np.eye(3)[labels]
    cross = centered.T @ (one_hot - one_hot.mean(axis=0)) / len(features)
    inverse = np.linalg.inv(centered.T @ centered / len(features))
    projection = np.eye(9) - cross @ np.linalg.pinv(cross.T @ inverse @ cross) @ cross.T @ inverse
    expected = centered @ projection.T + features.mean(axis=0)
    np.testing.assert_allclose(leace.transform(features), expected, atol=1e-8)
    assert not np.allclose(projection, projection.T)


@pytest.mark.parametrize("constructor", [MeanProjection, LEACE])
def test_singular_covariance_is_supported_without_ridge_or_test_data(constructor):
    generator = np.random.default_rng(12)
    labels = np.tile([0, 1], 50)
    latent = generator.normal(size=(100, 3))
    latent[:, 0] += labels * 2
    features = np.column_stack((latent, latent[:, 0], latent[:, 1] * 2, np.ones(100)))
    eraser = constructor().fit(features, labels)
    assert len(eraser.eigenvalues_) == 3
    assert eraser.removed_rank == 1
    assert np.isfinite(eraser.transform(features)).all()
    np.testing.assert_allclose(eraser.transform(features).mean(axis=0), features.mean(axis=0), atol=1e-10)


@pytest.mark.parametrize("constructor", [MeanProjection, LEACE])
def test_unidentifiable_and_invalid_inputs_fail_instead_of_returning_identity(constructor):
    features, labels = synthetic()
    with pytest.raises(RuntimeError, match="fit"):
        constructor().transform(features)
    with pytest.raises(UnidentifiableErasure, match="one training label"):
        constructor().fit(features, np.ones(len(features)))
    with pytest.raises(UnidentifiableErasure, match="covariance support"):
        constructor().fit(np.ones_like(features), labels)
    with pytest.raises(ValueError, match="align"):
        constructor().fit(features, labels[:-1])
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        constructor().fit(features, labels)


def test_scaled_perturbation_is_not_falsely_reported_as_rank_removal():
    basis = np.eye(4)[:, :1]
    with pytest.raises(ValueError, match="true-rank"):
        AffineRemoval(np.zeros(4), basis * 0.5, basis, "invalid-scaled-projector")


@pytest.mark.parametrize("constructor", [MeanProjection, LEACE])
def test_twenty_controls_match_true_rank_and_measured_rms_and_reproduce(constructor):
    generator = np.random.default_rng(7)
    labels = np.tile(np.arange(3), 200)
    features = generator.normal(size=(600, 9)) * np.geomspace(0.5, 10, 9)
    features[:, 2:4] += np.array([[2, 0], [-1, 2], [-1, -2]])[labels]
    eraser = constructor().fit(features, labels)
    factors = []
    for draw in range(20):
        seed = control_seed(20260930, f"synthetic-{eraser.method}", draw)
        removal, metadata = matched_random_removal(
            eraser, features, seed=seed, settings=CONTROL_SETTINGS
        )
        assert removal.removed_rank == eraser.removed_rank
        assert metadata["relative_rms_error"] <= 1e-5
        assert metadata["relative_operator_difference"] > 0.001
        projection = np.eye(features.shape[1]) - removal.left @ removal.right.T
        assert np.linalg.matrix_rank(projection, tol=1e-7) == features.shape[1] - eraser.removed_rank
        np.testing.assert_allclose(removal.transform(removal.transform(features)), removal.transform(features), atol=1e-8)
        full_difference = np.linalg.norm(
            eraser.transformation.left @ eraser.transformation.right.T - removal.left @ removal.right.T
        ) / np.linalg.norm(eraser.transformation.left @ eraser.transformation.right.T)
        assert metadata["relative_operator_difference"] == pytest.approx(full_difference)
        factors.append(removal.left)
    assert any(not np.allclose(factors[0], factor) for factor in factors[1:])
    first, metadata = matched_random_removal(
        eraser, features, seed=control_seed(20260930, f"synthetic-{eraser.method}", 0),
        settings=CONTROL_SETTINGS,
    )
    np.testing.assert_array_equal(first.left, factors[0])


def test_unmatchable_controls_are_not_resampled_or_silently_accepted():
    features, labels = synthetic()
    eraser = MeanProjection().fit(features, labels)
    with pytest.raises(ControlMatchError, match="bracket"):
        matched_random_removal(
            eraser, features, seed=5,
            settings={**CONTROL_SETTINGS, "power_bounds": [0.0, 0.0]},
        )
    full_rank = MeanProjection().fit(features[:, :2], labels)
    assert full_rank.removed_rank == 2
    with pytest.raises(ControlMatchError, match="full covariance-support"):
        matched_random_removal(full_rank, features[:, :2], seed=5, settings=CONTROL_SETTINGS)


def test_fixed_c_refit_reproduces_the_m3_selected_original_head():
    features, labels = synthetic()
    config = ProbeConfig()
    selected = fit_probe(features, labels.astype(str), config)
    refit = fit_fixed_probe(
        features, labels.astype(str), config, c=selected.best_params_["classifier__C"]
    )
    np.testing.assert_array_equal(selected.predict(features), refit.predict(features))
    np.testing.assert_allclose(selected.predict_proba(features), refit.predict_proba(features), atol=0, rtol=0)
