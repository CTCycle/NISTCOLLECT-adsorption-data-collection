from __future__ import annotations

import numpy as np

from adsmod_core.services.modeling.fitting import compute_metrics

###############################################################################
def test_metrics_use_observed_minus_predicted_and_conditional_information_criteria() -> (
    None
):
    metrics = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0]), 1)
    assert metrics.values["sse"] == 1.0
    assert metrics.values["rmse"] == (1.0 / 3.0) ** 0.5
    assert "mse" not in metrics.values
    assert metrics.values["aic"] is not None

###############################################################################
def test_known_sigma_chi_square_and_weighted_likelihood() -> None:
    metrics = compute_metrics(
        np.array([1.0, 2.0]),
        np.array([0.5, 2.5]),
        1,
        np.array([0.5, 0.5]),
        "inverse_sigma",
    )
    assert metrics.values["chi_square"] == 2.0
    assert metrics.values["aic"] is not None
