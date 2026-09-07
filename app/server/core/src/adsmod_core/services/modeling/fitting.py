from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from adsmod_core.contracts.fitting import (
    FitMetrics,
    FittedParameter,
    FittingRequest,
    ModelDefinition,
    ModelFitResult,
    ModelParameterDefinition,
    PredictionPoint,
)
from adsmod_common.units import UnitConversionError, UnitRegistry
from adsmod_core.services.modeling.models import (
    AdsorptionModels,
    ModelSpec,
    ParameterSpec,
)


CONDITION_WARNING = 1e12
CURVE_POINT_COUNT = 200
MODEL_VERSION = "2.0"

###############################################################################
@dataclass
class MetricResult:
    values: dict[str, float | None]
    warnings: list[str] = field(default_factory=list)

###############################################################################
@dataclass
class FitComputation:
    spec: ModelSpec
    status: str
    message: str
    parameters: np.ndarray | None
    standard_errors: np.ndarray | None
    ci95_low: np.ndarray | None
    ci95_high: np.ndarray | None
    metrics: MetricResult
    predicted: np.ndarray
    curve_pressure: np.ndarray
    curve_uptake: np.ndarray
    function_evaluations: int | None
    jacobian_rank: int | None
    condition_number: float | None
    warnings: list[str] = field(default_factory=list)
    rank: int | None = None

###############################################################################
def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None

###############################################################################
def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
    uncertainty: np.ndarray | None = None,
    weighting: str = "unweighted",
) -> MetricResult:
    residuals = observed - predicted
    n = int(observed.size)
    k = int(parameter_count)
    warnings: list[str] = []
    sse = float(np.sum(residuals**2, dtype=np.float64))
    rmse = math.sqrt(sse / n) if n > 0 else None
    mae = float(np.mean(np.abs(residuals))) if n > 0 else None

    centered = observed - float(np.mean(observed)) if n else observed
    tss = float(np.sum(centered**2, dtype=np.float64))
    if n == 0 or tss <= 0:
        r_squared = None
        warnings.append(
            "R² is undefined because the observed uptake has zero variance."
        )
    else:
        r_squared = 1.0 - sse / tss

    if r_squared is None or n <= k + 1:
        adjusted_r_squared = None
        warnings.append(
            "Adjusted R² is undefined because there are insufficient residual degrees of freedom."
        )
    else:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)

    if uncertainty is None:
        chi_square = None
    elif (
        uncertainty.size != n
        or np.any(~np.isfinite(uncertainty))
        or np.any(uncertainty <= 0)
    ):
        chi_square = None
        warnings.append(
            "Chi-square is undefined because uptake uncertainties are incomplete or invalid."
        )
    else:
        chi_square = float(np.sum((residuals / uncertainty) ** 2))

    if (
        weighting == "inverse_sigma"
        and uncertainty is not None
        and uncertainty.size == n
        and np.all(np.isfinite(uncertainty))
        and np.all(uncertainty > 0)
    ):
        # Known-sigma Gaussian likelihood; model parameters only are counted.
        chi_square = float(np.sum((residuals / uncertainty) ** 2))
        log_likelihood = float(
            -0.5 * (chi_square + np.sum(np.log(2.0 * math.pi * uncertainty**2)))
        )
        likelihood_k = k
    elif n > 0 and sse > 0:
        # Unknown common variance estimated from residuals; count that variance parameter.
        log_likelihood = float(-0.5 * n * (math.log(2.0 * math.pi * sse / n) + 1.0))
        likelihood_k = k + 1
    else:
        log_likelihood = None
        likelihood_k = k + 1
        warnings.append(
            "Information criteria are undefined for an empty or exact-zero-residual fit."
        )
    if log_likelihood is None:
        aic = aicc = bic = None
    else:
        aic = -2.0 * log_likelihood + 2.0 * likelihood_k
        bic = -2.0 * log_likelihood + likelihood_k * math.log(n) if n > 0 else None
        aicc = (
            aic + (2.0 * likelihood_k * (likelihood_k + 1.0)) / (n - likelihood_k - 1.0)
            if n > likelihood_k + 1
            else None
        )
        if aicc is None:
            warnings.append(
                "AICc is undefined because the sample size is too small for the likelihood parameter count."
            )

    return MetricResult(
        values={
            "sse": finite_or_none(sse),
            "rmse": finite_or_none(rmse) if rmse is not None else None,
            "mae": finite_or_none(mae) if mae is not None else None,
            "r_squared": finite_or_none(r_squared) if r_squared is not None else None,
            "adjusted_r_squared": finite_or_none(adjusted_r_squared)
            if adjusted_r_squared is not None
            else None,
            "chi_square": finite_or_none(chi_square)
            if chi_square is not None
            else None,
            "aic": finite_or_none(aic) if aic is not None else None,
            "aicc": finite_or_none(aicc) if aicc is not None else None,
            "bic": finite_or_none(bic) if bic is not None else None,
        },
        warnings=warnings,
    )

###############################################################################
def pressure_factor(unit: str, pressure_basis: str) -> float:
    resolved = UnitRegistry.pressure_unit(unit)
    if pressure_basis == "relative":
        if resolved == "1":
            return 1.0
        if resolved == "%":
            return 0.01
        raise UnitConversionError(
            "Relative-pressure results can be displayed only as p/p0 or percent."
        )
    if resolved not in UnitRegistry.PRESSURE_TO_PA:
        raise UnitConversionError("A dimensional pressure display unit is required.")
    return UnitRegistry.PRESSURE_TO_PA[resolved]

###############################################################################
def parameter_unit(
    parameter: ParameterSpec,
    *,
    pressure_unit: str,
    uptake_unit: str,
    related_value: float | None = None,
) -> str:
    kind = parameter.unit_kind
    if kind == "pressure^-1":
        return f"{pressure_unit}⁻¹"
    if kind == "uptake":
        return uptake_unit
    if kind == "dimensionless":
        return "1"
    if kind == "energy^-2":
        return "mol²/J²"
    if kind == "uptake/pressure":
        return f"{uptake_unit}/{pressure_unit}"
    if kind == "pressure^-beta":
        exponent = related_value if related_value is not None else "β"
        return (
            f"{pressure_unit}^(-{exponent:g})"
            if isinstance(exponent, float)
            else f"{pressure_unit}^(-β)"
        )
    if kind == "freundlich":
        exponent = related_value if related_value is not None else "n"
        return (
            f"{uptake_unit}·{pressure_unit}^(-1/{exponent:g})"
            if isinstance(exponent, float)
            else f"{uptake_unit}·{pressure_unit}^(-1/n)"
        )
    return "1"

###############################################################################
def parameter_to_display(
    parameter: ParameterSpec,
    value: float,
    *,
    pressure_factor_value: float,
    uptake_factor_value: float,
    related_value: float | None,
) -> float:
    kind = parameter.unit_kind
    if kind == "pressure^-1":
        return value * pressure_factor_value
    if kind == "uptake":
        return value / uptake_factor_value
    if kind in {"dimensionless", "energy^-2"}:
        return value
    if kind == "uptake/pressure":
        return value * pressure_factor_value / uptake_factor_value
    if kind == "pressure^-beta":
        return value * pressure_factor_value ** float(related_value or 1.0)
    if kind == "freundlich":
        return (
            value
            * pressure_factor_value ** (1.0 / float(related_value or 1.0))
            / uptake_factor_value
        )
    return value

###############################################################################
def display_parameter_value(
    value: float | None,
    parameter: ParameterSpec,
    *,
    pressure_factor_value: float,
    uptake_factor_value: float,
    related_value: float,
) -> float | None:
    if value is None:
        return None
    return parameter_to_display(
        parameter,
        float(value),
        pressure_factor_value=pressure_factor_value,
        uptake_factor_value=uptake_factor_value,
        related_value=related_value,
    )

###############################################################################
def parameter_from_display(
    parameter: ParameterSpec,
    value: float,
    *,
    pressure_factor_value: float,
    uptake_factor_value: float,
    related_value: float | None,
) -> float:
    kind = parameter.unit_kind
    if kind == "pressure^-1":
        return value / pressure_factor_value
    if kind == "uptake":
        return value * uptake_factor_value
    if kind in {"dimensionless", "energy^-2"}:
        return value
    if kind == "uptake/pressure":
        return value * uptake_factor_value / pressure_factor_value
    if kind == "pressure^-beta":
        return value / pressure_factor_value ** float(related_value or 1.0)
    if kind == "freundlich":
        return (
            value
            * uptake_factor_value
            / pressure_factor_value ** (1.0 / float(related_value or 1.0))
        )
    return value

###############################################################################
def _fit_residual(
    parameters: np.ndarray,
    models: AdsorptionModels,
    model_name: str,
    pressure: np.ndarray,
    uptake: np.ndarray,
    temperature_k: float,
    pressure_basis: str,
    saturation_pressure_pa: float | None,
    sigma: np.ndarray | None,
) -> np.ndarray:
    predicted = models.evaluate(
        model_name,
        pressure,
        parameters,
        temperature_k=temperature_k,
        pressure_basis=pressure_basis,  # type: ignore[arg-type]
        saturation_pressure_pa=saturation_pressure_pa,
    )
    values = uptake - predicted
    return values / sigma if sigma is not None else values

###############################################################################
class ModelSolver:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.models = AdsorptionModels()

    # -------------------------------------------------------------------------
    def initial_configuration(
        self,
        spec: ModelSpec,
        pressure: np.ndarray,
        uptake: np.ndarray,
        temperature_k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positive_p = pressure[pressure > 0]
        p_min = float(np.min(positive_p)) if positive_p.size else 1.0
        p_median = float(np.median(positive_p)) if positive_p.size else 1.0
        p_max = float(np.max(positive_p)) if positive_p.size else 1.0
        q_max = max(float(np.max(uptake)), np.finfo(float).eps)
        q_median = max(float(np.median(uptake)), np.finfo(float).eps)

        initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for parameter in spec.parameters:
            if parameter.unit_kind == "pressure^-1":
                guess = 1.0 / p_median
                low = max(parameter.lower, 1.0 / (p_max * 1e6))
                high = min(parameter.upper, 1e3 / p_min)
                if spec.key == "temkin":
                    low = max(low, 1.0 / p_min)
                    guess = max(guess, low * 1.05)
            elif parameter.unit_kind == "uptake":
                guess = q_max * (
                    0.55
                    if "qsat" in parameter.name and spec.key == "dual_site_langmuir"
                    else 1.1
                )
                low = max(parameter.lower, q_max * 1e-8)
                high = min(parameter.upper, q_max * 100.0)
            elif parameter.unit_kind == "freundlich":
                guess = q_median / (p_median**1.0)
                low = max(parameter.lower, guess * 1e-6)
                high = min(parameter.upper, guess * 1e6)
            elif parameter.unit_kind == "energy^-2":
                span = (
                    UnitRegistry.GAS_CONSTANT_J_MOL_K
                    * temperature_k
                    * max(1.0, math.log(max(p_max / p_min, 1.0)))
                )
                guess = 1.0 / (span * span)
                low = parameter.lower
                high = min(parameter.upper, max(guess * 1e6, parameter.lower * 10))
            elif parameter.name == "beta" and spec.key == "redlich_peterson":
                guess, low, high = 0.8, 0.05, 1.0
            else:
                guess, low, high = 1.0, parameter.lower, parameter.upper
            if high <= low:
                high = low * 10.0
            initial.append(float(np.clip(guess, low, high)))
            lower.append(low)
            upper.append(high)

        if spec.key == "dual_site_langmuir":
            initial[0] = min(upper[0], max(lower[0], 10.0 / p_median))
            initial[2] = min(upper[2], max(lower[2], 0.1 / p_median))
        return np.asarray(initial), np.asarray(lower), np.asarray(upper)

    # -------------------------------------------------------------------------
    def fit(
        self,
        *,
        spec: ModelSpec,
        pressure: np.ndarray,
        uptake: np.ndarray,
        uncertainty: np.ndarray | None,
        temperature_k: float,
        pressure_basis: str,
        saturation_pressure_pa: float | None,
        optimizer: str,
        max_evaluations: int,
        overrides: dict[str, Any],
        display_pressure_unit: str,
        display_uptake_unit: str,
        molar_mass_g_mol: float | None,
        weighting: str = "unweighted",
    ) -> FitComputation:
        parameter_count = len(spec.parameters)
        if pressure.size <= parameter_count:
            return self.failure(
                spec,
                pressure,
                f"{spec.name} has {parameter_count} fitted parameters but only "
                f"{pressure.size} observations; n must be greater than k.",
            )

        try:
            initial, lower, upper = self.initial_configuration(
                spec, pressure, uptake, temperature_k
            )
            p_factor = pressure_factor(display_pressure_unit, pressure_basis)
            _, q_factor, _ = UnitRegistry.uptake_factor_to_mol_kg(
                display_uptake_unit, molar_mass_g_mol
            )
            related_override = overrides.get("n") or overrides.get("beta")
            related_value = (
                float(related_override.initial) if related_override is not None else 1.0
            )
            for index, parameter in enumerate(spec.parameters):
                override = overrides.get(parameter.name)
                if override is None:
                    continue
                lower[index] = parameter_from_display(
                    parameter,
                    override.lower,
                    pressure_factor_value=p_factor,
                    uptake_factor_value=q_factor,
                    related_value=related_value,
                )
                upper[index] = parameter_from_display(
                    parameter,
                    override.upper,
                    pressure_factor_value=p_factor,
                    uptake_factor_value=q_factor,
                    related_value=related_value,
                )
                initial[index] = parameter_from_display(
                    parameter,
                    override.initial,
                    pressure_factor_value=p_factor,
                    uptake_factor_value=q_factor,
                    related_value=related_value,
                )
            if (
                np.any(lower >= upper)
                or np.any(initial < lower)
                or np.any(initial > upper)
            ):
                raise ValueError(
                    f"{spec.name} parameter bounds or initial values are invalid."
                )

            sigma = (
                uncertainty
                if uncertainty is not None
                and uncertainty.size == pressure.size
                and np.all(np.isfinite(uncertainty))
                and np.all(uncertainty > 0)
                else None
            )
            if weighting == "inverse_sigma" and sigma is None:
                raise ValueError(
                    "Inverse-sigma weighting requires a complete positive uncertainty series."
                )

            starts = [initial]
            if parameter_count >= 3:
                starts.extend(
                    [
                        np.clip(initial * 0.5, lower, upper),
                        np.clip(initial * 2.0, lower, upper),
                    ]
                )
            candidates = []
            for start in starts:
                result = least_squares(
                    _fit_residual,
                    start,
                    args=(
                        self.models,
                        spec.key,
                        pressure,
                        uptake,
                        temperature_k,
                        pressure_basis,
                        saturation_pressure_pa,
                        sigma,
                    ),
                    bounds=(lower, upper),
                    method=optimizer,
                    loss="linear",
                    x_scale="jac",
                    max_nfev=max_evaluations,
                )
                predicted_candidate = self.models.evaluate(
                    spec.key,
                    pressure,
                    result.x,
                    temperature_k=temperature_k,
                    pressure_basis=pressure_basis,  # type: ignore[arg-type]
                    saturation_pressure_pa=saturation_pressure_pa,
                )
                candidate_sse = float(np.sum((uptake - predicted_candidate) ** 2))
                candidates.append((candidate_sse, result, predicted_candidate))
            _, result, predicted = min(candidates, key=lambda item: item[0])

            if not result.success:
                return self.failure(
                    spec,
                    pressure,
                    f"Optimizer did not converge: {result.message}",
                    function_evaluations=int(result.nfev),
                )
            optimal = np.asarray(result.x, dtype=np.float64)
            jacobian = np.asarray(result.jac, dtype=np.float64)
            rank = int(np.linalg.matrix_rank(jacobian))
            condition = (
                float(np.linalg.cond(jacobian))
                if jacobian.size and rank > 0
                else math.inf
            )
            warnings: list[str] = []
            covariance = None
            standard_errors = ci_low = ci_high = None
            dof = pressure.size - parameter_count
            if rank < parameter_count:
                warnings.append(
                    "Parameter uncertainty is unavailable because the Jacobian is rank deficient."
                )
            elif not math.isfinite(condition) or condition > CONDITION_WARNING:
                warnings.append(
                    "The fit is poorly conditioned; parameter estimates may not be identifiable."
                )
            else:
                if sigma is not None:
                    covariance = np.linalg.inv(jacobian.T @ jacobian)
                else:
                    residual_physical = uptake - predicted
                    if dof <= 0:
                        warnings.append(
                            "Parameter uncertainty is undefined because residual degrees of freedom are not positive."
                        )
                    else:
                        variance = float(np.sum(residual_physical**2)) / dof
                        covariance = np.linalg.inv(jacobian.T @ jacobian) * variance
                if covariance is None:
                    standard_errors = ci_low = ci_high = None
                    diagonal = np.array([-1.0])
                else:
                    diagonal = np.diag(covariance)
                if np.any(diagonal < 0) or not np.all(np.isfinite(diagonal)):
                    warnings.append(
                        "Parameter uncertainty is unavailable because covariance is invalid."
                    )
                    covariance = None
                else:
                    standard_errors = np.sqrt(diagonal)
                    critical = float(student_t.ppf(0.975, dof))
                    ci_low = optimal - critical * standard_errors
                    ci_high = optimal + critical * standard_errors

            if spec.key == "dual_site_langmuir" and optimal[0] < optimal[2]:
                permutation = np.array([2, 3, 0, 1])
                optimal = optimal[permutation]
                if standard_errors is not None:
                    standard_errors = standard_errors[permutation]
                    ci_low = ci_low[permutation] if ci_low is not None else None
                    ci_high = ci_high[permutation] if ci_high is not None else None
                predicted = self.models.evaluate(
                    spec.key,
                    pressure,
                    optimal,
                    temperature_k=temperature_k,
                    pressure_basis=pressure_basis,  # type: ignore[arg-type]
                    saturation_pressure_pa=saturation_pressure_pa,
                )

            metrics = compute_metrics(
                uptake,
                predicted,
                parameter_count,
                uncertainty=sigma,
                weighting=weighting,
            )
            curve_pressure = self.curve_grid(pressure, spec)
            curve_uptake = self.models.evaluate(
                spec.key,
                curve_pressure,
                optimal,
                temperature_k=temperature_k,
                pressure_basis=pressure_basis,  # type: ignore[arg-type]
                saturation_pressure_pa=saturation_pressure_pa,
            )
            warnings.extend(metrics.warnings)
            status = "warning" if warnings else "success"
            return FitComputation(
                spec=spec,
                status=status,
                message=str(result.message),
                parameters=optimal,
                standard_errors=standard_errors,
                ci95_low=ci_low,
                ci95_high=ci_high,
                metrics=metrics,
                predicted=predicted,
                curve_pressure=curve_pressure,
                curve_uptake=curve_uptake,
                function_evaluations=int(result.nfev),
                jacobian_rank=rank,
                condition_number=finite_or_none(condition),
                warnings=warnings,
            )
        except (ValueError, UnitConversionError, FloatingPointError) as exc:
            return self.failure(spec, pressure, str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def curve_grid(pressure: np.ndarray, spec: ModelSpec) -> np.ndarray:
        minimum = float(np.min(pressure))
        maximum = float(np.max(pressure))
        positive = pressure[pressure > 0]
        if maximum == minimum:
            return np.asarray([minimum])
        if positive.size and float(np.max(positive)) / float(np.min(positive)) >= 1000:
            grid = np.geomspace(
                float(np.min(positive)), float(np.max(positive)), CURVE_POINT_COUNT
            )
            if minimum == 0 and spec.key not in {"freundlich", "temkin"}:
                grid = np.insert(grid, 0, 0.0)
            return grid
        start = minimum
        if start == 0 and spec.key in {"freundlich", "temkin"}:
            start = float(np.min(positive))
        return np.linspace(start, maximum, CURVE_POINT_COUNT)

    # -------------------------------------------------------------------------
    @staticmethod
    def failure(
        spec: ModelSpec,
        pressure: np.ndarray,
        message: str,
        *,
        function_evaluations: int | None = None,
    ) -> FitComputation:
        return FitComputation(
            spec=spec,
            status="failed",
            message=message,
            parameters=None,
            standard_errors=None,
            ci95_low=None,
            ci95_high=None,
            metrics=MetricResult(
                values={
                    key: None
                    for key in (
                        "sse",
                        "rmse",
                        "mae",
                        "r_squared",
                        "adjusted_r_squared",
                        "chi_square",
                        "aic",
                        "aicc",
                        "bic",
                    )
                }
            ),
            predicted=np.asarray([], dtype=np.float64),
            curve_pressure=np.asarray([], dtype=np.float64),
            curve_uptake=np.asarray([], dtype=np.float64),
            function_evaluations=function_evaluations,
            jacobian_rank=None,
            condition_number=None,
            warnings=[message],
        )

###############################################################################
class FittingPipeline:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.solver = ModelSolver()

    # -------------------------------------------------------------------------
    @staticmethod
    def input_hash(series: dict[str, Any]) -> str:
        payload = {
            key: series[key]
            for key in (
                "dataset_id",
                "isotherm_id",
                "component_id",
                "observation_ids",
                "pressure",
                "uptake",
                "temperature_k",
                "pressure_basis",
                "saturation_pressure_pa",
            )
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def run(
        self, series: dict[str, Any], request: FittingRequest
    ) -> list[FitComputation]:
        pressure = np.asarray(series["pressure"], dtype=np.float64)
        uptake = np.asarray(series["uptake"], dtype=np.float64)
        if pressure.ndim != 1 or uptake.ndim != 1 or pressure.size != uptake.size:
            raise ValueError(
                "Canonical pressure and uptake arrays must be aligned one-dimensional arrays of equal length."
            )
        if pressure.size < 2:
            raise ValueError("At least two observations are required for fitting.")
        if np.any(~np.isfinite(pressure)) or np.any(pressure < 0):
            raise ValueError(
                "Canonical pressure values must be finite and non-negative."
            )
        if np.any(~np.isfinite(uptake)) or np.any(uptake < 0):
            raise ValueError("Canonical uptake values must be finite and non-negative.")
        uncertainty_values = series.get("uptake_stddev") or []
        uncertainty = (
            np.asarray(uncertainty_values, dtype=np.float64)
            if uncertainty_values
            and all(value is not None for value in uncertainty_values)
            else None
        )

        resolved_specs: list[ModelSpec] = []
        seen: set[str] = set()
        for model in request.models:
            spec = self.solver.models.get_spec(model)
            if spec.key not in seen:
                seen.add(spec.key)
                resolved_specs.append(spec)
        computations = [
            self.solver.fit(
                spec=spec,
                pressure=pressure,
                uptake=uptake,
                uncertainty=uncertainty,
                temperature_k=float(series["temperature_k"]),
                pressure_basis=series["pressure_basis"],
                saturation_pressure_pa=series.get("saturation_pressure_pa"),
                optimizer=request.optimizer,
                max_evaluations=request.max_evaluations,
                overrides=request.parameter_configuration.get(spec.key, {}),
                display_pressure_unit=request.display_units.pressure,
                display_uptake_unit=request.display_units.uptake,
                molar_mass_g_mol=series.get("adsorbate_molar_mass_g_mol"),
                weighting=request.weighting,
            )
            for spec in resolved_specs
        ]
        successful = [
            item
            for item in computations
            if item.status != "failed" and item.metrics.values["sse"] is not None
        ]
        successful.sort(
            key=lambda item: (
                item.metrics.values["aicc"] is None,
                item.metrics.values["aicc"]
                if item.metrics.values["aicc"] is not None
                else item.metrics.values["aic"]
                if item.metrics.values["aic"] is not None
                else item.metrics.values["sse"],
            )
        )
        for rank, item in enumerate(successful, start=1):
            item.rank = rank
        return computations

    # -------------------------------------------------------------------------
    def catalog(
        self,
        *,
        pressure_unit: str,
        uptake_unit: str,
        pressure_basis: str = "absolute",
        molar_mass_g_mol: float | None = None,
    ) -> list[ModelDefinition]:
        p_factor = pressure_factor(pressure_unit, pressure_basis)
        _, q_factor, _ = UnitRegistry.uptake_factor_to_mol_kg(
            uptake_unit, molar_mass_g_mol
        )
        definitions: list[ModelDefinition] = []
        for spec in self.solver.models.model_names:
            model = self.solver.models.get_spec(spec)
            definitions.append(
                ModelDefinition(
                    key=model.key,
                    name=model.name,
                    equation_latex=model.equation_latex,
                    assumptions=model.assumptions,
                    pressure_requirement=model.pressure_requirement,
                    requires_temperature=model.requires_temperature,
                    reference=model.reference,
                    parameters=[
                        ModelParameterDefinition(
                            name=parameter.name,
                            label=parameter.label,
                            lower=parameter_to_display(
                                parameter,
                                parameter.lower,
                                pressure_factor_value=p_factor,
                                uptake_factor_value=q_factor,
                                related_value=1.0,
                            ),
                            upper=parameter_to_display(
                                parameter,
                                parameter.upper,
                                pressure_factor_value=p_factor,
                                uptake_factor_value=q_factor,
                                related_value=1.0,
                            ),
                            initial=parameter_to_display(
                                parameter,
                                math.sqrt(parameter.lower * parameter.upper),
                                pressure_factor_value=p_factor,
                                uptake_factor_value=q_factor,
                                related_value=1.0,
                            ),
                            unit=parameter_unit(
                                parameter,
                                pressure_unit=pressure_unit,
                                uptake_unit=uptake_unit,
                                related_value=1.0,
                            ),
                        )
                        for parameter in model.parameters
                    ],
                )
            )
        return definitions

    # -------------------------------------------------------------------------
    @staticmethod
    def to_response_result(
        computation: FitComputation,
        *,
        pressure: np.ndarray,
        uptake: np.ndarray,
        pressure_basis: str,
        pressure_unit: str,
        uptake_unit: str,
        molar_mass_g_mol: float | None,
    ) -> ModelFitResult:
        p_factor = pressure_factor(pressure_unit, pressure_basis)
        _, q_factor, _ = UnitRegistry.uptake_factor_to_mol_kg(
            uptake_unit, molar_mass_g_mol
        )
        parameters: list[FittedParameter] = []
        if computation.parameters is not None:
            related = {
                parameter.name: computation.parameters[index]
                for index, parameter in enumerate(computation.spec.parameters)
            }
            related_value = related.get("n", related.get("beta", 1.0))
            for index, parameter in enumerate(computation.spec.parameters):
                parameters.append(
                    FittedParameter(
                        name=parameter.name,
                        label=parameter.label,
                        value=float(
                            display_parameter_value(
                                computation.parameters[index],
                                parameter,
                                pressure_factor_value=p_factor,
                                uptake_factor_value=q_factor,
                                related_value=float(related_value),
                            )
                        ),
                        standard_error=display_parameter_value(
                            computation.standard_errors[index]
                            if computation.standard_errors is not None
                            else None,
                            parameter,
                            pressure_factor_value=p_factor,
                            uptake_factor_value=q_factor,
                            related_value=float(related_value),
                        ),
                        ci95_low=display_parameter_value(
                            computation.ci95_low[index]
                            if computation.ci95_low is not None
                            else None,
                            parameter,
                            pressure_factor_value=p_factor,
                            uptake_factor_value=q_factor,
                            related_value=float(related_value),
                        ),
                        ci95_high=display_parameter_value(
                            computation.ci95_high[index]
                            if computation.ci95_high is not None
                            else None,
                            parameter,
                            pressure_factor_value=p_factor,
                            uptake_factor_value=q_factor,
                            related_value=float(related_value),
                        ),
                        unit=parameter_unit(
                            parameter,
                            pressure_unit=pressure_unit,
                            uptake_unit=uptake_unit,
                            related_value=float(related_value),
                        ),
                    )
                )

        observed_predictions: list[PredictionPoint] = []
        curve: list[PredictionPoint] = []
        if computation.status != "failed":
            observed_predictions = [
                PredictionPoint(
                    pressure=float(value / p_factor),
                    observed=float(observed / q_factor),
                    predicted=float(predicted / q_factor),
                    residual=float((observed - predicted) / q_factor),
                )
                for value, observed, predicted in zip(
                    pressure, uptake, computation.predicted, strict=True
                )
            ]
            curve = [
                PredictionPoint(
                    pressure=float(value / p_factor),
                    predicted=float(predicted / q_factor),
                )
                for value, predicted in zip(
                    computation.curve_pressure, computation.curve_uptake, strict=True
                )
            ]
        return ModelFitResult(
            model=computation.spec.key,
            name=computation.spec.name,
            status=computation.status,  # type: ignore[arg-type]
            convergence_message=computation.message,
            function_evaluations=computation.function_evaluations,
            jacobian_rank=computation.jacobian_rank,
            condition_number=computation.condition_number,
            parameters=parameters,
            metrics=FitMetrics(**computation.metrics.values),
            observed_predictions=observed_predictions,
            curve=curve,
            warnings=computation.warnings,
            rank=computation.rank,
        )
