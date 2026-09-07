from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from adsmod_common.units import UnitRegistry


PressureBasis = Literal["absolute", "partial", "relative"]
ModelFunction = Callable[..., np.ndarray]

###############################################################################
@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    lower: float
    upper: float
    unit_kind: str

###############################################################################
@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    equation_latex: str
    parameters: tuple[ParameterSpec, ...]
    assumptions: str
    pressure_requirement: str
    requires_temperature: bool
    reference: str
    function: ModelFunction

###############################################################################
def langmuir(pressure: np.ndarray, k: float, qsat: float) -> np.ndarray:
    kp = k * pressure
    return qsat * kp / (1.0 + kp)

###############################################################################
def sips(pressure: np.ndarray, k: float, qsat: float, n: float) -> np.ndarray:
    kp_n = np.power(k * pressure, n)
    return qsat * kp_n / (1.0 + kp_n)

###############################################################################
def freundlich(pressure: np.ndarray, k: float, n: float) -> np.ndarray:
    return k * np.power(pressure, 1.0 / n)

###############################################################################
def temkin(pressure: np.ndarray, k: float, beta: float) -> np.ndarray:
    return beta * np.log(k * pressure)

###############################################################################
def toth(pressure: np.ndarray, k: float, qsat: float, t: float) -> np.ndarray:
    kp = k * pressure
    return qsat * kp / np.power(1.0 + np.power(kp, t), 1.0 / t)

###############################################################################
def dubinin_radushkevich(
    relative_pressure: np.ndarray,
    qsat: float,
    beta: float,
    *,
    temperature_k: float,
) -> np.ndarray:
    relative = np.asarray(relative_pressure, dtype=np.float64)
    output = np.zeros_like(relative)
    positive = relative > 0
    potential = (
        UnitRegistry.GAS_CONSTANT_J_MOL_K
        * temperature_k
        * np.log(1.0 / relative[positive])
    )
    output[positive] = qsat * np.exp(-beta * potential * potential)
    return output

###############################################################################
def dual_site_langmuir(
    pressure: np.ndarray,
    k1: float,
    qsat1: float,
    k2: float,
    qsat2: float,
) -> np.ndarray:
    return langmuir(pressure, k1, qsat1) + langmuir(pressure, k2, qsat2)

###############################################################################
def redlich_peterson(
    pressure: np.ndarray, k: float, a: float, beta: float
) -> np.ndarray:
    return k * pressure / (1.0 + a * np.power(pressure, beta))

###############################################################################
def jovanovic(pressure: np.ndarray, k: float, qsat: float) -> np.ndarray:
    return qsat * (1.0 - np.exp(-k * pressure))

###############################################################################
def _affinity(name: str = "k") -> ParameterSpec:
    return ParameterSpec(name, "Affinity coefficient", 1e-16, 1.0, "pressure^-1")

###############################################################################
def _capacity(name: str = "qsat", label: str = "Saturation capacity") -> ParameterSpec:
    return ParameterSpec(name, label, 1e-16, 1e6, "uptake")


MODEL_SPECS: dict[str, ModelSpec] = {
    "langmuir": ModelSpec(
        key="langmuir",
        name="Langmuir",
        equation_latex=r"q = q_{sat}\frac{Kp}{1+Kp}",
        parameters=(_affinity(), _capacity()),
        assumptions="Single-component monolayer adsorption on energetically equivalent sites.",
        pressure_requirement="A consistent equilibrium pressure scale; K carries the reciprocal pressure unit.",
        requires_temperature=False,
        reference="NIST SRD-205 User Guide, Eq. 1; Langmuir, JACS 1918, DOI 10.1021/ja02242a004.",
        function=langmuir,
    ),
    "sips": ModelSpec(
        key="sips",
        name="Sips (Langmuir-Freundlich)",
        equation_latex=r"q = q_{sat}\frac{(Kp)^n}{1+(Kp)^n}",
        parameters=(
            _affinity(),
            _capacity(),
            ParameterSpec("n", "Heterogeneity exponent", 0.05, 10.0, "dimensionless"),
        ),
        assumptions="Empirical single-component heterogeneous-site saturation model.",
        pressure_requirement="A consistent equilibrium pressure scale; K carries the reciprocal pressure unit.",
        requires_temperature=False,
        reference="NIST SRD-205 User Guide, Eq. 3; Sips, J. Chem. Phys. 1948, DOI 10.1063/1.1746922.",
        function=sips,
    ),
    "freundlich": ModelSpec(
        key="freundlich",
        name="Freundlich",
        equation_latex=r"q = K_F p^{1/n}",
        parameters=(
            ParameterSpec("k", "Freundlich coefficient", 1e-16, 1e6, "freundlich"),
            ParameterSpec("n", "Freundlich exponent", 0.05, 10.0, "dimensionless"),
        ),
        assumptions="Empirical heterogeneous-surface model without a saturation limit.",
        pressure_requirement="Strictly positive equilibrium pressure for fitting and prediction.",
        requires_temperature=False,
        reference="NIST SRD-205 User Guide, Eq. 2; IUPAC Gold Book term 14702.",
        function=freundlich,
    ),
    "temkin": ModelSpec(
        key="temkin",
        name="Temkin",
        equation_latex=r"q = B\ln(Kp)",
        parameters=(
            _affinity(),
            ParameterSpec("beta", "Temkin loading coefficient", 1e-16, 1e6, "uptake"),
        ),
        assumptions="Heat of adsorption decreases linearly with surface coverage.",
        pressure_requirement="Strictly positive pressure and Kp > 0; negative predicted loading is invalid.",
        requires_temperature=False,
        reference="Temkin and Pyzhev, Acta Physicochim. URSS 1940, 12, 327-356.",
        function=temkin,
    ),
    "toth": ModelSpec(
        key="toth",
        name="Toth",
        equation_latex=r"q = \frac{q_{sat}Kp}{[1+(Kp)^t]^{1/t}}",
        parameters=(
            _affinity(),
            _capacity(),
            ParameterSpec(
                "t", "Toth heterogeneity parameter", 0.05, 10.0, "dimensionless"
            ),
        ),
        assumptions="Single-component heterogeneous-site saturation model with Langmuir limit t=1.",
        pressure_requirement="A consistent non-negative equilibrium pressure scale.",
        requires_temperature=False,
        reference="Toth, Acta Chim. Acad. Sci. Hung. 1971, 69, 311-328.",
        function=toth,
    ),
    "dubinin_radushkevich": ModelSpec(
        key="dubinin_radushkevich",
        name="Dubinin-Radushkevich",
        equation_latex=r"q = q_{sat}\exp[-\beta(RT\ln(p_0/p))^2]",
        parameters=(
            _capacity(),
            ParameterSpec(
                "beta",
                "Adsorption-energy coefficient",
                1e-20,
                1.0,
                "energy^-2",
            ),
        ),
        assumptions="Micropore filling for condensable vapours using Polanyi adsorption potential.",
        pressure_requirement="Relative pressure 0 <= p/p0 <= 1, or dimensional pressure with a positive saturation pressure p0.",
        requires_temperature=True,
        reference="Dubinin-Radushkevich equation; Carbon 2001, DOI 10.1016/S0008-6223(00)00265-7.",
        function=dubinin_radushkevich,
    ),
    "dual_site_langmuir": ModelSpec(
        key="dual_site_langmuir",
        name="Dual-Site Langmuir",
        equation_latex=(
            r"q = q_{1,sat}\frac{K_1p}{1+K_1p}"
            r"+q_{2,sat}\frac{K_2p}{1+K_2p}"
        ),
        parameters=(
            _affinity("k1"),
            _capacity("qsat1", "Site 1 capacity"),
            _affinity("k2"),
            _capacity("qsat2", "Site 2 capacity"),
        ),
        assumptions="Two independent Langmuir site families; parameter labels are canonicalized by affinity.",
        pressure_requirement="A consistent non-negative equilibrium pressure scale.",
        requires_temperature=False,
        reference="Dual-site extension of the Langmuir single-component isotherm.",
        function=dual_site_langmuir,
    ),
    "redlich_peterson": ModelSpec(
        key="redlich_peterson",
        name="Redlich-Peterson",
        equation_latex=r"q = \frac{K_Rp}{1+a_Rp^\beta}",
        parameters=(
            ParameterSpec(
                "k", "Redlich-Peterson coefficient", 1e-16, 1e6, "uptake/pressure"
            ),
            ParameterSpec("a", "Denominator coefficient", 1e-16, 1e6, "pressure^-beta"),
            ParameterSpec(
                "beta", "Redlich-Peterson exponent", 0.05, 1.0, "dimensionless"
            ),
        ),
        assumptions="Empirical three-parameter interpolation between Langmuir and Freundlich behaviour.",
        pressure_requirement="A consistent non-negative equilibrium pressure scale.",
        requires_temperature=False,
        reference="Redlich and Peterson, J. Phys. Chem. 1959, DOI 10.1021/j150576a611.",
        function=redlich_peterson,
    ),
    "jovanovic": ModelSpec(
        key="jovanovic",
        name="Jovanovic",
        equation_latex=r"q = q_{sat}[1-\exp(-Kp)]",
        parameters=(_affinity(), _capacity()),
        assumptions="Single-component monolayer adsorption with an exponential saturation approach.",
        pressure_requirement="A consistent non-negative equilibrium pressure scale.",
        requires_temperature=False,
        reference="Jovanovic, Kolloid-Z. Z. Polym. 1969, 235, 1203-1213.",
        function=jovanovic,
    ),
}

###############################################################################
class AdsorptionModels:
    model_names = tuple(MODEL_SPECS)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_spec(model_name: str) -> ModelSpec:
        key = model_name.strip().casefold().replace("-", "_").replace(" ", "_")
        if key == "sips_(langmuir_freundlich)":
            key = "sips"
        try:
            return MODEL_SPECS[key]
        except KeyError as exc:
            raise ValueError(f"Model '{model_name}' is not supported.") from exc

    # -------------------------------------------------------------------------
    @classmethod
    def evaluate(
        cls,
        model_name: str,
        pressure: np.ndarray,
        parameters: np.ndarray | list[float],
        *,
        temperature_k: float,
        pressure_basis: PressureBasis,
        saturation_pressure_pa: float | None,
    ) -> np.ndarray:
        spec = cls.get_spec(model_name)
        p = np.asarray(pressure, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        if params.shape != (len(spec.parameters),):
            raise ValueError(f"{spec.name} requires {len(spec.parameters)} parameters.")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("Pressure values must be finite and non-negative.")
        if spec.key in {"freundlich", "temkin"} and np.any(p <= 0):
            raise ValueError(f"{spec.name} requires strictly positive pressure.")

        if spec.key == "dubinin_radushkevich":
            if pressure_basis == "relative":
                relative = p
            else:
                if saturation_pressure_pa is None or saturation_pressure_pa <= 0:
                    raise ValueError(
                        "Dubinin-Radushkevich fitting requires saturation pressure p0."
                    )
                relative = p / saturation_pressure_pa
            if np.any(relative < 0) or np.any(relative > 1):
                raise ValueError(
                    "Dubinin-Radushkevich requires relative pressure between 0 and 1."
                )
            predicted = dubinin_radushkevich(
                relative,
                *params,
                temperature_k=temperature_k,
            )
        else:
            predicted = spec.function(p, *params)

        if spec.key == "temkin" and np.any(predicted < 0):
            raise ValueError(
                "Temkin parameters produce negative uptake within the fitted domain."
            )
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"{spec.name} produced non-finite predictions.")
        return np.asarray(predicted, dtype=np.float64)
