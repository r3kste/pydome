from typing import Literal

from dataclasses import dataclass
from pydome.units import *  # noqa: F403


@dataclass(kw_only=True)
class Material:
    """Material"""

    name: str
    designation: Literal["steel", "nitralloy", "chrome"]
    H_B: float
    modulus_of_elasticity: Stress
    poissons_ratio: float


Steel = Material(
    name="Steel",
    designation="steel",
    H_B=250,
    modulus_of_elasticity=Stress(3e7, StressUnit.PSI),
    poissons_ratio=0.3,
)

Nitralloy = Material(
    name="Nitralloy",
    designation="nitralloy",
    H_B=135,
    modulus_of_elasticity=Stress(3e6, StressUnit.PSI),
    poissons_ratio=0.3,
)
