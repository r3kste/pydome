from typing import Literal

from dataclasses import dataclass
from pydome.units import *  # noqa: F403


@dataclass(kw_only=True)
class Material:
    """Material"""

    name: str
    H_B: float
    grade: int
    surface_finish: Literal["case", "nitrided"] | None
    modulus_of_elasticity: Stress
    poissons_ratio: float


Steel = Material(
    name="Steel",
    H_B=250,
    grade=1,
    surface_finish=None,
    modulus_of_elasticity=Stress(3e7, StressUnit.PSI),
    poissons_ratio=0.3
)

Nitralloy = Material(
    name="Nitralloy",
    H_B=135,
    grade=1,
    surface_finish="nitrided",
    modulus_of_elasticity=Stress(3e6, StressUnit.PSI),
    poissons_ratio=0.3
)