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


Steel = Material(
    name="Steel",
    H_B=250,
    grade=1,
    surface_finish=None,
)
