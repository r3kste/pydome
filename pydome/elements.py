from pydome.units import *  # noqa: F403
from pydome.materials import Material
from dataclasses import dataclass


@dataclass(kw_only=True)
class Gear:
    """Gear

    Attributes:
        N (int): Number of teeth
        n (AngularVelocity): Rotational speed
        T (Torque): Torque
        m (Length): Module
        P (InverseLength): Diametral pitch
    """

    n_teeth: int
    rpm: AngularVelocity
    diametral_pitch: InverseLength
    face_width: Length
    desired_cycles: int
    material: Material
    quality_number: int
    is_crowned: bool
    is_through_hardened: bool

    @property
    def H_B(self) -> float:
        """Brinell Hardness Number (BHN) of the gear material."""
        return self.material.H_B

    @property
    def grade(self) -> int:
        """AGMA quality grade of the gear material."""
        return self.material.grade

    @property
    def surface_finish(self) -> str:
        """Surface finish type of the gear material."""
        return self.material.surface_finish

    @property
    def pitch_diameter(self) -> Length:
        """Pitch diameter of the gear."""
        self.diametral_pitch.to(InverseLengthUnit.PER_INCH)
        diametral_pitch_mag = self.diametral_pitch.magnitude
        pitch_diameter_mag = self.n_teeth / diametral_pitch_mag
        return Length(magnitude=pitch_diameter_mag, unit=LengthUnit.INCH)
