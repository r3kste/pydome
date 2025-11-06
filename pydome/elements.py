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
    diametral_pitch: InverseLength | None = None
    pitch_diameter: Length | None = None
    face_width: Length
    desired_cycles: int
    material: Material
    quality_number: int
    is_crowned: bool
    pressure_angle: Angle

    grade: int
    heat_treatment: str

    @property
    def H_B(self) -> float:
        """Brinell Hardness Number (BHN) of the gear material."""
        return self.material.H_B

    @property
    def get_pitch_diameter(self) -> Length:
        """Pitch diameter of the gear."""
        if self.pitch_diameter is not None:
            return self.pitch_diameter

        self.diametral_pitch.to(InverseLengthUnit.PER_INCH)
        diametral_pitch_mag = self.diametral_pitch.magnitude
        pitch_diameter_mag = self.n_teeth / diametral_pitch_mag
        return Length(pitch_diameter_mag, LengthUnit.INCH)

    @property
    def get_diametral_pitch(self) -> Length:
        """Diametral Pitch"""
        if self.diametral_pitch is not None:
            return self.diametral_pitch

        self.pitch_diameter.to(LengthUnit.INCH)
        pitch_diameter_mag = self.pitch_diameter.magnitude
        diametral_pitch_mag = self.n_teeth / pitch_diameter_mag
        return InverseLength(diametral_pitch_mag, InverseLengthUnit.PER_INCH)
