from typing import Any

from sympy import symbols
from sympy.physics.continuum_mechanics import Beam

from pydome.constants import pi
from pydome.units import *  # noqa: F403


class Shaft:
    def __init__(
        self,
        *,
        length: Length,
        diameter: Length,
        modulus_of_elasticity: Stress,
        density: Density | None,
    ) -> None:
        self.length = length
        self.diameter = diameter
        self.modulus_of_elasticity = modulus_of_elasticity
        self.density = density
        self.beam = self.create_beam()

    def create_beam(self) -> Beam:
        beam = Beam(
            length=self.length.to(LengthUnit.M).magnitude,
            elastic_modulus=self.modulus_of_elasticity.to(StressUnit.PA).magnitude,
            second_moment=(
                pi * (self.diameter.to(LengthUnit.M).magnitude / 2) ** 4 / 4
            ),
            area=(pi * (self.diameter.to(LengthUnit.M).magnitude / 2) ** 2),
        )
        return beam

    @property
    def cross_sectional_area(self) -> Area:
        area_mag = pi * (self.diameter.to(LengthUnit.M).magnitude / 2) ** 2
        return Area(area_mag, AreaUnit.M2)

    def apply_load(
        self,
        *,
        load: Any,
        order: int,
        start: Length,
        end: Length | None = None,
    ):
        """
        Apply a load to the beam.

        Args:
            load (Any): The load value corresponding to the order of the load
                or SymPy symbol.
            order (int): The order of the load.
                - -2: Moment
                - -1: Point Load
                - 0: Constant Uniform Load

        """
        if isinstance(load, Unit):
            load_mag = load.to(ForceUnit.NEWTON).magnitude
        else:
            load_mag = load
        start_mag = start.to(LengthUnit.M).magnitude
        end_mag = end.to(LengthUnit.M).magnitude if end is not None else None

        self.beam.apply_load(
            value=load_mag,
            order=order,
            start=start_mag,
            end=end_mag,
        )

    def solve_for_reaction_loads(self, *reactions: Any) -> None:
        self.beam.solve_for_reaction_loads(*reactions)

    def max_bmoment(self) -> Torque:
        pos, max_bmoment_mag = self.beam.max_bmoment()
        return Torque(max_bmoment_mag, TorqueUnit.Nm)


def demo():
    length = Length(80, LengthUnit.MM)
    diameter = Length(8, LengthUnit.MM)
    E = Stress(200e9, StressUnit.PA)
    density = Density(7850, DensityUnit.KG_PER_M3)

    shaft = Shaft(
        length=length,
        diameter=diameter,
        modulus_of_elasticity=E,
        density=density,
    )

    # Reaction Forces at Supports
    R1, R2 = symbols("R1 R2")
    pos1 = Length(10, LengthUnit.MM)
    pos2 = Length(70, LengthUnit.MM)
    shaft.apply_load(load=R1, order=-1, start=pos1)
    shaft.apply_load(load=R2, order=-1, start=pos2)

    # Loads
    weight_carrier = Force(0.5, ForceUnit.NEWTON)
    weight_chuck = Force(2.5, ForceUnit.NEWTON)
    pos_weight_carrier = Length(0, LengthUnit.MM)
    pos_weight_chuck = Length(80, LengthUnit.MM)
    shaft.apply_load(load=weight_carrier, order=-1, start=pos_weight_carrier)
    shaft.apply_load(load=weight_chuck, order=-1, start=pos_weight_chuck)

    # Self-weight
    if shaft.density is not None:
        g = Acceleration(9.80665, AccelerationUnit.MPS2)

        density_mag = shaft.density.to(DensityUnit.KG_PER_M3).magnitude
        area_mag = shaft.cross_sectional_area.to(AreaUnit.M2).magnitude
        g_mag = g.to(AccelerationUnit.MPS2).magnitude

        w_self_weight_mag = density_mag * area_mag * g_mag
        w_self_weight = Force(w_self_weight_mag, ForceUnit.NEWTON)
        shaft.apply_load(
            load=w_self_weight, order=0, start=Length(0, LengthUnit.MM), end=length
        )

    # Boundary Conditions

    # deflection at supports is zero
    shaft.beam.bc_deflection = [
        (pos1.to(LengthUnit.M).magnitude, 0),
        (pos2.to(LengthUnit.M).magnitude, 0),
    ]

    shaft.solve_for_reaction_loads(R1, R2)

    shaft.beam.plot_shear_force()
    shaft.beam.plot_bending_moment()
    print("Max Bending Moment:", shaft.max_bmoment())
