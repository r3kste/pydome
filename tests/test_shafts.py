from pydome.units import *
from pydome.shafts import Shaft
from pydome.utilities import approx


def test_example_3_7():
    shaft = Shaft(
        length=Length(0.3048, LengthUnit.M),
        diameter=Length(10, LengthUnit.MM),
        modulus_of_elasticity=Stress(200e9, StressUnit.PA),
        density=None,
    )

    shaft.apply_load(
        load=Force(-1628.25, ForceUnit.NEWTON),
        order=-1,
        start=Length(0, LengthUnit.MM),
    )
    shaft.apply_load(
        load=Force(2171, ForceUnit.NEWTON),
        order=-1,
        start=Length(76.2, LengthUnit.MM),
    )
    shaft.apply_load(
        load=Force(-542.75, ForceUnit.NEWTON),
        order=-1,
        start=Length(304.8, LengthUnit.MM),
    )

    max_bmoment = shaft.max_bmoment()
    assert max_bmoment.equals(Torque(12405.7e-2, TorqueUnit.Nm))
