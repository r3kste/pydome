import pytest

from pydome.units import Length, LengthUnit


def test_Length_unit_conversion():
    dist_mm = Length(1000.0, LengthUnit.MM)
    dist_inch = dist_mm.to(LengthUnit.INCH)
    assert pytest.approx(dist_inch.magnitude, 0.0001) == 39.3701

    dist_inch_2 = Length(39.3701, LengthUnit.INCH)
    dist_mm_2 = dist_inch_2.to(LengthUnit.MM)
    assert pytest.approx(dist_mm_2.magnitude, 0.0001) == 1000.0
