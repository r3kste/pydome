from pydome.units import *  # noqa: F403


def alpha(
    *,
    w: AngularVelocity,
    r: Length,
) -> Acceleration:
    """Angular acceleration"""

    w_mag = w.to(AngularVelocityUnit.RAD_per_SEC).magnitude
    r_mag = r.to(LengthUnit.M).magnitude
    a_mag = w_mag**2 * r_mag

    return Acceleration(a_mag, AccelerationUnit.MPS2)

