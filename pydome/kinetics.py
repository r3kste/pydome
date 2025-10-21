from pydome.units import *  # noqa: F403


def force(
    *,
    m: Mass | None = None,
    alpha: Acceleration | None = None,
    T: Torque | None = None,
    r: Length | None = None,
) -> Force:
    """Centrifugal force"""

    if m is not None and alpha is not None:
        m_mag = m.to(MassUnit.KG).magnitude
        a_mag = alpha.to(AccelerationUnit.MPS2).magnitude
        F_mag = m_mag * a_mag
    elif T is not None and r is not None:
        T_mag = T.to(TorqueUnit.Nm).magnitude
        r_mag = r.to(LengthUnit.M).magnitude
        F_mag = T_mag / r_mag

    return Force(F_mag, ForceUnit.NEWTON)


def power(
    *,
    F: Force | None = None,
    V: Velocity | None = None,
    T: Torque | None = None,
    n: AngularVelocity | None = None,
) -> Power:
    """Power calculation"""

    if F is not None and V is not None:
        F_mag = F.to(ForceUnit.NEWTON).magnitude
        V_mag = V.to(VelocityUnit.MPS).magnitude
        P_mag = F_mag * V_mag
    elif T is not None and n is not None:
        T_mag = T.to(TorqueUnit.Nm).magnitude
        n_mag = n.to(AngularVelocityUnit.RAD_per_SEC).magnitude
        P_mag = T_mag * n_mag

    return Power(P_mag, PowerUnit.WATT)


def torque(
    *,
    P: Power | None = None,
    n: AngularVelocity | None = None,
    F: Force | None = None,
    r: Length | None = None,
) -> Torque:
    """Torque calculation"""

    if P is not None and n is not None:
        P_mag = P.to(PowerUnit.WATT).magnitude
        n_mag = n.to(AngularVelocityUnit.RAD_per_SEC).magnitude
        T_mag = P_mag / n_mag
    elif F is not None and r is not None:
        F_mag = F.to(ForceUnit.NEWTON).magnitude
        r_mag = r.to(LengthUnit.M).magnitude
        T_mag = F_mag * r_mag

    return Torque(T_mag, TorqueUnit.Nm)
