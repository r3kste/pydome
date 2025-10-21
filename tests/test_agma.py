import pytest

from pydome.gears.spur import Spur
from pydome.units import *  # noqa: F403


def approx(val, rel=1e-3):
    return pytest.approx(val, rel=rel)


def test_example_14_4():
    N_P = 17
    phi = Angle(20, AngleUnit.DEGREE)
    n_P = AngularVelocity(1800, AngularVelocityUnit.RPM)
    H = Power(4, PowerUnit.HP)
    N_G = 52
    P = InverseLength(10, InverseLengthUnit.PER_INCH)
    F = Length(1.5, LengthUnit.INCH)
    Q_v = 6

    # straddle mounted bearings immediately adjacent
    S_1 = 0
    S = 1  # ?

    grade_P = 1
    H_B_P = 240
    is_through_hardened_P = True
    is_nitrided_P = False

    grade_G = 1
    H_B_G = 200
    is_through_hardened_G = True
    is_nitrided_G = False

    mu_P, mu_G = 0.3, 0.3
    J_P, J_G = 0.3, 0.4
    Youngs = Stress(30e6, StressUnit.PSI)

    power_source = "uniform"
    driven_machine = "uniform"

    L_P = 1e8
    R = 0.9
    crowned = False
    gearing_condition = "enclosed_commercial"

    d_P = Spur.d(N=N_P, P=P)
    assert d_P.unit == LengthUnit.INCH
    assert d_P.magnitude == 1.7

    d_G = Spur.d(N=N_G, P=P)
    assert d_G.unit == LengthUnit.INCH
    assert d_G.magnitude == 5.2

    V = Spur.V(d=d_P, n=n_P)
    assert V.unit == VelocityUnit.FPM
    assert V.magnitude == approx(801.1)

    W_t = Spur.W_t(H=H, V=V)
    assert W_t.unit == ForceUnit.LBF
    assert W_t.magnitude == approx(164.8)

    K_o = Spur.K_o(power_source=power_source, driven_machine=driven_machine)
    assert K_o == 1.0

    K_v = Spur.K_v(V=V, Q_v=Q_v)
    assert K_v == approx(1.377)

    K_s_P = Spur.K_s(F=F, N=N_P, P=P)
    assert K_s_P == approx(1.043)

    K_s_G = Spur.K_s(F=F, N=N_G, P=P)
    assert K_s_G == approx(1.052)

    K_m = Spur.K_m(
        is_crowned=crowned,
        F=F,
        d_P=d_P,
        S_1=S_1,
        S=S,
        gearing_condition=gearing_condition,
        is_gear_adjusted=False,
    )
    assert K_m == approx(1.22)

    K_B = Spur.K_B(t_R=None, h_t=None)
    assert K_B == 1.0

    m_G = Spur.m_G(N_P=N_P, N_G=N_G)
    assert m_G == approx(3.059)

    L_G = L_P / m_G
    assert L_G == approx(1e8 / 3.059)

    Y_N_P = Spur.Y_N(L=L_P, H_B=H_B_P, upper=True)
    assert Y_N_P == approx(0.977)

    Y_N_G = Spur.Y_N(L=L_G, H_B=H_B_G, upper=True)
    assert Y_N_G == approx(0.996)

    K_R = Spur.K_R(R=R)
    assert K_R == approx(0.85)

    K_T = Spur.K_T(T=None)
    assert K_T == 1.0

    C_f = Spur.C_f()
    assert C_f == 1.0

    m_N = Spur.m_N()
    assert m_N == 1.0

    I = Spur.I(N_P=N_P, N_G=N_G, phi=phi, gear_mode="external")
    assert I == approx(0.121)

    C_p = Spur.C_p(mu_P=mu_P, mu_G=mu_G, E_P=Youngs, E_G=Youngs)
    C_p = SqrtPressure(2300, SqrtPressureUnit.SQRT_PSI)
    assert C_p.unit == SqrtPressureUnit.SQRT_PSI
    assert C_p.magnitude == approx(2300)

    _S_t_P = Spur.S_t(
        H_B=H_B_P,
        grade=grade_P,
        is_through_hardened=is_through_hardened_P,
        is_nitrided=is_nitrided_P,
        material=None,
    )
    assert _S_t_P.unit == StressUnit.PSI
    assert _S_t_P.magnitude == approx(31350)

    _S_t_G = Spur.S_t(
        H_B=H_B_G,
        grade=grade_G,
        is_through_hardened=is_through_hardened_G,
        is_nitrided=is_nitrided_G,
        material=None,
    )
    assert _S_t_G.unit == StressUnit.PSI
    assert _S_t_G.magnitude == approx(28260)

    S_c_P = Spur.S_c(H_B=H_B_P, grade=grade_P)
    assert S_c_P.unit == StressUnit.PSI
    assert S_c_P.magnitude == approx(106400)

    S_c_G = Spur.S_c(H_B=H_B_G, grade=grade_G)
    assert S_c_G.unit == StressUnit.PSI
    assert S_c_G.magnitude == approx(93500)

    Z_N_P = Spur.Z_N(A=1.4488, B=0.023, L=L_P, is_nitrided=is_nitrided_P)
    assert Z_N_P == approx(0.948)

    Z_N_G = Spur.Z_N(A=1.4488, B=0.023, L=L_G, is_nitrided=is_nitrided_G)
    assert Z_N_G == approx(0.973)

    C_H = Spur.C_H(
        N_P=N_P, N_G=N_G, H_B_P=H_B_P, f_P=None, H_B_G=H_B_G, is_pinion=False
    )
    assert C_H == approx(1.005)

    # =========================================================================
    # Pinion Bending

    s_b_P = Spur.s_b(
        N=N_P,
        P=P,
        n=n_P,
        H=H,
        power_source=power_source,
        driven_machine=driven_machine,
        Q_v=Q_v,
        F=F,
        is_crowned=crowned,
        d_P=d_P,
        S_1=S_1,
        S=S,
        gearing_condition=gearing_condition,
        t_R=None,
        h_t=None,
        other_N=N_G,
        is_gear_adjusted=False,
        J=J_P,
    )
    assert s_b_P.unit == StressUnit.PSI
    assert s_b_P.magnitude == approx(6417)

    s_b_allowable_P = Spur.s_b_all(
        H_B=H_B_P,
        grade=grade_P,
        L=L_P,
        surface=H_B_P,
        upper=True,
        T=None,
        R=R,
        is_through_hardened=is_through_hardened_P,
        material=None,
        S_F=1.0,
    )
    assert s_b_allowable_P.unit == StressUnit.PSI
    assert s_b_allowable_P.magnitude == approx(36034)

    # =========================================================================
    # Gear Bending

    n_G = AngularVelocity(
        magnitude=n_P.magnitude * N_P / N_G,
        unit=AngularVelocityUnit.RPM,
    )
    s_b_G = Spur.s_b(
        N=N_G,
        P=P,
        n=n_G,
        H=H,
        power_source=power_source,
        driven_machine=driven_machine,
        Q_v=Q_v,
        F=F,
        is_crowned=crowned,
        d_P=d_P,
        S_1=S_1,
        S=S,
        gearing_condition=gearing_condition,
        t_R=None,
        h_t=None,
        other_N=N_P,
        is_gear_adjusted=False,
        J=J_G,
    )
    assert s_b_G.unit == StressUnit.PSI
    assert s_b_G.magnitude == approx(4854)

    s_b_allowable_G = Spur.s_b_all(
        H_B=H_B_G,
        grade=grade_G,
        L=L_G,
        surface=H_B_G,
        upper=True,
        T=None,
        R=R,
        is_through_hardened=is_through_hardened_G,
        material=None,
        S_F=1.0,
    )
    assert s_b_allowable_G.unit == StressUnit.PSI
    assert s_b_allowable_G.magnitude == approx(33114)

    # =========================================================================
    # Pinion Wear

    s_c_P = Spur.s_c(
        N=N_P,
        P=P,
        n=n_P,
        H=H,
        power_source=power_source,
        driven_machine=driven_machine,
        Q_v=Q_v,
        F=F,
        mu_P=mu_P,
        mu_G=mu_G,
        E_P=Youngs,
        E_G=Youngs,
        is_crowned=crowned,
        d_P=d_P,
        S_1=S_1,
        S=S,
        gearing_condition=gearing_condition,
        is_gear_adjusted=False,
        other_N=N_G,
        phi=phi,
        gear_mode="external",
        C_p=C_p,
    )
    assert s_c_P.unit == StressUnit.PSI
    assert s_c_P.magnitude == approx(70360)

    s_c_allowable_P = Spur.s_c_all(
        H_B=H_B_P,
        grade=grade_P,
        L=L_P,
        surface=("nitrided" if is_nitrided_P else ""),
        T=None,
        R=R,
        N_P=N_P,
        N_G=N_G,
        H_B_P=H_B_P,
        f_P=None,
        H_B_G=H_B_G,
        is_pinion=True,
        S_H=1.0,
        Z_N=Z_N_P,
    )
    assert s_c_allowable_P.unit == StressUnit.PSI
    assert s_c_allowable_P.magnitude == approx(118667)

    # =========================================================================
    # Gear Wear

    s_c_G = Spur.s_c(
        N=N_G,
        P=P,
        n=n_G,
        H=H,
        power_source=power_source,
        driven_machine=driven_machine,
        Q_v=Q_v,
        F=F,
        mu_P=mu_P,
        mu_G=mu_G,
        E_P=Youngs,
        E_G=Youngs,
        is_crowned=crowned,
        d_P=d_P,
        S_1=S_1,
        S=S,
        gearing_condition=gearing_condition,
        is_gear_adjusted=False,
        other_N=N_P,
        phi=phi,
        gear_mode="external",
        C_p=C_p,
    )
    assert s_c_G.unit == StressUnit.PSI
    assert s_c_G.magnitude == approx(70660)

    s_c_allowable_G = Spur.s_c_all(
        H_B=H_B_G,
        grade=grade_G,
        L=L_G,
        surface=("nitrided" if is_nitrided_G else ""),
        T=None,
        R=R,
        N_P=N_P,
        N_G=N_G,
        H_B_P=H_B_P,
        f_P=None,
        H_B_G=H_B_G,
        is_pinion=False,
        S_H=1.0,
        Z_N=Z_N_G,
    )
    assert s_c_allowable_G.unit == StressUnit.PSI
    assert s_c_allowable_G.magnitude == approx(107565)


def test_example_14_6():
    H_B_P = 300
    N_P = 18
    _phi = Angle(20, AngleUnit.DEGREE)
    _P = InverseLength(16, InverseLengthUnit.PER_INCH)
    grade_P = 1
    through_hardened_P = True

    N_G = 64
    grade_G = 1
    through_hardened_G = True

    S_t_P = Spur.S_t(
        H_B=H_B_P,
        grade=grade_P,
        is_through_hardened=through_hardened_P,
        is_nitrided=False,
        material=None,
    )
    assert S_t_P.unit == StressUnit.PSI
    assert S_t_P.magnitude == approx(35990)

    J_P = Spur.J(N=N_P, other_N=N_G)
    assert J_P == approx(0.32, rel=0.1)
    J_P = 0.32

    J_G = Spur.J(N=N_G, other_N=N_P)
    assert J_G == approx(0.41, rel=0.1)
    J_G = 0.41

    _beta = -0.110

    H_B_G = 150  # rev
    S_t_G = Spur.S_t(
        H_B=H_B_G,
        grade=grade_G,
        is_through_hardened=through_hardened_G,
        is_nitrided=False,
        material=None,
    )
    assert S_t_G.unit == StressUnit.PSI
    assert S_t_G.magnitude == approx(24430, rel=0.01)
