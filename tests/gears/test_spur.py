from pydome.gears.spur import Spur
from pydome.materials import *  # noqa: F403
from pydome.solver import solve_for_parameters
from pydome.units import *  # noqa: F403
from pydome.utilities import approx


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

    lewis_form_factor_P = Spur.Y(N=N_P)
    K_s_P = Spur.K_s(F=F, Y=lewis_form_factor_P, P=P)
    assert K_s_P == approx(1.043)

    lewis_form_factor_G = Spur.Y(N=N_G)
    K_s_G = Spur.K_s(F=F, Y=lewis_form_factor_G, P=P)
    assert K_s_G == approx(1.052)

    C_mc = Spur.C_mc(is_crowned=crowned)
    C_pf = Spur.C_pf(F=F, d_P=d_P)
    C_pm = Spur.C_pm(S_1=S_1, S=S)
    C_ma = Spur.C_ma(F=F, gearing_condition=gearing_condition)
    C_e = Spur.C_e(is_gear_adjusted=False)

    K_m = Spur.K_m(C_mc=C_mc, C_pf=C_pf, C_pm=C_pm, C_ma=C_ma, C_e=C_e)
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

    I = Spur.I(m_G=m_G, m_N=m_N, phi=phi, gear_mode="external")
    assert I == approx(0.121)

    C_p = Spur.C_p(mu_P=mu_P, mu_G=mu_G, E_P=Youngs, E_G=Youngs)
    C_p = SqrtPressure(2300, SqrtPressureUnit.SQRT_PSI)
    assert C_p.unit == SqrtPressureUnit.SQRT_PSI
    assert C_p.magnitude == approx(2300)

    S_t_P = Spur.S_t(
        H_B=H_B_P,
        grade=grade_P,
        is_through_hardened=is_through_hardened_P,
        is_nitrided=is_nitrided_P,
        material=None,
    )
    assert S_t_P.unit == StressUnit.PSI
    assert S_t_P.magnitude == approx(31350)

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

    C_H = Spur.C_H(m_G=m_G, H_B_P=H_B_P, f_P=None, H_B_G=H_B_G, is_pinion=False)
    assert C_H == approx(1.005)

    # =========================================================================
    # Pinion Bending

    s_b_P = Spur.s_b(
        W_t=W_t, K_o=K_o, K_v=K_v, K_s=K_s_P, K_m=K_m, K_B=K_B, J=J_P, P=P, F=F
    )
    assert s_b_P.unit == StressUnit.PSI
    assert s_b_P.magnitude == approx(6417)

    s_b_allowable_P = Spur.s_b_all(S_t=S_t_P, Y_N=Y_N_P, K_R=K_R, K_T=K_T, S_F=1.0)
    assert s_b_allowable_P.unit == StressUnit.PSI
    assert s_b_allowable_P.magnitude == approx(36034)

    # =========================================================================
    # Gear Bending

    n_G = AngularVelocity(
        n_P.magnitude * N_P / N_G,
        AngularVelocityUnit.RPM,
    )
    s_b_G = Spur.s_b(
        W_t=W_t, K_o=K_o, K_v=K_v, K_s=K_s_G, K_m=K_m, K_B=K_B, J=J_G, P=P, F=F
    )
    assert s_b_G.unit == StressUnit.PSI
    assert s_b_G.magnitude == approx(4854)

    s_b_allowable_G = Spur.s_b_all(S_t=_S_t_G, Y_N=Y_N_G, K_R=K_R, K_T=K_T, S_F=1.0)
    assert s_b_allowable_G.unit == StressUnit.PSI
    assert s_b_allowable_G.magnitude == approx(33114)

    # =========================================================================
    # Pinion Wear

    s_c_P = Spur.s_c(
        C_p=C_p,
        W_t=W_t,
        K_o=K_o,
        K_v=K_v,
        K_s=K_s_P,
        K_m=K_m,
        C_f=C_f,
        d_P=d_P,
        I=I,
        F=F,
    )
    assert s_c_P.unit == StressUnit.PSI
    assert s_c_P.magnitude == approx(70360)

    s_c_allowable_P = Spur.s_c_all(
        S_c=S_c_P, Z_N=Z_N_P, C_H=C_H, K_T=K_T, K_R=K_R, S_H=1.0
    )
    assert s_c_allowable_P.equals(Stress(118667, StressUnit.PSI), rel=1e-2)

    # =========================================================================
    # Gear Wear

    s_c_G = Spur.s_c(
        C_p=C_p,
        W_t=W_t,
        K_o=K_o,
        K_v=K_v,
        K_s=K_s_G,
        K_m=K_m,
        C_f=C_f,
        d_P=d_P,
        I=I,
        F=F,
    )
    assert s_c_G.equals(Stress(70660, StressUnit.PSI), rel=1e-2)

    s_c_allowable_G = Spur.s_c_all(
        S_c=S_c_G, Z_N=Z_N_G, C_H=C_H, K_T=K_T, K_R=K_R, S_H=1.0
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


def test_example_14_8():
    m_G = 4
    H = Power(100, PowerUnit.HP)
    n_P = AngularVelocity(1120, AngularVelocityUnit.RPM)
    L_P = 1e9
    L_G = L_P / m_G
    R = 0.95
    material_P = Nitralloy
    S_F = 2
    S_H = 1.4

    phi = Angle(20, AngleUnit.DEGREE)
    N_P = 18
    N_G = 72
    Q_v = 6
    K_B = 1.0
    K_o = 1.0

    lewis_form_factor_P = Spur.Y(N=N_P)
    assert lewis_form_factor_P == approx(0.309)

    lewis_form_factor_G = Spur.Y(N=N_G)
    assert lewis_form_factor_G == approx(0.4324)

    J_P = Spur.J(N=N_P, other_N=N_G)
    assert J_P == approx(0.32, rel=0.1)

    J_G = Spur.J(N=N_G, other_N=N_P)
    assert J_G == approx(0.415, rel=0.1)

    Y_N_P = Spur.Y_N(L=L_P, H_B=material_P.H_B)
    assert Y_N_P == approx(0.938)

    Y_N_G = Spur.Y_N(L=L_G, H_B=material_P.H_B)
    assert Y_N_G == approx(0.961)

    Z_N_P = Spur.Z_N(L=L_P, is_nitrided=True)
    assert Z_N_P == approx(0.900)

    Z_N_G = Spur.Z_N(L=L_G, is_nitrided=True)
    assert Z_N_G == approx(0.929)

    P = InverseLength(4, InverseLengthUnit.PER_INCH)  # assumption
    d_P = Spur.d(N=N_P, P=P)
    assert d_P.equals(Length(4.5, LengthUnit.INCH))

    d_G = Spur.d(N=N_G, P=P)
    assert d_G.equals(Length(18.0, LengthUnit.INCH))

    V = Spur.V(d=d_P, n=n_P)
    assert V.equals(Velocity(1319, VelocityUnit.FPM))

    W_t = Spur.W_t(H=H, V=V)
    assert W_t.equals(Force(2502, ForceUnit.LBF))

    K_v = Spur.K_v(V=V, Q_v=Q_v)
    assert K_v == approx(1.480)

    F = Length(3, LengthUnit.INCH)  # assumption 3p <= F <= 5p. So take F=4pi/P

    K_s_P = Spur.K_s(F=F, Y=lewis_form_factor_P, P=P)
    assert K_s_P == approx(1.137)

    C_mc = Spur.C_mc(is_crowned=False)
    assert C_mc == 1
    C_pm = Spur.C_pm(S_1=0, S=1)
    assert C_pm == 1
    C_pf = Spur.C_pf(F=F, d_P=d_P)
    assert C_pf == approx(0.0667)
    C_ma = Spur.C_ma(F=F, gearing_condition="enclosed_commercial")
    assert C_ma == approx(0.175, rel=1e-2)
    C_e = Spur.C_e(is_gear_adjusted=False)
    assert C_e == 1
    K_m = Spur.K_m(C_mc=C_mc, C_pf=C_pf, C_pm=C_pm, C_ma=C_ma, C_e=C_e)
    assert K_m == approx(1.242, rel=2e-3)

    m_N = Spur.m_N()
    assert m_N == 1.0

    I = Spur.I(m_G=m_G, m_N=m_N, phi=phi, gear_mode="external")
    assert I == approx(0.1286)

    s_b_P = Spur.s_b(
        W_t=W_t, K_o=K_o, K_v=K_v, K_s=K_s_P, K_m=K_m, K_B=K_B, J=J_P, P=P, F=F
    )
    assert s_b_P.equals(Stress(21788, StressUnit.PSI), rel=5e-2)

    s_b_P_mag = s_b_P.to(StressUnit.PSI).magnitude
    s_b_all_P = Stress(s_b_P_mag, StressUnit.PSI)

    K_R = Spur.K_R(R=R)
    assert K_R == approx(0.85, rel=5e-2)

    K_T = Spur.K_T(T=None)
    assert K_T == 1.0

    res, params = solve_for_parameters(
        func=Spur.s_b_all,
        known_params={
            "Y_N": Y_N_P,
            "K_R": K_R,
            "K_T": K_T,
            "S_F": S_F,
        },
        unknown_params={"S_t": Stress},
        target_value=s_b_all_P,
        bounds={"S_t": (Stress(10000, StressUnit.PSI), Stress(50000, StressUnit.PSI))},
    )

    S_t = params["S_t"]
    assert S_t.equals(Stress(41114, StressUnit.PSI), rel=5e-2)
