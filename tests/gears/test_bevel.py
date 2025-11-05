from pydome.gears.bevel import Bevel
from pydome.materials import *  # noqa: F403
from pydome.solver import solve_for_parameters
from pydome.units import *  # noqa: F403
from pydome.utilities import approx


def test_example_15_1():
    P = InverseLength(5, InverseLengthUnit.PER_INCH)
    N = 25
    F = Length(1.1, LengthUnit.INCH)
    phi = Angle(20, AngleUnit.DEGREE)
    grade = 1
    is_through_hardened = True
    H_B = 180
    is_crowned = False
    Q_v = 7
    S_F = S_H = 1.0
    no_of_straddling_members = 0
    t = Temperature(77, TemperatureUnit.FAHRENHEIT)

    L = 1e7
    R = 0.99
    n = AngularVelocity(600, AngularVelocityUnit.RPM)

    d_p = Bevel.d(N=N, P=P)
    assert d_p.equals(Length(5.0, LengthUnit.INCH))

    V_t = Bevel.V(d=d_p, n=n)
    assert V_t.equals(Velocity(785.4, VelocityUnit.FPM))

    K_o = Bevel.K_o(power_source="uniform", driven_machine="uniform")
    assert K_o == 1.0

    K_v = Bevel.K_v(Q_v=Q_v, V=V_t)
    assert K_v == approx(1.299)

    V_t_max = Bevel.V_t_max(Q_v=Q_v)
    assert V_t_max.equals(Velocity(4769, VelocityUnit.FPM))

    K_s = Bevel.K_s(P=P)
    assert K_s == approx(0.529)

    K_mb = Bevel.K_mb(no_of_straddling_members=no_of_straddling_members)
    assert K_mb == 1.25

    K_m = Bevel.K_m(F=F, K_mb=K_mb)
    assert K_m == approx(1.254)

    I = Bevel.I(N_pinion=N, N_gear=N)
    assert I == approx(0.065, rel=2e-3)

    J_P = Bevel.J(N_gear=N, N_mate=N)
    assert J_P == approx(0.216, rel=4e-3)

    K_L = Bevel.K_L(N_L=L, is_critical=True)
    assert K_L == approx(1)

    C_L = Bevel.C_L(N_L=L)
    assert C_L == approx(1.32)

    C_H = Bevel.C_H(m_G=1, H_B_P=H_B, f_P=None, H_B_G=H_B, is_pinion=True)
    assert C_H == 1

    K_x = Bevel.K_x()
    assert K_x == 1

    K_T = Bevel.K_T(t=t)
    assert K_T == 1

    K_R = Bevel.K_R(R=R)
    assert K_R == approx(1.0)

    C_R = Bevel.C_R(R=R)
    assert C_R == approx(1.0)

    s_at = Bevel.S_t(
        H_B=H_B,
        grade=grade,
        is_through_hardened=is_through_hardened,
        is_case_hardened=False,
    )
    assert s_at.equals(Stress(10020, StressUnit.PSI))

    s_wt = Bevel.s_b_all(S_t=s_at, K_L=K_L, K_T=K_T, K_R=K_R, S_F=S_F)
    assert s_wt.equals(Stress(10020, StressUnit.PSI))

    s_t = s_wt.copy()

    res, params = solve_for_parameters(
        func=Bevel.s_t,
        target_value=s_t,
        known_params=dict(
            P=P,
            F=F,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            J=J_P,
            K_x=K_x,
        ),
        unknown_params=dict(W_t=Force),
        bounds={"W_t": (Force(100, ForceUnit.LBF), Force(1000, ForceUnit.LBF))},
    )
    W_t = params["W_t"]
    assert W_t.equals(Force(552.6, ForceUnit.LBF), rel=3e-3)

    s_ac = Bevel.S_c(
        H_B=H_B,
        grade=grade,
        is_through_hardened=is_through_hardened,
        is_case_hardened=False,
    )
    assert s_ac.equals(Stress(85000, StressUnit.PSI))

    s_wc = Bevel.s_c_all(S_c=s_ac, C_L=C_L, C_H=C_H, K_T=K_T, C_R=C_R, S_H=S_H)
    assert s_wc.equals(Stress(112200, StressUnit.PSI))

    C_p_mag = 2290

    C_p = SqrtPressure(
        C_p_mag,
        SqrtPressureUnit.SQRT_PSI,
    )

    C_s = Bevel.C_s(F=F)
    assert C_s == approx(0.575)

    C_xc = Bevel.C_xc(is_crowned=is_crowned)
    assert C_xc == 2.0

    s_c = s_wc.copy()
    res, params = solve_for_parameters(
        func=Bevel.s_c,
        target_value=s_c,
        known_params=dict(
            d_p=d_p,
            F=F,
            K_o=K_o,
            K_v=K_v,
            K_m=K_m,
            C_s=C_s,
            C_p=C_p,
            C_xc=C_xc,
            I=I,
        ),
        unknown_params=dict(W_t=Force),
        bounds={"W_t": (Force(100, ForceUnit.LBF), Force(2000, ForceUnit.LBF))},
    )
    W_t = params["W_t"]
    assert W_t.equals(Force(458.1, ForceUnit.LBF), rel=3e-3)

    L = 1e9
    R = 0.995
    S_F = S_H = 1.5

    K_L = Bevel.K_L(N_L=L, is_critical=True)
    assert K_L == approx(0.8618)

    K_R = Bevel.K_R(R=R)
    assert K_R == approx(1.075)

    C_R = Bevel.C_R(R=R)
    assert C_R == approx(1.037)

    C_L = Bevel.C_L(N_L=L)
    assert C_L == approx(1)

    s_wt = Bevel.s_b_all(S_t=s_at, K_L=K_L, K_T=K_T, K_R=K_R, S_F=S_F)
    assert s_wt.equals(Stress(5355, StressUnit.PSI))

    s_t = s_wt.copy()
    res, params = solve_for_parameters(
        func=Bevel.s_t,
        target_value=s_t,
        known_params=dict(
            P=P,
            F=F,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            J=J_P,
            K_x=K_x,
        ),
        unknown_params=dict(W_t=Force),
        bounds={"W_t": (Force(100, ForceUnit.LBF), Force(1000, ForceUnit.LBF))},
    )
    W_t = params["W_t"]
    assert W_t.equals(Force(295.4, ForceUnit.LBF), rel=3e-3)

    s_wc = Bevel.s_c_all(S_c=s_ac, C_L=C_L, C_H=C_H, K_T=K_T, C_R=C_R, S_H=S_H)
    assert s_wc.equals(Stress(54640, StressUnit.PSI))

    s_c = s_wc.copy()
    res, params = solve_for_parameters(
        func=Bevel.s_c,
        target_value=s_c,
        known_params=dict(
            d_p=d_p,
            F=F,
            K_o=K_o,
            K_v=K_v,
            K_m=K_m,
            C_s=C_s,
            C_p=C_p,
            C_xc=C_xc,
            I=I,
        ),
        unknown_params=dict(W_t=Force),
        bounds={"W_t": (Force(50, ForceUnit.LBF), Force(500, ForceUnit.LBF))},
    )
    W_t = params["W_t"]
    # assert W_t.equals(Force(108.6, ForceUnit.LBF), rel=5e-3)


def test_example_15_2():
    H = Power(6.85, PowerUnit.HP)
    n_P = AngularVelocity(900, AngularVelocityUnit.RPM)
    m_G = 3
    t = Temperature(300, TemperatureUnit.FAHRENHEIT)
    phi = Angle(25, AngleUnit.DEGREE)
    S_F = 2.0
    N_P = 20
    grade = 1
    load = "uniform"
    is_crowned = True
    R = 0.995
    L_P = 1e9
    L_G = L_P / m_G

    n_G = n_P / m_G
    N_G = N_P * m_G

    S_H = np.sqrt(2)

    no_of_straddling_members = 0

    C_L_G = Bevel.C_L(N_L=L_G)
    assert C_L_G == approx(1.068)

    C_L_P = Bevel.C_L(N_L=L_P)
    assert C_L_P == approx(1)

    K_L_G = Bevel.K_L(N_L=L_G, is_critical=True)
    assert K_L_G == approx(0.8929)

    K_L_P = Bevel.K_L(N_L=L_P, is_critical=True)
    assert K_L_P == approx(0.8618)

    K_R = Bevel.K_R(R=R)
    assert K_R == approx(1.075)

    C_R = Bevel.C_R(R=R)
    assert C_R == approx(1.037)

    K_T = Bevel.K_T(t=t)
    assert K_T == approx(1.070)

    K_x = Bevel.K_x()
    assert K_x == 1

    C_xc = Bevel.C_xc(is_crowned=is_crowned)
    assert C_xc == 1.5

    gamma, GAMMA = Bevel.pitch_angle(
        N_p=N_P, N_g=N_G, shaft_angle=Angle(90, AngleUnit.DEGREE)
    )
    assert gamma.equals(Angle(18.43, AngleUnit.DEGREE), rel=2e-3)
    assert GAMMA.equals(Angle(71.57, AngleUnit.DEGREE), rel=2e-3)

    I = Bevel.I(N_pinion=N_P, N_gear=N_G)
    assert I == approx(0.0825, rel=1e-2)

    J_P = Bevel.J(N_gear=N_G, N_mate=N_P)
    assert J_P == approx(0.248, rel=1e-2)

    J_G = Bevel.J(N_gear=N_P, N_mate=N_G)
    assert J_G == approx(0.202, rel=1e-2)

    P = InverseLength(8, InverseLengthUnit.PER_INCH)  # decision1

    K_s = Bevel.K_s(P=P)
    assert K_s == approx(0.5134)

    d_P = Bevel.d(N=N_P, P=P)
    assert d_P.equals(Length(2.5, LengthUnit.INCH))

    d_G = Bevel.d(N=N_G, P=P)
    assert d_G.equals(Length(7.5, LengthUnit.INCH))

    v_t = Bevel.V(d=d_P, n=n_P)
    assert v_t.equals(Velocity(589, VelocityUnit.FPM))

    W_t = Bevel.W_t(H=H, V=v_t)
    assert W_t.equals(Force(383.8, ForceUnit.LBF))

    A_0 = Bevel.cone_distance(d_P=d_P, d_G=d_G, shaft_angle=Angle(90, AngleUnit.DEGREE))
    assert A_0.equals(Length(3.954, LengthUnit.INCH), rel=1e-3)

    F = Bevel.face_width_max(A_0=A_0, P=P)
    assert F.equals(Length(1.186, LengthUnit.INCH), rel=1e-3)

    F = Length(1.25, LengthUnit.INCH)  # decision2

    C_s = Bevel.C_s(F=F)
    assert C_s == approx(0.5937, rel=1e-3)

    K_mb = Bevel.K_mb(no_of_straddling_members=no_of_straddling_members)

    K_m = Bevel.K_m(F=F, K_mb=K_mb)
    assert K_m == approx(1.256, rel=1e-3)

    Q_v = 6  # decision3

    K_v = Bevel.K_v(Q_v=Q_v, V=v_t)
    assert K_v == approx(1.325, rel=1e-3)

    is_case_hardened = True
    is_through_hardened = False  # decision4

    S_c = Bevel.S_c(
        H_B=None,
        grade=grade,
        is_case_hardened=is_case_hardened,
        is_through_hardened=is_through_hardened,
    )
    assert S_c.equals(Stress(200000, StressUnit.PSI))

    S_t = Bevel.S_t(
        H_B=None,
        grade=grade,
        is_case_hardened=is_case_hardened,
        is_through_hardened=is_through_hardened,
    )
    assert S_t.equals(Stress(30000, StressUnit.PSI))

    K_o = Bevel.K_o(power_source=load, driven_machine=load)
    assert K_o == 1.0

    s_t_G = Bevel.s_t(
        W_t=W_t,
        P=P,
        F=F,
        K_o=K_o,
        K_v=K_v,
        K_s=K_s,
        K_m=K_m,
        J=J_G,
        K_x=K_x,
    )
    assert s_t_G.equals(Stress(10390, StressUnit.PSI), rel=1e-3)

    s_wt_G = Bevel.s_b_all(S_t=S_t, K_L=K_L_G, K_T=K_T, K_R=K_R, S_F=S_F)
    assert s_wt_G.equals(Stress(11640, StressUnit.PSI), rel=1e-3)

    s_f_b_G = Bevel.s_f_b(s_b_all=s_wt_G, s_b=s_t_G)
    assert s_f_b_G == approx(1.12, rel=1e-2)

    s_t_P = s_t_G * (J_G / J_P)
    assert s_t_P.equals(Stress(8463, StressUnit.PSI), rel=1e-2)

    s_wt_P = Bevel.s_b_all(S_t=S_t, K_L=K_L_P, K_T=K_T, K_R=K_R, S_F=S_F)
    assert s_wt_P.equals(Stress(11240, StressUnit.PSI), rel=1e-3)

    s_f_b_P = Bevel.s_f_b(s_b_all=s_wt_P, s_b=s_t_P)
    assert s_f_b_P == approx(1.33, rel=1e-2)

    C_p = SqrtPressure(2290, SqrtPressureUnit.SQRT_PSI)

    s_c_G = Bevel.s_c(
        W_t=W_t,
        d_p=d_P,
        F=F,
        K_o=K_o,
        K_v=K_v,
        K_m=K_m,
        C_s=C_s,
        C_p=C_p,
        C_xc=C_xc,
        I=I,
    )
    assert s_c_G.equals(Stress(107560, StressUnit.PSI), rel=1e-2)

    s_wc_G = Bevel.s_c_all(S_c=S_c, C_L=C_L_G, C_H=1, K_T=K_T, C_R=C_R, S_H=S_H)
    assert s_wc_G.equals(Stress(136120, StressUnit.PSI), rel=1e-2)

    s_f_c_G = Bevel.s_f_c(s_c_all=s_wc_G, s_c=s_c_G)
    assert s_f_c_G == approx(1.266, rel=1e-2)

    s_wc_P = Bevel.s_c_all(S_c=S_c, C_L=C_L_P, C_H=1, K_T=K_T, C_R=C_R, S_H=S_H)
    assert s_wc_P.equals(Stress(127450, StressUnit.PSI), rel=1e-2)

    s_f_c_P = Bevel.s_f_c(s_c_all=s_wc_P, s_c=s_c_G)
    assert s_f_c_P == approx(1.186, rel=1e-2)
