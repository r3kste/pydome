from pydome.gears.bevel import Bevel
from pydome.materials import *
from pydome.solver import solve_for_parameters
from pydome.units import *  # noqa: F403
from pydome.utilities import approx


def test_example_15_1():
    P_d = InverseLength(5, InverseLengthUnit.PER_INCH)
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

    d_p = Bevel.d(N=N, P_d=P_d)
    assert d_p.equals(Length(5.0, LengthUnit.INCH))

    V_t = Bevel.V(d=d_p, n=n)
    assert V_t.equals(Velocity(785.4, VelocityUnit.FPM))

    K_o = Bevel.K_o(power_source="uniform", driven_machine="uniform")
    assert K_o == 1.0

    K_v = Bevel.K_v(Q_v=Q_v, V=V_t)
    assert K_v == approx(1.299)

    V_t_max = Bevel.V_t_max(Q_v=Q_v)
    assert V_t_max.equals(Velocity(4769, VelocityUnit.FPM))

    K_s = Bevel.K_s(P_d=P_d)
    assert K_s == approx(0.529)

    K_mb = Bevel.K_mb(no_of_straddling_members=no_of_straddling_members)
    assert K_mb == 1.25

    K_m = Bevel.K_m(F=F, K_mb=K_mb)
    assert K_m == approx(1.254)

    I = Bevel.I(N_pinion=N, N_gear=N)
    J_P = 0.216
    J_G = 0.216

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

    s_at = Bevel.S_t(H_B=H_B, grade=grade, is_through_hardened=is_through_hardened)
    assert s_at.equals(Stress(10020, StressUnit.PSI))

    s_wt = Bevel.s_b_all(S_t=s_at, K_L=K_L, K_T=K_T, K_R=K_R, S_F=S_F)
    assert s_wt.equals(Stress(10020, StressUnit.PSI))

    s_t = s_wt.copy()

    res, params = solve_for_parameters(
        func=Bevel.s_t,
        target_value=s_t,
        known_params=dict(
            P_d=P_d,
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

    s_ac = Bevel.S_c(H_B=H_B, grade=grade, is_through_hardened=is_through_hardened)
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
            P_d=P_d,
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
