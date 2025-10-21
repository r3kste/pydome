from pydome.gears.epicyclic import Planetary
from pydome.units import *  # noqa: F403
from pydome import kinematics, kinetics, utilities
from pydome.utilities import approx

def test_AGMA_6123_C16_Annex_C():
    # Low speed stage
    center_distance = Length(139, LengthUnit.MM)
    face_width = Length(104, LengthUnit.MM)
    normal_module = Length(6, LengthUnit.MM)
    pressure_angle = Angle(20, AngleUnit.DEGREE)
    helix_ange = Angle(0, AngleUnit.DEGREE)

    teeth_sun = 16
    profile_shift_coefficient_sun = 0.738
    teeth_planet = 28
    profile_shift_coefficient_planet = 0.629
    teeth_ring = 74
    profile_shift_coefficient_ring = -0.799
    number_of_planets = 3
    rpm_input = AngularVelocity(61, AngularVelocityUnit.RPM)
    torque_input = Torque(23.5e3, TorqueUnit.Nm)

    design_life = Time(10, TimeUnit.YEAR)
    # lubricant = ISO 220 EP

    m_G = Planetary.m_G(N_S=teeth_sun, N_R=teeth_ring)
    assert m_G == 5.625

    n_C = AngularVelocity(rpm_input.magnitude, rpm_input.unit)
    n_S = AngularVelocity(rpm_input.magnitude * m_G, rpm_input.unit)
    assert n_S.equals(AngularVelocity(343.125, AngularVelocityUnit.RPM))

    n_S_C = Planetary.n_S_C(n_S=n_S, n_C=n_C)
    assert n_S_C.equals(AngularVelocity(282.125, AngularVelocityUnit.RPM))

    assert n_S.unit == AngularVelocityUnit.RPM

    K_A_gear = 1.44
    load_spectrum = [
        (Torque(50e3, TorqueUnit.Nm), Time(0, TimeUnit.SECOND)),
        (Torque(47.5e3, TorqueUnit.Nm), Time(0, TimeUnit.SECOND)),
        (Torque(45e3, TorqueUnit.Nm), Time(3.15e2, TimeUnit.SECOND)),
        (Torque(42.5e3, TorqueUnit.Nm), Time(3.15e2, TimeUnit.SECOND)),
        (Torque(40e3, TorqueUnit.Nm), Time(3.15e2, TimeUnit.SECOND)),
        (Torque(37.5e3, TorqueUnit.Nm), Time(1.25e2, TimeUnit.SECOND)),
        (Torque(35e3, TorqueUnit.Nm), Time(2.21e4, TimeUnit.SECOND)),
        (Torque(32.5e3, TorqueUnit.Nm), Time(3.79e5, TimeUnit.SECOND)),
        (Torque(30e3, TorqueUnit.Nm), Time(2.99e5, TimeUnit.SECOND)),
        (Torque(27.5e3, TorqueUnit.Nm), Time(4.16e6, TimeUnit.SECOND)),
        (Torque(25e3, TorqueUnit.Nm), Time(3.66e7, TimeUnit.SECOND)),
        (Torque(22.5e3, TorqueUnit.Nm), Time(4.64e7, TimeUnit.SECOND)),
        (Torque(20e3, TorqueUnit.Nm), Time(4.05e7, TimeUnit.SECOND)),
        (Torque(17.5e3, TorqueUnit.Nm), Time(3.21e7, TimeUnit.SECOND)),
        (Torque(15e3, TorqueUnit.Nm), Time(4.10e7, TimeUnit.SECOND)),
        (Torque(10e3, TorqueUnit.Nm), Time(5.20e7, TimeUnit.SECOND)),
        (Torque(5e3, TorqueUnit.Nm), Time(3.03e7, TimeUnit.SECOND)),
        (Torque(0, TorqueUnit.Nm), Time(3.15e7, TimeUnit.SECOND)),
    ]
    total_time = sum([t.magnitude for _, t in load_spectrum])

    bearing_exponent = 10 / 3
    effective_load = (
        sum(
            F.magnitude**bearing_exponent * (t.magnitude / total_time)
            for F, t in load_spectrum
        )
    ) ** (1 / bearing_exponent)
    mean_load = sum(F.magnitude * (t.magnitude / total_time) for F, t in load_spectrum)
    K_A_bearings = 0.788

    # K_gamma = Planetary.K_gamma(T_Branch=None, T_Nom=None, N_CP=number_of_planets, application_level=2)
    K_gamma = 1.05

    T_C = Torque(torque_input.magnitude, torque_input.unit)
    T_S = Torque(torque_input.magnitude / m_G, torque_input.unit)

    assert T_S.equals(Torque(4.178e3, TorqueUnit.Nm), rel=0.5)

    F_eff = Planetary.H_M(T_S=T_S, n_S_C=n_S_C, K_gamma=K_gamma, N_CP=number_of_planets)
    assert F_eff.equals(Power(43.197e3, PowerUnit.WATT))

    S_H = 1.11  # ANSI/AGMA 6336-6
    S_F = 1.56

    K_Hbeta = 1.25  # due to rigid planet bearings

    L_10 = Time(50000, TimeUnit.HOUR)

    w_P = n_C.copy()
    assert w_P.equals(AngularVelocity(6.388, AngularVelocityUnit.RAD_per_SEC))

    alpha_rP = kinematics.alpha(w=w_P, r=center_distance)
    assert alpha_rP.equals(Acceleration(5.672, AccelerationUnit.MPS2))

    F_P = kinetics.force(m=Mass(11.602, MassUnit.KG), alpha=alpha_rP)
    assert F_P.equals(Force(65.808, ForceUnit.NEWTON))

    W_t = Planetary.F_t(
        F_Nom=kinetics.force(T=T_C, r=center_distance),
        K_gamma=K_gamma,
        K_A=K_A_bearings,
        N_CP=number_of_planets,
    )
    assert W_t.equals(Force(46630, ForceUnit.NEWTON))

    P_mag = utilities.vector_magnitude(
        W_t.to(ForceUnit.NEWTON).magnitude, F_P.to(ForceUnit.NEWTON).magnitude
    )
    F_eff = Force(P_mag, ForceUnit.NEWTON)
    assert F_eff.equals(Force(46630, ForceUnit.NEWTON))

    # L_10 = Bearings.Roller.Cylindrical.L_10()
    L_10 = Time(60149, TimeUnit.HOUR)

    Q_0 = kinetics.power(T=torque_input, n=rpm_input)
    assert Q_0.equals(Power(150e3, PowerUnit.WATT))

    eta_m = 0.97  # gearbox efficiency
    eta_e = 0.98  # generator efficiency

    eta_0 = eta_m * eta_e
    assert eta_0 == approx(0.95)
