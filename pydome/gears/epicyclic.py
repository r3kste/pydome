from math import cos, sin, ceil
from typing import Any, Literal

import optimagic as om

from pydome import kinetics, solver
from pydome.constants import pi
from pydome.materials import Material
from pydome.elements import Gear
from pydome.gears.spur import Spur
from pydome.units import *  # noqa: F403


class Planetary(Spur):
    def K_v(
        *,
        V: Velocity,
        Q_v: int,
    ) -> float:
        """Dynamic factor"""
        return max(1.05, Spur.K_v(V=V, Q_v=Q_v))

    def K_H_beta(
        *,
        C_mc: float,
        C_pf: float,
        C_pm: float,
        C_ma: float,
        C_e: float,
        quality_number: int | None = None,
    ):
        """Load distribution factor"""

        if any(param is None for param in [C_mc, C_pf, C_pm, C_ma, C_e]):
            if quality_number is None or quality_number < 7:
                return 1.4
            return 1.15

        return max(
            1.15,
            Spur.K_m(C_mc=C_mc, C_pf=C_pf, C_pm=C_pm, C_ma=C_ma, C_e=C_e),
        )

    def K_A():
        """Application factor"""
        pass

    def V_rel(
        *,
        d: Length,
        n_rel: AngularVelocity,
    ) -> Velocity:
        """Relative pitch line velocity

        Args:
            d (Length): Pitch diameter
            n_rel (AngularVelocity): Relative rotational speed
        """

        Spur.V(d=d, n=n_rel)

    def f_pt_T(
        *,
        V_rel: Velocity,
        is_external: bool,
    ) -> Length:
        """Single pitch deviation tolerance"""
        V_rel_mag = V_rel.to(VelocityUnit.MPS).magnitude

        _val = (0.508 / V_rel_mag) ** 0.4337

        if is_external:
            _val *= 76
        else:
            _val *= 102

        _val *= 1e-3  # Convert to millimeters
        return Length(_val, LengthUnit.MM)

    def m_G(
        *,
        N_S: int,
        N_R: int,
    ) -> float:
        """Speed Ratio"""
        return (N_S + N_R) / N_S

    def n_S_C(
        *,
        n_S,
        n_C,
    ) -> AngularVelocity:
        """Speed of sun gear relative to carrier"""
        n_S_mag = n_S.to(AngularVelocityUnit.RPM).magnitude
        n_C_mag = n_C.to(AngularVelocityUnit.RPM).magnitude

        n_S_C_mag = n_S_mag - n_C_mag
        return AngularVelocity(n_S_C_mag, AngularVelocityUnit.RPM)

    def n_R_C(
        *,
        n_R,
        n_C,
    ) -> AngularVelocity:
        """Speed of ring gear relative to the carrier"""
        n_R_mag = n_R.to(AngularVelocityUnit.RPM).magnitude
        n_C_mag = n_C.to(AngularVelocityUnit.RPM).magnitude

        n_R_C_mag = n_R_mag - n_C_mag
        return AngularVelocity(n_R_C_mag, AngularVelocityUnit.RPM)

    def n_P_C(
        *,
        n_S_C: AngularVelocity | None,
        n_R_C: AngularVelocity | None,
        N_S: int | None,
        N_R: int | None,
        N_P: int,
    ) -> AngularVelocity:
        """Speed of planet gear relative to the carrier"""
        if n_S_C is not None and N_S is not None:
            n_S_C_mag = n_S_C.to(AngularVelocityUnit.RPM).magnitude
            n_P_C_mag = -n_S_C_mag * N_S / N_P

        elif n_R_C is not None and N_R is not None:
            n_R_C_mag = n_R_C.to(AngularVelocityUnit.RPM).magnitude
            n_P_C_mag = n_R_C_mag * N_R / N_P

        else:
            raise ValueError("Either n_S_C and N_S or n_R_C and N_R must be provided.")

        return AngularVelocity(n_P_C_mag, AngularVelocityUnit.RPM)

    def C(
        *,
        theta: Angle,
        d_S: Length,
        d_P: Length,
        d_P_O: Length,
    ) -> Length:
        """Clearance

        Args:
            theta (Angle):
            d_S (Length): Sun pitch diameter
            d_P (Length): Planet pitch diameter
            d_P_O (Length): Planet outside diameter
        """

        theta_mag = theta.to(AngleUnit.RADIAN).magnitude
        d_S_mag = d_S.to(LengthUnit.MM).magnitude
        d_P_mag = d_P.to(LengthUnit.MM).magnitude
        d_P_O_mag = d_P_O.to(LengthUnit.MM).magnitude

        C_mag = sin(theta_mag / 2) * (d_S_mag + d_P_mag) - d_P_O_mag
        return Length(C_mag, LengthUnit.MM)

    def K_gamma(
        *,
        T_Branch: Torque | None,
        T_Nom: Torque | None,
        N_CP: int | None,
        application_level: int | None = None,
    ) -> float:
        """Mesh Load Factor

        Args:
            T_Branch (Torque): Torque in branch with heaviest load
            T_Nom (Torque): Nominal torque
            N_CP (int): Number of planets

        Table:
            AGMA 6123-C16 - Table 7
        """
        if any(param is None for param in [T_Branch, T_Nom, N_CP]):
            if application_level is None:
                pass

            # TODO: this is for N_CP = 3 only
            return 1.05

        T_Branch_mag = T_Branch.to(TorqueUnit.Nm).magnitude
        T_Nom_mag = T_Nom.to(TorqueUnit.Nm).magnitude

        return (T_Branch_mag / T_Nom_mag) * (N_CP)

    def W_t_per_mesh(
        *,
        F_Nom: Force,
        K_gamma: float,
        K_A: float,
        N_CP: int,
    ) -> Force:
        """Transmitted tangential load per mesh

        Args:
            F_Nom (Force): Nominal tangential load
            K_gamma (float): Mesh load factor
            K_A (float): Application factor
            N_CP (int): Number of planets
        """

        F_Nom_mag = F_Nom.to(ForceUnit.NEWTON).magnitude

        F_t_mag = F_Nom_mag * K_gamma * K_A / N_CP
        return Force(F_t_mag, ForceUnit.NEWTON)

    def H_M(
        *,
        T_S: Torque,
        n_S_C: AngularVelocity,
        K_gamma: float,
        N_CP: int,
    ) -> Power:
        """Transmitted Power by a mesh

        Args:
            T_S (Torque): Torque on sun gear
            n_S_C (AngularVelocity): Speed of sun gear relative to carrier
            K_gamma (float): Mesh load factor
            N_CP (int): Number of planets
        """

        T_S_mag = T_S.to(TorqueUnit.Nm).magnitude
        n_S_C_mag = n_S_C.to(AngularVelocityUnit.RPM).magnitude

        H_M_mag = (2 * pi * T_S_mag * n_S_C_mag * K_gamma) / (60 * N_CP)

        return Power(H_M_mag, PowerUnit.WATT)

    def C_G(
        *,
        N_P: int,
        N_S: int,
    ) -> float:
        """Gear Ratio Factor

        Args:
            N_P (int): Number of teeth on planet gear
            N_S (int): Number of teeth on sun gear
        """

        return N_P / (N_S + N_P)

    def estimate_d_wS(
        *,
        T_Nom: Torque,
        K_gamma: float,
        face_width_to_diameter_ratio: float,
        K: float,
        C_G: float,
        N_CP: int,
    ) -> Length:
        """Preliminary estimate for operating pitch diameter of sun gear

        Args:
            T_Nom (Torque): Nominal torque
            T_Branch (Torque, optional): Torque in branch with heaviest load
            face_width_to_diameter_ratio (float): Face width to diameter ratio
            K (float): Load intensity factor
            C_G (float): Gear ratio factor
            N_CP (int): Number of planets
        """

        T_Nom_mag = T_Nom.to(TorqueUnit.Nm).magnitude
        K_mag = K.to(StressUnit.MPA).magnitude

        d_wS_mag = (
            (2000 * K_gamma * T_Nom_mag)
            / (face_width_to_diameter_ratio * K_mag * C_G * N_CP)
        ) ** (1 / 3)

        return Length(d_wS_mag, LengthUnit.MM)

    def u_e(
        *,
        N_P: int,
        N_S: int,
    ) -> float:
        """Planet to sun tooth ratio

        Args:
            N_P (int): Number of teeth on planet gear
            N_S (int): Number of teeth on sun gear
        """
        return N_P / N_S

    def d_wP(
        *,
        d_wS: Length,
        u_e: float,
    ) -> Length:
        """Operating pitch diameter of planet gear

        Args:
            d_wS (Length): Operating pitch diameter of sun gear
            N_P (int): Number of teeth on planet gear
            N_S (int): Number of teeth on sun gear
        """

        d_wS_mag = d_wS.to(LengthUnit.MM).magnitude

        d_wP_mag = d_wS_mag * u_e
        return Length(d_wP_mag, LengthUnit.MM)

    def a(
        *,
        d_wS: Length,
        d_wP: Length,
    ) -> Length:
        """Center distance between sun and planet gears

        Args:
            d_wS (Length): Operating pitch diameter of sun gear
            d_wP (Length): Operating pitch diameter of planet gear
        """

        d_wS_mag = d_wS.to(LengthUnit.MM).magnitude
        d_wP_mag = d_wP.to(LengthUnit.MM).magnitude

        a_mag = (d_wS_mag + d_wP_mag) / 2
        return Length(a_mag, LengthUnit.MM)

    def estimate_alpha_wt() -> Angle:
        """Estimated operating transverse pressure angle of sun to planet mesh"""
        return Angle(22.5, AngleUnit.DEGREE)

    def d_bS(
        *,
        a: Length,
        alpha_wt: Angle,
        u_e: float,
    ) -> Length:
        """Base Circle Diameter of Sun Pinion

        Args:
            a (Length): Center distance between sun and planet gears
            alpha_wt (Angle): Operating transverse pressure angle of sun to planet mesh
            u_e (float): Planet to sun tooth ratio
        """

        a_mag = a.to(LengthUnit.MM).magnitude
        alpha_wt_rad = alpha_wt.to(AngleUnit.RADIAN).magnitude

        d_bS_mag = (2 * a_mag * cos(alpha_wt_rad)) / (1 + u_e)
        return Length(d_bS_mag, LengthUnit.MM)

    def p_b(
        *,
        d_bS: Length,
        N_S: int,
    ) -> Length:
        """Transverse base pitch

        Args:
            d_bS (Length): Base circle diameter of sun gear
            N_S (int): Number of teeth on sun gear
        """

        d_bS_mag = d_bS.to(LengthUnit.MM).magnitude

        p_b_mag = (pi * d_bS_mag) / N_S
        return Length(p_b_mag, LengthUnit.MM)

    def d_bP(
        *,
        p_b: Length,
        N_P: int,
    ) -> Length:
        """Base Circle Diameter of Planet Gear

        Args:
            p_b (Length): Transverse base pitch
            N_P (int): Number of teeth on planet gear
        """

        p_b_mag = p_b.to(LengthUnit.MM).magnitude

        d_bP_mag = (p_b_mag * N_P) / pi
        return Length(d_bP_mag, LengthUnit.MM)

    def verify_operating_pressure_angle(
        *,
        alpha_wt: Angle,
        d_bS: Length,
        d_bP: Length,
        a: Length,
    ) -> None:
        alpha_wt_mag = alpha_wt.to(AngleUnit.RADIAN).magnitude
        d_bS_mag = d_bS.to(LengthUnit.MM).magnitude
        d_bP_mag = d_bP.to(LengthUnit.MM).magnitude
        a_mag = a.to(LengthUnit.MM).magnitude

        assert cos(alpha_wt_mag) == (d_bS_mag + d_bP_mag) / (2 * a_mag)

    def t_R(
        *,
        d_f: Length,
        d_bore: Length,
    ) -> Length:
        """Rim Thickeness

        Args:
            d_f (Length): Root diameter
            d_bore (Length): Bore diameter
        """
        d_f_mag = d_f.to(LengthUnit.MM).magnitude
        d_bore_mag = d_bore.to(LengthUnit.MM).magnitude

        t_R_mag = (d_f_mag - d_bore_mag) / 2
        return Length(t_R_mag, LengthUnit.MM)

    def clamping_force_on_ring(
        *, d_bc: Length, mu: float, T_max: Torque, u_G: float
    ) -> Force:
        """Clamping force on the ring gear

        Args:
            d_bc (Length): Bolt circle diameter
            mu (float): Friction coefficient
            T_max (Torque): Maximum input torque
            u_G (float): Speed ratio
        """

        d_bc_mag = d_bc.to(LengthUnit.M).magnitude
        T_max_mag = T_max.to(TorqueUnit.Nm).magnitude
        T_R_mag = T_max_mag * (1 - (1 / u_G))

        F_clamp_mag = (2 * T_R_mag) / (d_bc_mag * mu)
        return Force(F_clamp_mag, ForceUnit.NEWTON)

    def get_pitch_line_velocity(
        *,
        gear: Gear,
        gear_member: Literal["sun", "planet", "ring"],
        mate: Gear,
        mate_member: Literal["sun", "planet", "ring"],
        **operating_conditions: Any,
    ) -> Velocity:
        # Pitch Line Velocities are same
        # So pitch line velocity of mate (sun) = pitch line velocity of gear (planet)
        if gear_member == "sun":
            sun_pitch_diameter = gear.get_pitch_diameter
            sun_rpm = gear.rpm
        elif mate_member == "sun":
            sun_pitch_diameter = mate.get_pitch_diameter
            sun_rpm = mate.rpm
        else:
            if gear_member == "planet":
                ring_pitch_diameter = mate.get_pitch_diameter
                planet_pitch_diameter = gear.get_pitch_diameter
                planet_rpm_relative_to_carrier = gear.rpm
            if gear_member == "ring":
                ring_pitch_diameter = gear.get_pitch_diameter
                planet_pitch_diameter = mate.get_pitch_diameter
                planet_rpm_relative_to_carrier = mate.rpm

            sun_pitch_diameter = ring_pitch_diameter - 2 * planet_pitch_diameter
            N_S = operating_conditions["N_S"]
            N_R = operating_conditions["N_R"]
            N_P = operating_conditions["N_P"]
            sun_rpm = (
                planet_rpm_relative_to_carrier
                * Planetary.u_e(N_P=N_P, N_S=N_S)
                / (1 - 1 / Planetary.m_G(N_S=N_S, N_R=N_R))
            )

        V = Planetary.V(d=sun_pitch_diameter, n=sun_rpm)
        return V

    def _bending_safety_factor(
        *,
        gear: Gear,
        gear_member: Literal["sun", "planet", "ring"],
        mate: Gear,
        mate_member: Literal["sun", "planet", "ring"],
        is_pinion: bool,
        **operating_conditions: Any,
    ) -> float:
        """Bending safety factor calculation for a gear mesh in a planetary gear set

        Args:
            gear (Gear): The gear for which the bending safety factor is being calculated.
            gear_member (Literal["sun", "planet", "ring"]): The member type of the gear.
            mate (Gear): The mating gear.
            mate_member (Literal["sun", "planet", "ring"]): The member type of the mating gear.
            is_pinion (bool): True if the gear is the driving gear (pinion), False otherwise.
            operating_conditions (dict): A dictionary containing the operating conditions required for the calculation.
        """

        if is_pinion:
            pinion, driven = gear, mate
        else:
            pinion, driven = mate, gear

        H = operating_conditions["H"]
        V = Planetary.get_pitch_line_velocity(
            gear=gear,
            gear_member=gear_member,
            mate=mate,
            mate_member=mate_member,
            **operating_conditions,
        )
        W_t = Planetary.W_t(H=H, V=V)

        N_CP = operating_conditions["N_CP"]
        K_gamma = get_value("K_gamma", operating_conditions)
        K_A = get_value("K_A", operating_conditions)

        W_t_per_mesh = Planetary.W_t_per_mesh(
            F_Nom=W_t, K_gamma=K_gamma, K_A=K_A, N_CP=N_CP
        )

        power_source = operating_conditions["power_source"]
        driven_machine = operating_conditions["driven_machine"]
        K_o = Planetary.K_o(power_source=power_source, driven_machine=driven_machine)

        K_v = Planetary.K_v(V=V, Q_v=gear.quality_number)

        Y = Planetary.Y(N=gear.n_teeth)
        K_s = Planetary.K_s(F=gear.face_width, Y=Y, P=gear.get_diametral_pitch)

        C_mc = Planetary.C_mc(is_crowned=gear.is_crowned)
        C_pf = Planetary.C_pf(F=gear.face_width, d_P=pinion.get_pitch_diameter)
        C_pm = Planetary.C_pm(S_1=None, S=None)
        gearing_condition = operating_conditions["gearing_condition"]
        C_ma = Planetary.C_ma(F=gear.face_width, gearing_condition=gearing_condition)
        C_e = Planetary.C_e(is_gear_adjusted=False)
        K_m = Planetary.K_H_beta(
            C_mc=C_mc,
            C_pf=C_pf,
            C_pm=C_pm,
            C_ma=C_ma,
            C_e=C_e,
            quality_number=gear.quality_number,
        )

        K_B = Planetary.K_B(t_R=None, h_t=None)
        J = Planetary.J(N=gear.n_teeth, other_N=mate.n_teeth)

        s_b = Planetary.s_b(
            W_t=W_t_per_mesh,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            K_B=K_B,
            P=gear.get_diametral_pitch,
            F=gear.face_width,
            J=J,
        )

        S_t = Planetary.S_t(
            heat_treatment=gear.heat_treatment,
            grade=gear.grade,
            H_B=gear.material.H_B,
            designation=gear.material.designation,
        )
        Y_N = Planetary.Y_N(
            L=gear.desired_cycles,
            H_B=gear.material.H_B,
            upper=True,
        )

        K_T = get_value("K_T", operating_conditions)
        K_R = get_value("K_R", operating_conditions)
        S_F = get_value("S_F", operating_conditions)

        s_b_all = Planetary.s_b_all(
            S_t=S_t,
            Y_N=Y_N,
            K_T=K_T,
            K_R=K_R,
            S_F=S_F,
        )

        s_f_b = Planetary.s_f_b(s_b=s_b, s_b_all=s_b_all)
        return s_f_b

    def _pitting_safety_factor(
        gear: Gear,
        gear_member: Literal["sun", "planet", "ring"],
        mate: Gear,
        mate_member: Literal["sun", "planet", "ring"],
        is_pinion: bool,
        **operating_conditions: Any,
    ) -> float:
        """Pitting safety factor"""

        if is_pinion:
            pinion, driven = gear, mate
        else:
            pinion, driven = mate, gear

        C_p = Planetary.C_p(
            mu_P=pinion.material.poissons_ratio,
            mu_G=driven.material.poissons_ratio,
            E_P=pinion.material.modulus_of_elasticity,
            E_G=driven.material.modulus_of_elasticity,
        )

        H = get_value("H", operating_conditions)
        V = Planetary.get_pitch_line_velocity(
            gear=gear,
            gear_member=gear_member,
            mate=mate,
            mate_member=mate_member,
            **operating_conditions,
        )
        W_t = Planetary.W_t(H=H, V=V)

        K_gamma = get_value("K_gamma", operating_conditions)
        K_A = get_value("K_A", operating_conditions)
        N_CP = get_value("N_CP", operating_conditions)
        W_t_per_mesh = Planetary.W_t_per_mesh(
            F_Nom=W_t, K_gamma=K_gamma, K_A=K_A, N_CP=N_CP
        )

        power_source = get_value("power_source", operating_conditions)
        driven_machine = get_value("driven_machine", operating_conditions)
        K_o = Planetary.K_o(power_source=power_source, driven_machine=driven_machine)

        K_v = Planetary.K_v(V=V, Q_v=gear.quality_number)

        Y = Planetary.Y(N=gear.n_teeth)
        K_s = Planetary.K_s(F=gear.face_width, Y=Y, P=gear.get_diametral_pitch)

        C_mc = Planetary.C_mc(is_crowned=gear.is_crowned)
        C_pf = Planetary.C_pf(F=gear.face_width, d_P=pinion.get_pitch_diameter)
        C_pm = Planetary.C_pm(S_1=None, S=None)
        gearing_condition = get_value("gearing_condition", operating_conditions)
        C_ma = Planetary.C_ma(F=gear.face_width, gearing_condition=gearing_condition)
        C_e = Planetary.C_e(is_gear_adjusted=False)
        K_m = Planetary.K_H_beta(
            C_mc=C_mc,
            C_pf=C_pf,
            C_pm=C_pm,
            C_ma=C_ma,
            C_e=C_e,
            quality_number=gear.quality_number,
        )

        C_f = Planetary.C_f()

        m_G = get_value("m_G", operating_conditions)
        m_N = get_value("m_N", operating_conditions)
        phi = gear.pressure_angle
        gear_mode = get_value("gear_mode", operating_conditions)

        I = Planetary.I(
            m_G=m_G,
            m_N=m_N,
            phi=phi,
            gear_mode=gear_mode,
        )

        s_c = Planetary.s_c(
            C_p=C_p,
            W_t=W_t_per_mesh,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            C_f=C_f,
            d_P=pinion.get_pitch_diameter,
            F=gear.face_width,
            I=I,
        )

        S_c = Planetary.S_c(
            heat_treatment=gear.heat_treatment,
            H_B=gear.material.H_B,
            grade=gear.grade,
        )

        Z_N = Planetary.Z_N(
            L=gear.desired_cycles, is_nitrided=gear.heat_treatment == "nitrided"
        )

        C_H = Planetary.C_H(
            m_G=m_G,
            H_B_P=pinion.material.H_B,
            f_P=None,
            H_B_G=driven.material.H_B,
            is_pinion=is_pinion,
        )

        K_T = get_value("K_T", operating_conditions)
        K_R = get_value("K_R", operating_conditions)
        S_H = get_value("S_H", operating_conditions)

        s_c_all = Planetary.s_c_all(
            S_c=S_c,
            Z_N=Z_N,
            C_H=C_H,
            K_T=K_T,
            K_R=K_R,
            S_H=S_H,
        )

        s_f_c = Planetary.s_f_c(s_c=s_c, s_c_all=s_c_all)
        return s_f_c

    def bending_safety_factor(
        *,
        sun: Gear,
        planet: Gear,
        ring: Gear,
        operating_conditions: dict,
        return_all: bool = False,
    ) -> float:
        """Bending safety factor"""
        s_b_f_SP = Planetary._bending_safety_factor(
            gear=sun,
            gear_member="sun",
            mate=planet,
            mate_member="planet",
            is_pinion=True,
            **operating_conditions,
            gear_mode="external",
        )

        s_b_f_PS = Planetary._bending_safety_factor(
            gear=planet,
            gear_member="planet",
            mate=sun,
            mate_member="sun",
            is_pinion=False,
            **operating_conditions,
            gear_mode="external",
        )

        s_b_f_PR = Planetary._bending_safety_factor(
            gear=planet,
            gear_member="planet",
            mate=ring,
            mate_member="ring",
            is_pinion=True,
            **operating_conditions,
            gear_mode="internal",
        )

        s_b_f_RP = Planetary._bending_safety_factor(
            gear=ring,
            gear_member="ring",
            mate=planet,
            mate_member="planet",
            is_pinion=False,
            **operating_conditions,
            gear_mode="internal",
        )

        if return_all:
            return s_b_f_SP, s_b_f_PS, s_b_f_PR, s_b_f_RP
        else:
            return min(s_b_f_SP, s_b_f_PS, s_b_f_PR, s_b_f_RP)

    def pitting_safety_factor(
        *,
        sun: Gear,
        planet: Gear,
        ring: Gear,
        operating_conditions: dict,
        return_all: bool = False,
    ) -> float:
        """Pitting safety factor"""
        s_c_f_SP = Planetary._pitting_safety_factor(
            gear=sun,
            gear_member="sun",
            mate=planet,
            mate_member="planet",
            is_pinion=True,
            **operating_conditions,
            gear_mode="external",
        )

        s_c_f_PS = Planetary._pitting_safety_factor(
            gear=planet,
            gear_member="planet",
            mate=sun,
            mate_member="sun",
            is_pinion=False,
            **operating_conditions,
            gear_mode="external",
        )

        s_c_f_PR = Planetary._pitting_safety_factor(
            gear=planet,
            gear_member="planet",
            mate=ring,
            mate_member="ring",
            is_pinion=True,
            **operating_conditions,
            gear_mode="internal",
        )

        s_c_f_RP = Planetary._pitting_safety_factor(
            gear=ring,
            gear_member="ring",
            mate=planet,
            mate_member="planet",
            is_pinion=False,
            **operating_conditions,
            gear_mode="internal",
        )

        if return_all:
            return s_c_f_SP, s_c_f_PS, s_c_f_PR, s_c_f_RP
        else:
            return min(s_c_f_SP, s_c_f_PS, s_c_f_PR, s_c_f_RP)

    def safety_factor(
        *,
        sun: Gear,
        planet: Gear,
        ring: Gear,
        operating_conditions: dict,
        return_all: bool = False,
    ) -> tuple[float, float]:
        """Bending and Pitting safety factors"""
        s_b_f = Planetary.bending_safety_factor(
            sun=sun,
            planet=planet,
            ring=ring,
            operating_conditions=operating_conditions,
            return_all=return_all,
        )

        s_c_f = Planetary.pitting_safety_factor(
            sun=sun,
            planet=planet,
            ring=ring,
            operating_conditions=operating_conditions,
            return_all=return_all,
        )

        if return_all:
            return {
                "Bending Sun - Planet Mesh": s_b_f[0],
                "Bending Planet - Sun Mesh": s_b_f[1],
                "Bending Planet - Ring Mesh": s_b_f[2],
                "Bending Ring - Planet Mesh": s_b_f[3],
                "Pitting Sun - Planet Mesh": s_c_f[0],
                "Pitting Planet - Sun Mesh": s_c_f[1],
                "Pitting Planet - Ring Mesh": s_c_f[2],
                "Pitting Ring - Planet Mesh": s_c_f[3],
            }
        else:
            return min(s_b_f, s_c_f)


def get_value(
    attr_name: str,
    operating_conditions: dict,
    default: Any | None = None,
    max_depth: int = 3,
) -> Any:
    if attr_name in operating_conditions:
        return operating_conditions[attr_name]
    else:
        attr = getattr(Planetary, attr_name, default)

        if callable(attr):
            import inspect

            sig = inspect.signature(attr)

            params = {}
            for arg in sig.parameters.keys():
                params[arg] = get_value(
                    attr_name=arg,
                    operating_conditions=operating_conditions,
                    default=None,
                    max_depth=max_depth - 1,
                )
            return attr(**params)
        else:
            return attr


def demo(optimize: bool = True, **inputs):
    from pydome.materials import Steel

    n_S = get_value("n_S", inputs, AngularVelocity(900, AngularVelocityUnit.RPM))
    n_C = get_value("n_C", inputs, AngularVelocity(225, AngularVelocityUnit.RPM))
    m_G = n_S / n_C
    H = get_value("H", inputs, Power(600, PowerUnit.WATT))
    N_CP = get_value("N_CP", inputs, 3)

    N_S = get_value("N_S", inputs, 24)
    res, params = solver.solve_for_parameters(
        func=Planetary.m_G,
        target_value=m_G,
        known_params={"N_S": N_S},
        unknown_params={"N_R": float},
        bounds={"N_R": (16, 120)},
    )
    N_R = round(params["N_R"])
    N_P = round((N_R - N_S) / 2)

    T_Nom = kinetics.torque(P=H, n=n_S)
    K_gamma = get_value("K_gamma", inputs, 1.05)
    K = get_value("K", inputs, Stress(1.38, StressUnit.MPA))
    C_G = Planetary.C_G(N_P=N_P, N_S=N_S)
    face_width_to_diameter_ratio = get_value(
        "face_width_to_diameter_ratio", inputs, 1.2
    )
    esimated_d_S = Planetary.estimate_d_wS(
        T_Nom=T_Nom,
        K_gamma=K_gamma,
        face_width_to_diameter_ratio=face_width_to_diameter_ratio,
        K=K,
        C_G=C_G,
        N_CP=N_CP,
    )

    estimated_face_width = face_width_to_diameter_ratio * esimated_d_S

    u_e = Planetary.u_e(N_P=N_P, N_S=N_S)
    esimated_d_P = Planetary.d_wP(d_wS=esimated_d_S, u_e=u_e)

    unknowns, bounds = {}, {}

    sun = Gear(
        n_teeth=N_S,
        rpm=n_S,
        pitch_diameter=get_value("sun.pitch_diameter", inputs, solver.UNKNOWN),
        face_width=get_value("sun.face_width", inputs, solver.UNKNOWN),
        desired_cycles=get_value("desired_cycles", inputs, 1e10),
        material=Steel,
        quality_number=get_value("quality_number", inputs, 6),
        is_crowned=get_value("is_crowned", inputs, False),
        pressure_angle=get_value("pressure_angle", inputs, Angle(20, AngleUnit.DEGREE)),
        heat_treatment=get_value("sun_heat_treatment", inputs, "through"),
        grade=get_value("sun_grade", inputs, 1),
    )
    unknowns["sun.pitch_diameter"] = Length
    unknowns["sun.face_width"] = Length
    bounds["sun.pitch_diameter"] = (0.5 * esimated_d_S, 1.5 * esimated_d_S)
    bounds["sun.face_width"] = (0.5 * estimated_face_width, 1.5 * estimated_face_width)

    n_S_C = Planetary.n_S_C(n_S=n_S, n_C=n_C)
    planet = Gear(
        n_teeth=N_P,
        rpm=Planetary.n_P_C(N_S=N_S, N_R=N_R, N_P=N_P, n_S_C=n_S_C, n_R_C=None),
        pitch_diameter=get_value("planet.pitch_diameter", inputs, solver.UNKNOWN),
        face_width=get_value("planet.face_width", inputs, solver.UNKNOWN),
        desired_cycles=get_value("desired_cycles", inputs, 1e10),
        material=Steel,
        quality_number=get_value("quality_number", inputs, 6),
        is_crowned=get_value("is_crowned", inputs, False),
        pressure_angle=get_value("pressure_angle", inputs, Angle(20, AngleUnit.DEGREE)),
        heat_treatment=get_value("planet_heat_treatment", inputs, "through"),
        grade=get_value("planet_grade", inputs, 1),
    )
    unknowns["planet.pitch_diameter"] = Length
    unknowns["planet.face_width"] = Length
    bounds["planet.pitch_diameter"] = (0.5 * esimated_d_P, 1.5 * esimated_d_P)
    bounds["planet.face_width"] = (
        0.5 * estimated_face_width,
        1.5 * estimated_face_width,
    )

    ring = Gear(
        n_teeth=N_R,
        rpm=AngularVelocity(0, AngularVelocityUnit.RPM),
        pitch_diameter=get_value("ring.pitch_diameter", inputs, solver.UNKNOWN),
        face_width=get_value("ring.face_width", inputs, solver.UNKNOWN),
        desired_cycles=get_value("desired_cycles", inputs, 1e10),
        material=Steel,
        quality_number=get_value("quality_number", inputs, 6),
        is_crowned=get_value("is_crowned", inputs, False),
        pressure_angle=get_value("pressure_angle", inputs, Angle(20, AngleUnit.DEGREE)),
        heat_treatment=get_value("ring_heat_treatment", inputs, "through"),
        grade=get_value("ring_grade", inputs, 1),
    )
    unknowns["ring.pitch_diameter"] = Length
    unknowns["ring.face_width"] = Length
    bounds["ring.pitch_diameter"] = (
        0.5 * (esimated_d_S + 2 * esimated_d_P),
        1.5 * (esimated_d_S + 2 * esimated_d_P),
    )
    bounds["ring.face_width"] = (0.5 * estimated_face_width, 1.5 * estimated_face_width)

    operating_conditions = dict(
        H=H,
        N_S=N_S,
        N_R=N_R,
        N_P=N_P,
        N_CP=N_CP,
        power_source="uniform",
        driven_machine="uniform",
        gearing_condition="enclosed_commercial",
        T=Temperature(25, TemperatureUnit.CELSIUS),
        R=0.99,
        S_F=2.0,
        S_H=1.4,
        K_gamma=K_gamma,
        application_level=2,
        K_A=1,
    )
    operating_conditions.update(inputs)

    if optimize:
        constraints = [
            om.EqualityConstraint(
                selector=lambda params: [
                    params["planet.face_width"],
                    params["sun.face_width"],
                ]
            ),
            om.EqualityConstraint(
                selector=lambda params: [
                    params["ring.face_width"],
                    params["sun.face_width"],
                ]
            ),
            om.NonlinearConstraint(
                selector=lambda params: [
                    params["planet.pitch_diameter"],
                    params["sun.pitch_diameter"],
                ],
                func=lambda x: x[0] / x[1],
                value=u_e,
            ),
            om.NonlinearConstraint(
                selector=lambda params: [
                    params["sun.pitch_diameter"],
                    params["sun.face_width"],
                ],
                func=lambda x: x[1] / x[0],
                value=face_width_to_diameter_ratio,
            ),
            om.NonlinearConstraint(
                selector=lambda params: [
                    params["ring.pitch_diameter"],
                    params["sun.pitch_diameter"],
                    params["planet.pitch_diameter"],
                ],
                func=lambda x: x[0] - (x[1] + 2 * x[2]),
                value=0,
            ),
        ]

        res, params = solver.solve_for_parameters(
            func=Planetary.safety_factor,
            target_value=1.0,
            known_params={
                "sun": sun,
                "planet": planet,
                "ring": ring,
                "operating_conditions": operating_conditions,
            },
            unknown_params=unknowns,
            bounds=bounds,
            constraints=constraints,
        )

        params["_n_P_C"] = Planetary.n_P_C(
            N_S=N_S,
            N_P=N_P,
            N_R=N_R,
            n_S_C=n_S_C,
            n_R_C=None,
        )
        return res, params
    else:
        return Planetary.safety_factor(
            sun=sun,
            planet=planet,
            ring=ring,
            operating_conditions=operating_conditions,
            return_all=True,
        )
