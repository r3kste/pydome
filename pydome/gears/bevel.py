from math import cos, exp, log, sin, sqrt, tan, atan, pi, atan2, log10
from typing import Literal
import numpy as np
from pydome.units import *
from pydome.gears.spur import Spur


class Bevel(Spur):
    def d(*, N: int, P_d: InverseLength) -> Length:
        """
        Pitch diameter at large end

        Args:
            N (int): Number of teeth
            P_d (InverseLength): Outer transverse diametral pitch
        """

        P_d_mag = P_d.to(InverseLengthUnit.PER_INCH).magnitude

        d_mag = N / P_d_mag
        return Length(d_mag, LengthUnit.INCH)

    def V(*, d: Length, n: AngularVelocity) -> Velocity:
        """
        Pitch-line velocity at outer pitch circle

        Args:
            d (Length): Pitch diameter
            n (AngularVelocity): Rotational speed
        """

        d_mag = d.to(LengthUnit.INCH).magnitude
        n_mag = n.to(AngularVelocityUnit.RPM).magnitude
        V_mag = (pi * d_mag * n_mag) / 12
        return Velocity(V_mag, VelocityUnit.FPM)

    def W_t(
        *,
        H: Power,
        V: Velocity,
    ) -> Force:
        """Tangential Load

        Args:
            H (Power): Power transmitted
            V (Velocity): Pitch line velocity
        """

        H_mag = H.to(PowerUnit.HP).magnitude
        V_mag = V.to(VelocityUnit.FPM).magnitude
        W_t_mag = (33000 * H_mag) / V_mag
        return Force(W_t_mag, ForceUnit.LBF)

    def cone_distance(*, d_P: Length, d_G: Length, shaft_angle: Angle) -> Length:
        """
        Outer cone distance

        Args:
            d_P: Pinion pitch diameter
            d_G: Gear pitch diameter
            shaft_angle_deg: Shaft angle
        """
        d_P_mag = d_P.to(LengthUnit.INCH).magnitude
        d_G_mag = d_G.to(LengthUnit.INCH).magnitude
        shaft_angle_rad = shaft_angle.to(AngleUnit.DEGREE).magnitude * pi / 180

        A_0_mag = (d_P_mag + d_G_mag) / (2 * sin(shaft_angle_rad / 2))
        return Length(A_0_mag, LengthUnit.INCH)

    def pitch_angle(*, N_p: int, N_g: int, shaft_angle_deg: float = 90) -> tuple:
        """
        Cone angles (pitch angles) for pinion and gear (Eq. 15-24).

        Args:
            N_p: Pinion teeth
            N_g: Gear teeth
            shaft_angle_deg: Shaft angle (degrees, default 90)
        """
        shaft_angle_rad = shaft_angle_deg * pi / 180
        gamma_p = atan(N_p / N_g * tan(shaft_angle_rad / 2))
        gamma_g = atan(N_g / N_p * tan(shaft_angle_rad / 2))
        return gamma_p, gamma_g

    @staticmethod
    def face_width_max(A_0: float, P_d: float) -> float:
        """
        Maximum recommended face width

        Args:
            A_0: Outer cone distance
            P_d: Outer transverse diametral pitch (teeth/inch)
        """
        return min(0.3 * A_0, 10 / P_d)

    def K_o(
        *,
        power_source: Literal["uniform", "light_shock", "medium_shock", "heavy_shock"],
        driven_machine: Literal[
            "uniform", "light_shock", "medium_shock", "heavy_shock"
        ],
    ) -> float:
        """
        Overload factor K_o

        Args:
            power_source: Character of prime mover
            driven_machine: Character of load on driven machine
        """
        table = {
            "uniform": {
                "uniform": 1.00,
                "light_shock": 1.25,
                "medium_shock": 1.50,
                "heavy_shock": 1.75,
            },
            "light_shock": {
                "uniform": 1.10,
                "light_shock": 1.35,
                "medium_shock": 1.60,
                "heavy_shock": 1.85,
            },
            "medium_shock": {
                "uniform": 1.25,
                "light_shock": 1.50,
                "medium_shock": 1.75,
                "heavy_shock": 2.00,
            },
            "heavy_shock": {
                "uniform": 1.50,
                "light_shock": 1.75,
                "medium_shock": 2.00,
                "heavy_shock": 2.25,
            },
        }
        return table[power_source][driven_machine]

    def K_v(
        *,
        V: Velocity,
        Q_v: int,
    ) -> float:
        """Dynamic Factor

        Args:
            V (Velocity): Pitch line velocity
            Q_v (int): Quality number
        """

        return Spur.K_v(V=V, Q_v=Q_v)

    def V_t_max(*, Q_v: int) -> Velocity:
        """Maximum pitch line velocity"""

        return Spur.V_t_max(Q_v=Q_v)

    def K_s(*, P_d: InverseLength) -> float:
        """
        Size factor for bending K_s

        Args:
            P_d: Outer transverse diametral pitch
        """
        P_d_mag = P_d.to(InverseLengthUnit.PER_INCH).magnitude

        if P_d_mag <= 16:
            return 0.4867 + 0.2132 / P_d_mag
        else:
            return 0.5

    def K_mb(
        *,
        no_of_straddling_members: int,
    ) -> float:
        """
        Mesh basis factor K_mb

        Args:
            no_of_straddling_members: Number of straddling members (0, 1, or 2)
        """
        if no_of_straddling_members == 2:
            return 1.0
        elif no_of_straddling_members == 1:
            return 1.10
        elif no_of_straddling_members == 0:
            return 1.25
        else:
            raise ValueError("Number of straddling members must be 0, 1, or 2")

    def K_m(*, F: Length, K_mb: float) -> float:
        """
        Load distribution factor K_m

        Args:
            F: Face width
            K_mb: Mesh basis factor (1.00 both straddle, 1.10 one straddle, 1.25 neither)
        """

        F_mag = F.to(LengthUnit.INCH).magnitude

        return K_mb + 0.0036 * (F_mag**2)

    def C_s(*, F: Length) -> float:
        """
        Size factor for pitting resistance C_s

        Args:
            F: Face width
        """

        F_mag = F.to(LengthUnit.INCH).magnitude

        if F_mag < 0.5:
            return 0.5
        elif F_mag <= 4.5:
            return 0.125 * F_mag + 0.4375
        else:
            return 1.0

    def C_xc(*, is_crowned: bool) -> float:
        """
        Crowning factor for pitting C_xc

        Args:
            is_crowned: True for properly crowned teeth, False for uncrowned
        """
        return 1.5 if is_crowned else 2.0

    def K_x() -> float:
        """
        Lengthwise curvature factor for bending K_x (Eq. 15-13).
        For straight-bevel gears, K_x = 1.0
        """
        return 1.0

    def K_T(*, t: Temperature) -> float:
        """
        Temperature factor KT

        Args:
            T_f: Gear blank or oil temperature (°F, default 70°F)
        """

        t_mag = t.to(TemperatureUnit.FAHRENHEIT).magnitude

        if t_mag <= 250:
            return 1.0
        else:
            return (460 + t_mag) / (460 + 250)

    def K_R(*, R: float) -> float:
        """
        Reliability factor for bending KR

        Args:
            R: Reliability (0.50 to 0.9999)
        """

        if R == 0.5:
            return 0.7
        elif R == 0.9:
            return 0.85
        else:
            if 0.9 < R < 0.99:
                return 0.7 - 0.15 * log10(1 - R)
            elif 0.99 <= R <= 0.999:
                return 0.50 - 0.25 * log10(1 - R)
            else:
                raise ValueError("Reliability R must be between 0.50 and 0.999")

    @staticmethod
    def C_R(*, R: float) -> float:
        """
        Reliability factor for pitting CR

        Args:
            R: Reliability (0.50 to 0.9999)
        """

        return np.sqrt(Bevel.K_R(R=R))

    def C_L(*, N_L: int) -> float:
        """
        Stress-cycle factor for pitting resistance CL
        For carburized case-hardened steel bevel gears.

        Args:
            N_L: Number of load cycles (pinion revolutions)
        """
        if 1e3 <= N_L < 1e4:
            return 2
        elif 1e4 <= N_L <= 1e10:
            return 3.4822 * (N_L ** (-0.0602))
        else:
            raise ValueError("N_L must be between 1e3 and 1e10")

    def K_L(*, N_L: int, is_critical: bool) -> float:
        """
        Stress-cycle factor for bending strength KL
        For carburized case-hardened steel bevel gears.

        Args:
            N_L: Number of load cycles (pinion revolutions)
            is_critical: Use critical region coefficients if True
        """
        if 1e2 <= N_L < 1e3:
            return 2.7
        elif 1e3 <= N_L < 3e6:
            return 6.1514 * (N_L ** (-0.1192))
        elif 3e6 <= N_L <= 1e10:
            if is_critical:
                return 1.683 * (N_L ** (-0.0323))
            else:
                return 1.3558 * (N_L ** (-0.0178))
        else:
            raise ValueError("N_L must be between 1e2 and 1e10")

    def C_H(
        *,
        m_G: float,
        H_B_P: float | None,
        f_P: float | None,
        H_B_G: float,
        is_pinion: bool,
    ):
        """Hardness Ratio Factor for Pitting Resistance

        Args:
            N_P (int): Pinion teeth
            N_G (int): Gear teeth
            H_BP (float): Brinell Hardness of Pinion
            f_P (float): Surface finish factor of Pinion
            H_BG (float): Brinell Hardness of Gear
        """

        return Spur.C_H(
            m_G=m_G,
            H_B_P=H_B_P,
            f_P=f_P,
            H_B_G=H_B_G,
            is_pinion=is_pinion,
        )

    def J(*, N_pinion: int, N_gear: int) -> tuple:
        """
        Bending strength geometry factor J for pinion and gear.
        Interpolated from AGMA 2003-B97 Figure 15-7 (20° pressure angle, 90° shaft).

        Args:
            N_pinion: Number of pinion teeth
            N_gear: Number of gear teeth
        """
        gear_teeth = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

        J_table = {
            13: [
                0.200,
                0.225,
                0.248,
                0.263,
                0.275,
                0.286,
                0.295,
                0.303,
                0.310,
                0.323,
                0.334,
                0.343,
                0.351,
                0.358,
            ],
            16: [
                0.210,
                0.240,
                0.265,
                0.280,
                0.293,
                0.304,
                0.313,
                0.321,
                0.328,
                0.341,
                0.352,
                0.361,
                0.369,
                0.377,
            ],
            19: [
                0.216,
                0.250,
                0.278,
                0.293,
                0.308,
                0.319,
                0.329,
                0.337,
                0.344,
                0.358,
                0.369,
                0.379,
                0.387,
                0.395,
            ],
            20: [
                0.217,
                0.251,
                0.279,
                0.295,
                0.310,
                0.321,
                0.331,
                0.340,
                0.348,
                0.362,
                0.373,
                0.382,
                0.391,
                0.399,
            ],
            25: [
                0.222,
                0.258,
                0.287,
                0.305,
                0.321,
                0.333,
                0.343,
                0.352,
                0.360,
                0.376,
                0.388,
                0.398,
                0.407,
                0.415,
            ],
            30: [
                0.225,
                0.263,
                0.295,
                0.314,
                0.331,
                0.343,
                0.354,
                0.363,
                0.372,
                0.388,
                0.401,
                0.412,
                0.421,
                0.430,
            ],
        }

        if N_pinion not in J_table:
            closest_N = min(J_table.keys(), key=lambda x: abs(x - N_pinion))
            row = J_table[closest_N]
        else:
            row = J_table[N_pinion]

        J_p = np.interp(N_pinion, gear_teeth, row)
        J_g = np.interp(N_gear, gear_teeth, row)

        return J_p, J_g

    def I(*, N_pinion: int, N_gear: int) -> float:
        """
        Pitting resistance geometry factor I for straight-bevel gears.
        Simplified from AGMA 2003-B97 Figure 15-6 (20° pressure angle, 90° shaft).

        Args:
            N_pinion: Number of pinion teeth
            N_gear: Number of gear teeth
        """
        I_table = {
            (10, 10): 0.115,
            (10, 20): 0.105,
            (10, 30): 0.095,
            (10, 40): 0.085,
            (13, 13): 0.125,
            (13, 26): 0.108,
            (13, 39): 0.095,
            (13, 52): 0.087,
            (16, 16): 0.130,
            (16, 32): 0.110,
            (16, 48): 0.095,
            (16, 64): 0.088,
            (20, 20): 0.140,
            (20, 40): 0.115,
            (20, 60): 0.098,
            (20, 80): 0.088,
            (25, 25): 0.145,
            (25, 50): 0.118,
            (25, 75): 0.100,
            (25, 100): 0.090,
            (30, 30): 0.150,
            (30, 60): 0.120,
            (30, 90): 0.102,
            (30, 120): 0.092,
        }

        key = (N_pinion, N_gear)
        if key in I_table:
            return I_table[key]
        else:
            m_G = N_gear / N_pinion
            if m_G == 1.0:
                return 0.08 + 0.0003 * (N_pinion + N_gear)
            else:
                return (0.08 + 0.0003 * (N_pinion + N_gear)) / sqrt(
                    1 + 0.5 * (m_G - 1) ** 2
                )

    def S_t(*, H_B: float, grade: int, is_through_hardened: bool) -> Stress:
        """
        Bending strength number S_t for steel gears

        Args:
            H_B: Brinell Hardness Number
            grade: Gear quality grade (1 or 2)
            is_through_hardened: True for through-hardened, False for case-hardened
        """
        if is_through_hardened:
            if grade == 1:
                m, c = 44, 2100
            elif grade == 2:
                m, c = 48, 5980
            return Stress((m * H_B + c), StressUnit.PSI)
        else:
            return Stress(22000, StressUnit.PSI)

    def S_c(*, H_B: float, grade: int, is_through_hardened: bool) -> Stress:
        """
        Contact strength number S_c for steel gears

        Args:
            H_B: Brinell Hardness Number
            grade: Gear quality grade (1 or 2)
        """
        if is_through_hardened:
            if grade == 1:
                m, c = 341, 23620
            elif grade == 2:
                m, c = 263.6, 29560
            return Stress((m * H_B + c), StressUnit.PSI)
        else:
            return Stress(145000, StressUnit.PSI)

    def s_t(
        *,
        W_t: Force,
        F: Length,
        P_d: InverseLength,
        K_o: float,
        K_v: float,
        K_s: float,
        K_m: float,
        K_x: float,
        J: float,
    ) -> Stress:
        """
        Bending stress at large end of tooth

        Args:
            W_t: Tangential load
            F: Face width
            P_d: Outer transverse diametral pitch
            K_o: Overload factor
            K_v: Dynamic factor
            K_s: Size factor for bending
            K_m: Load distribution factor
            K_x: Lengthwise curvature factor (1.0 for straight bevel)
            J: Bending strength geometry factor
        """

        W_t_mag = W_t.to(ForceUnit.LBF).magnitude
        F_mag = F.to(LengthUnit.INCH).magnitude
        P_d_mag = P_d.to(InverseLengthUnit.PER_INCH).magnitude

        s_t_mag = (W_t_mag * K_o * K_v * K_s * K_m * P_d_mag) / (K_x * F_mag * J)
        return Stress(s_t_mag, StressUnit.PSI)

    def s_c(
        W_t: Force,
        F: Length,
        d_p: Length,
        K_o: float,
        K_v: float,
        K_m: float,
        C_s: float,
        C_xc: float,
        I: float,
        C_p: SqrtPressure,
    ) -> Stress:
        """
        Contact (pitting) stress at large end of tooth

        Args:
            W_t: Tangential load (lbf)
            F: Face width
            d_p: Pinion pitch diameter at large end
            K_o: Overload factor
            K_v: Dynamic factor
            K_m: Load distribution factor
            C_s: Size factor for pitting
            C_xc: Crowning factor for pitting
            I: Geometry factor for pitting resistance
            C_p: Elastic coefficient (psi^0.5, default 2290 for steel)
        """
        C_p_mag = C_p.to(SqrtPressureUnit.SQRT_PSI).magnitude
        W_t_mag = W_t.to(ForceUnit.LBF).magnitude
        F_mag = F.to(LengthUnit.INCH).magnitude
        d_p_mag = d_p.to(LengthUnit.INCH).magnitude

        s_c_mag = C_p_mag * sqrt(
            (W_t_mag * K_o * K_v * K_m * C_s * C_xc) / (F_mag * d_p_mag * I)
        )
        return Stress(s_c_mag, StressUnit.PSI)

    def s_b_all(
        *, S_t: Stress, K_L: float, K_T: float, K_R: float, S_F: float
    ) -> Stress:
        """
        Allowable bending stress

        Args:
            S_t: Bending strength number
            K_L: Stress-cycle factor for bending
            K_T: Temperature factor
            K_R: Reliability factor for bending
        """

        S_t_mag = S_t.to(StressUnit.PSI).magnitude

        s_t_allowable_mag = (S_t_mag * K_L) / (S_F * K_T * K_R)
        return Stress(s_t_allowable_mag, StressUnit.PSI)

    def s_c_all(
        *, S_c: Stress, C_L: float, C_H: float, K_T: float, C_R: float, S_H: float
    ) -> Stress:
        """
        Allowable contact stress s_c_allowable

        Args:
            S_c: Contact strength number
            C_L: Stress-cycle factor for pitting
            C_H: Hardness ratio factor for pitting
            K_T: Temperature factor
            C_R: Reliability factor for pitting
            S_H: Safety factor for pitting
        """

        S_c_mag = S_c.to(StressUnit.PSI).magnitude

        s_c_allowable_mag = (S_c_mag * C_L * C_H) / (S_H * K_T * C_R)
        return Stress(s_c_allowable_mag, StressUnit.PSI)

    def safety_factor_bending(*, s_t_allowable: Stress, s_t: Stress) -> float:
        """
        Safety factor for bending S_F

        Args:
            s_t_allowable: Allowable bending stress
            s_t: Bending stress
        """

        s_t_allowable_mag = s_t_allowable.to(StressUnit.PSI).magnitude
        s_t_mag = s_t.to(StressUnit.PSI).magnitude

        return s_t_allowable_mag / s_t_mag

    def safety_factor_pitting(*, s_c_allowable: Stress, s_c: Stress) -> float:
        """
        Safety factor for pitting S_H

        Args:
            s_c_allowable: Allowable contact stress
            s_c: Contact stress
        """

        s_c_allowable_mag = s_c_allowable.to(StressUnit.PSI).magnitude
        s_c_mag = s_c.to(StressUnit.PSI).magnitude

        return s_c_allowable_mag / s_c_mag


'''
    def power_rating_bending(
        N_p: int,
        N_g: int,
        P_d: float,
        F: float,
        n_p: float,
        HB_p: float,
        HB_g: float,
        grade: int = 1,
        is_crowned: bool = True,
        Q_v: int = 6,
        R: float = 0.99,
        N_L: int = 1e7,
        K_o: float = 1.0,
        SF: float = 1.0,
    ) -> float:
        """
        Rate gearset for power based on bending strength (pinion).

        Args:
            N_p: Pinion teeth
            N_g: Gear teeth
            P_d: Outer transverse diametral pitch (teeth/inch)
            F: Face width
            n_p: Pinion speed (rpm)
            HB_p: Pinion Brinell Hardness
            HB_g: Gear Brinell Hardness
            grade: Gear quality grade (1 or 2)
            is_crowned: Whether teeth are crowned
            Q_v: Transmission accuracy number
            R: Reliability (0.50 to 0.9999)
            N_L: Number of load cycles (pinion revolutions)
            K_o: Overload factor
            SF: Safety factor for bending
        """
        d_p = Bevel.d(N_p, P_d)
        d_g = Bevel.d(N_g, P_d)
        V_t = Bevel.V(d_p, n_p)

        J_p, J_g = Bevel.J(N_p, N_g)

        Kv_val = Bevel.K_v(V_t, Q_v)
        Ks_val = Bevel.K_s(P_d)
        K_mb = 1.25
        Km_val = Bevel.K_m(F, K_mb)
        Kx_val = Bevel.K_x()
        KT_val = Bevel.K_T()
        KR_val = Bevel.K_R(R)
        KL_val = Bevel.K_L(int(N_L))

        St_p = Bevel.S_t(HB_p, grade, True)
        s_t_all = St_p * KL_val * SF * KT_val * KR_val

        W_t_perm = (s_t_all * P_d * F * J_p) / (K_o * Kv_val * Ks_val * Km_val * Kx_val)

        H_bending = W_t_perm * V_t / 33000

        return H_bending

    def power_rating_wear(
        N_p: int,
        N_g: int,
        P_d: float,
        F: float,
        n_p: float,
        HB_p: float,
        HB_g: float,
        grade: int = 1,
        is_crowned: bool = True,
        Q_v: int = 6,
        R: float = 0.99,
        N_L: int = 1e7,
        K_o: float = 1.0,
        SH: float = 1.0,
        C_p: float = 2290,
    ) -> float:
        """
        Rate gearset for power based on pitting (wear) resistance.

        Args:
            N_p: Pinion teeth
            N_g: Gear teeth
            P_d: Outer transverse diametral pitch (teeth/inch)
            F: Face width
            n_p: Pinion speed (rpm)
            HB_p: Pinion Brinell Hardness
            HB_g: Gear Brinell Hardness
            grade: Gear quality grade (1 or 2)
            is_crowned: Whether teeth are crowned
            Q_v: Transmission accuracy number
            R: Reliability (0.50 to 0.9999)
            N_L: Number of load cycles (pinion revolutions)
            K_o: Overload factor
            SH: Safety factor for pitting
            C_p: Elastic coefficient (psi^0.5, default 2290 for steel)
        """
        d_p = Bevel.d(N_p, P_d)
        V_t = Bevel.V(d_p, n_p)

        I_val = Bevel.I(N_p, N_g)

        Kv_val = Bevel.K_v(V_t, Q_v)
        Cs_val = Bevel.C_s(F)
        K_mb = 1.25
        Km_val = Bevel.K_m(F, K_mb)
        Cxc_val = Bevel.C_xc(is_crowned)
        KT_val = Bevel.K_T()
        CR_val = Bevel.C_R(R)
        CL_val = Bevel.C_L(int(N_L))
        CH_val = Bevel.C_H(HB_p, HB_g, is_pinion=False)

        Sc_p = Bevel.S_c(HB_p, grade)
        s_c_all = Sc_p * CL_val * CH_val * SH * KT_val * CR_val

        W_t_perm = (s_c_all**2 * F * d_p * I_val) / (
            C_p**2 * K_o * Kv_val * Km_val * Cs_val * Cxc_val
        )

        H_wear = W_t_perm * V_t / 33000

        return H_wear
'''
