from math import atan, atan2, cos, exp, log, log10, pi, sin, sqrt, tan
from typing import Literal

import numpy as np
import scipy

from pydome.gears.spur import Spur
from pydome.units import *  # noqa: F403


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

        # arr[gear_teeth] = [(pinion_teeth, I), ...]
        arr = {
            15: [(12.785299806576402, 0.05662180349932706), (13.017408123791103, 0.05678331090174967), (13.249516441005802, 0.05686406460296098), (13.481624758220503, 0.05694481830417228), (13.79110251450677, 0.05694481830417228), (14.100580270793037, 0.05694481830417228), (14.410058027079305, 0.05694481830417228), (14.719535783365572, 0.05694481830417228), (15.029013539651839, 0.05686406460296098)],
            20: [(12.86266924564797, 0.06211305518169583), (13.868471953578336, 0.0630820995962315), (14.951644100580271, 0.06356662180349934), (16.344294003868473, 0.06356662180349934), (17.42746615087041, 0.06332436069986541), (18.27852998065764, 0.0630013458950202), (18.974854932301742, 0.06259757738896367), (19.593810444874276, 0.06219380888290714), (20.058027079303677, 0.06187079407806192)],
            25: [(13.017408123791103, 0.06485868102288023), (14.100580270793037, 0.06663526244952894), (14.951644100580271, 0.0676043068640646), (16.034816247582206, 0.06849259757738896), (17.040618955512574, 0.06881561238223419), (18.27852998065764, 0.06873485868102289), (19.98065764023211, 0.06816958277254374), (21.914893617021278, 0.06703903095558547), (25.241779497098648, 0.06493943472409153)],
            30: [(13.094777562862669, 0.06720053835800807), (14.25531914893617, 0.06913862718707942), (14.951644100580271, 0.07002691790040377), (16.112185686653774, 0.07123822341857336), (17.35009671179884, 0.07212651413189772), (18.665377176015475, 0.07253028263795425), (20.599613152804643, 0.07261103633916555), (24.08123791102515, 0.07091520861372813), (30.038684719535784, 0.06695827725437417)],
            35: [(12.785299806576402, 0.0675235531628533), (13.79110251450677, 0.07002691790040377), (14.951644100580271, 0.07220726783310902), (16.344294003868473, 0.07390309555854643), (18.355899419729205, 0.07527590847913863), (20.522243713733076, 0.07567967698519515), (23.30754352030948, 0.07535666218034993), (27.098646034816248, 0.0734993270524899), (34.990328820116055, 0.06800807537012113)],
            40: [(12.785299806576402, 0.06881561238223419), (13.868471953578336, 0.07172274562584119), (14.951644100580271, 0.07366083445491252), (16.88588007736944, 0.0762449528936743), (19.903288201160542, 0.07810228802153432), (23.152804642166345, 0.07842530282637955), (25.009671179883945, 0.07802153432032302), (29.96131528046422, 0.07559892328398385), (39.70986460348163, 0.06938088829071333)],
            45: [(12.785299806576402, 0.06954239569313594), (14.177949709864603, 0.07301480484522208), (16.266924564796906, 0.07697173620457605), (19.98065764023211, 0.08020188425302827), (22.99806576402321, 0.08109017496635262), (25.009671179883945, 0.08109017496635262), (27.33075435203095, 0.0806056527590848), (34.990328820116055, 0.0763257065948856), (45.1257253384913, 0.06986541049798116)],
            50: [(12.785299806576402, 0.07026917900403769), (14.951644100580271, 0.07543741588156125), (18.355899419729205, 0.08004037685060565), (22.53384912959381, 0.08254374158815612), (25.009671179883945, 0.08286675639300135), (27.485493230174082, 0.08270524899057874), (32.12765957446808, 0.08117092866756394), (43.036750483559, 0.07503364737550472), (50.0, 0.07261103633916555)],
            60: [(12.785299806576402, 0.07067294751009422), (14.951644100580271, 0.0762449528936743), (17.35009671179884, 0.08004037685060565), (19.903288201160542, 0.08270524899057874), (25.009671179883945, 0.08545087483176313), (29.96131528046422, 0.08633916554508748), (32.90135396518376, 0.08633916554508748), (40.01934235976789, 0.08480484522207268), (50.0, 0.08189771197846568)],
            70: [(12.785299806576402, 0.07269179004037685), (14.951644100580271, 0.07850605652759085), (18.046421663442942, 0.08383580080753701), (21.914893617021278, 0.08835800807537013), (27.253384912959383, 0.09174966352624496), (32.66924564796905, 0.09304172274562585), (36.22823984526113, 0.09288021534320323), (42.340425531914896, 0.09166890982503365), (50.0, 0.08989232839838493)],
            80: [(12.785299806576402, 0.0764872139973082), (14.951644100580271, 0.08262449528936744), (17.81431334622824, 0.08835800807537013), (23.384912959381047, 0.09506056527590848), (29.96131528046422, 0.09869448183041724), (34.990328820116055, 0.09942126514131898), (39.94197292069633, 0.09893674293405115), (45.04835589941973, 0.09772543741588156), (50.0, 0.09643337819650068)],
            90: [(13.172147001934237, 0.07907133243607), (14.951644100580271, 0.08545087483176313), (18.201160541586074, 0.09288021534320323), (24.39071566731141, 0.10006729475100942), (29.96131528046422, 0.10305518169582772), (34.990328820116055, 0.10394347240915208), (40.01934235976789, 0.10353970390309555), (44.970986460348165, 0.10281292059219381), (50.0, 0.10192462987886945)],
            100: [(13.172147001934237, 0.08189771197846568), (14.951644100580271, 0.0879542395693136), (17.969052224371374, 0.09506056527590848), (21.06382978723404, 0.10006729475100942), (25.009671179883945, 0.10418573351278601), (30.038684719535784, 0.10725437415881561), (34.990328820116055, 0.1085464333781965), (39.94197292069633, 0.1084656796769852), (50.0, 0.1070121130551817)],
        }  # fmt:off

        points = []
        values = []
        for N_gear_val, vals in arr.items():
            for N_pinion_val, I_val in vals:
                points.append((N_pinion_val, N_gear_val))
                values.append(I_val)

        interpolator = scipy.interpolate.LinearNDInterpolator(points, values)

        I_interp = interpolator(N_pinion, N_gear)
        if I_interp is None:
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
            _, idx = tree.query([(N_pinion, N_gear)], k=1)
            I_interp = values[idx[0]]
        return float(I_interp)

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
