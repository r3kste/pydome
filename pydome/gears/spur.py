from math import cos, exp, log, sin, sqrt
from typing import Literal

import numpy as np

from pydome.constants import pi
from pydome.elements import Gear
from pydome.units import *  # noqa: F403


class Spur:
    """Spur Gear"""

    def d(
        *,
        N: int,
        P: InverseLength,
    ) -> Length:
        """Pitch Diameter

        Args:
            N (int): Number of teeth
            P (InverseLength): Diametral pitch
        """

        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude
        d_mag = N / P_mag
        return Length(d_mag, LengthUnit.INCH)

    def V(
        *,
        d: Length,
        n: AngularVelocity,
    ) -> Velocity:
        """Pitch Line Velocity

        Args:
            d (Length): Pitch diameter
            n (AngularVelocity): Rotational speed
        """

        d_mag = d.to(LengthUnit.INCH).magnitude
        n_mag = n.to(AngularVelocityUnit.RPM).magnitude
        V_mag = (pi * d_mag * n_mag) / 12
        return Velocity(abs(V_mag), VelocityUnit.FPM)

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

        V_mag = V.to(VelocityUnit.FPM).magnitude

        B = 0.25 * ((12 - Q_v) ** (2 / 3))
        A = 50 + 56 * (1 - B)

        return ((A + sqrt(V_mag)) / A) ** B

    def V_t_max(
        *,
        Q_v: int,
    ) -> Velocity:
        """Maximum recommended pitch line velocity

        Args:
            Q_v (int): Quality number
        """

        B = 0.25 * ((12 - Q_v) ** (2 / 3))
        A = 50 + 56 * (1 - B)

        V_mag = (A + (Q_v - 3)) ** 2
        return Velocity(V_mag, VelocityUnit.FPM)

    def K_o(
        *,
        power_source: Literal["uniform", "light_shock", "medium_shock"],
        driven_machine: Literal["uniform", "moderate_shock", "heavy_shock"],
    ) -> float:
        """Overload Factor

        Table:
            14-9
        """

        idx1 = ["uniform", "light_shock", "medium_shock"].index(power_source)
        idx2 = ["uniform", "moderate_shock", None, "heavy_shock"].index(driven_machine)

        return 1.0 + 0.25 * (idx1 + idx2)

    def Y(
        *,
        N: int,
    ) -> float:
        """Lewis Form Factor

        Args:
            N (int): Number of teeth

        Chart:
            Figure 14-6
        """
        _teeth = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 34,
                38, 43, 50, 60, 75, 100, 150, 300, 400, 1e308]  # fmt: off
        _vals = [0.245, 0.261, 0.277, 0.290, 0.296, 0.303, 0.309, 0.314, 0.322,
                0.328, 0.331, 0.337, 0.346, 0.353, 0.359, 0.371, 0.384, 0.397,
                0.409, 0.422, 0.435, 0.447, 0.460, 0.472, 0.480, 0.485]  # fmt: off

        return np.interp(N, _teeth, _vals)

    def K_s(
        *,
        F: Length,
        Y: float,
        P: InverseLength,
    ) -> float:
        """Size Factor

        Args:
            F (Length): Face width
            N (int): Number of teeth
            P (InverseLength): Diametral pitch

        Equation:
            14-29
        """

        F_mag = F.to(LengthUnit.INCH).magnitude
        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

        return 1.192 * ((F_mag * sqrt(Y)) / P_mag) ** 0.0535

    def K_m(
        *,
        C_mc: float,
        C_pf: float,
        C_pm: float,
        C_ma: float,
        C_e: float,
    ) -> float:
        """Load Distribution Factor

        Args:
            is_crowned (bool): Whether the gear is crowned
            F (Length): Face width
            d_P (Length): Pinion pitch diameter
            S_1 (float):
            S (float):
            gearing_condition (Literal): Gear environment.
            is_gear_adjusted (bool, default=False): Whether the gearing has been
                adjusted at assembly, or compatibility is improved by lapping,
                or both

        Assumptions:
            - Net face width to pinion pitch diameter ratio less than 2.0
            - Gear elements mounted between the bearings
            - Face widths upto 40 inches
            - Contact across the full face width of the narrowest member
        """

        return 1 + C_mc * (C_pf * C_pm + C_ma * C_e)

    def C_mc(
        *,
        is_crowned: bool | None,
    ) -> float:
        """Lead (Crown) Correction Factor"""
        return 0.8 if is_crowned else 1.0

    def C_pf(
        *,
        F: Length,
        d_P: Length,
    ) -> float:
        """Pinion Proportion Factor

        Args:
            F (Length): Face width
            d_P (Length): Pinion pitch diameter
        """

        F_mag = F.to(LengthUnit.INCH).magnitude
        d_P_mag = d_P.to(LengthUnit.INCH).magnitude

        if F_mag <= 1:
            return F_mag / (10 * d_P_mag) - 0.025
        elif 1 < F_mag <= 17:
            return F_mag / (10 * d_P_mag) - 0.0375 + 0.0125 * F_mag
        elif F_mag > 17:
            return (
                F_mag / (10 * d_P_mag) - 0.1109 + 0.0207 * F_mag - 0.000228 * F_mag**2
            )

    def C_pm(*, S_1: float | None, S: float | None) -> float:
        """Mesh Proportion Modifier

        Args:
            S_1 (float): offset of the pinion; i.e, the distance from the bearing span
                centerline to the pinion mid-face.
            S (float): bearing span; i.e, the distance between the bearing
                center lines.
        """
        if S_1 is None or S is None:
            return 1.0

        if S_1 / S < 0.175:
            return 1.0
        else:
            return 1.1

    _empirical_constants_for_C_ma: dict[str, dict[str, float]] = {
        "open": {"A": 0.247, "B": 0.0167, "C": -0.765 * 10**-4},
        "enclosed_commercial": {"A": 0.127, "B": 0.0158, "C": -0.930 * 10**-4},
        "enclosed_precise": {"A": 0.0675, "B": 0.0128, "C": -0.926 * 10**-4},
        "enclosed_extra_precise": {"A": 0.0036, "B": 0.0102, "C": -0.822 * 10**-4},
    }

    def C_ma(
        *,
        F: Length,
        gearing_condition: Literal[
            "open", "enclosed_commercial", "enclosed_precise", "enclosed_extra_precise"
        ],
    ) -> float:
        """Mesh Alignment Factor"""

        F_mag = F.to(LengthUnit.INCH).magnitude

        constants = Spur._empirical_constants_for_C_ma[gearing_condition]
        A = constants["A"]
        B = constants["B"]
        C = constants["C"]

        return A + B * F_mag + C * F_mag**2

    def C_e(
        *,
        is_gear_adjusted: bool,
    ) -> float:
        """Mesh Alignment Correction Factor"""
        return 0.8 if is_gear_adjusted else 1.0

    def K_B(
        *,
        t_R: Length | None,
        h_t: Length | None,
    ) -> float:
        """Rim Thickness Factor

        Args:
            t_R (Length): Rim thickness below the tooth
            h_t (Length): Tooth height

        Figure:
            14-16

        Source:
            ANSI/AGMA 2001-D04
        """

        if t_R is None or h_t is None:
            return 1.0

        t_R_mag = t_R.to(LengthUnit.INCH).magnitude
        h_t_mag = h_t.to(LengthUnit.INCH).magnitude

        # Backup ratio
        m_B = t_R_mag / h_t_mag

        if m_B >= 1.2:
            return 1.0
        else:
            return 1.6 * log(2.242 / m_B)

    def K_T(
        *,
        T: Temperature | None,
    ) -> float:
        """Temperature Factor

        Args:
            T (Temperature): Oil or gear-blank temperature
        """
        if T is None:
            return 1.0

        T_mag = T.to(TemperatureUnit.FAHRENHEIT).magnitude

        if T_mag <= 250:
            return 1.0
        else:
            raise NotImplementedError(
                "Temperature factor for T > 250 F is not implemented."
            )

    _reliability_factors: dict[float, float] = {
        0.50: 0.7,
        0.90: 0.85,
        0.99: 1,
        0.999: 1.25,
        0.9999: 1.5,
    }

    def K_R(
        *,
        R: float,
    ) -> float:
        """Reliability Factor"""

        if R in Spur._reliability_factors:
            return Spur._reliability_factors[R]
        else:
            if 0.5 < R < 0.99:
                return 0.658 - 0.0759 * log(1 - R)
            elif 0.99 <= R <= 0.9999:
                return 0.5 - 0.109 * log(1 - R)
            else:
                raise ValueError("R must be between 0.5 and 0.9999")

    def Y_N(
        *,
        L: int,
        H_B: float | Literal["case", "nitrided"],
        upper: bool = True,
        A: float | None = None,
        B: float | None = None,
    ) -> float:
        """Stress Cycle Factor for Bending Stress

        Args:
            L (int): Number of load cycles.
            hardness (float | Literal["case", "nitrided"]): Surface hardness in Brinell Hardness Number or
                "case" for case hardened or "nitrided" for nitrided gears.
            upper (bool, default=True): If True, use upper bound values.
            A (float | None, default=None): Multiplicative factor.
            B (float | None, default=None): Exponential factor.

        Figure:
            14-14

        """

        if A is not None and B is not None:
            return A / (L**B)

        if L > 3e6:
            if upper:
                A, B = 1.3558, 0.0178
            else:
                A, B = 1.6831, 0.0323

            return A / (L**B)

        L = max(L, 1e3)
        if H_B == "case":
            C = 6.1514 / (1e3**0.1192)
        elif H_B == "nitrided":
            C = 3.517 / (1e3**0.0817)
        else:
            _Y = np.array(
                [2.3194 / 1000**0.0538, 4.9404 / 1000**0.1045, 9.4518 / 1000**0.148]
            )
            _X = np.array([160, 250, 400])
            C = np.interp(H_B, _X, _Y)

        _convergent = 1.0397299315867656
        B = log(C / _convergent, 3e3)
        A = C * (1e3**B)
        return A / (L**B)

    def Z_N(
        *,
        L: int,
        is_nitrided: bool,
        A: float | None = None,
        B: float | None = None,
    ) -> float:
        """Stress Cycle Factor for Contact Stress

        Args:
            L (int): Number of load cycles
            is_nitrided (bool): True if nitrided surface, False otherwise
            A (float | None, default=None): Multiplicative factor.
            B (float | None, default=None): Exponential factor.

        Figure:
            14-15
        """

        if A is not None and B is not None:
            return A / (L**B)

        if L < 1e4:
            if is_nitrided:
                return 1.1
            else:
                return 1.5

        if is_nitrided:
            if L < 1e7:
                A, B = 1.249, 0.0138
            else:
                A, B = 1.4488, 0.023
        else:
            A, B = 2.466, 0.056

        return A / (L**B)

    def J(
        *,
        N: int,
        other_N: int,
    ) -> float:
        """Geometry Factor for Bending Strength

        Args:
            N (int): Number of teeth for which geometry factor is desired
            other_N (int): Number of teeth of mating gear

        Chart:
            Figure 14-6

        Source:
            melib

        """
        x = np.array([12, 15, 17, 20, 24, 30, 35, 40, 45, 50, 60, 80, 125, 275, 1e9])
        y = np.array([17, 25, 35, 50, 85, 170, 1000])

        # a[row,col] is the J for N1=x[col] and N2=y[row]
        a = np.array(
            [
                #  12     15     17     20     24     30     35     40     45     50     60     80    125    275    np.inf
                [0.209, 0.250, 0.293, 0.313, 0.334, 0.358, 0.371, 0.380, 0.388, 0.395, 0.407, 0.418, 0.432, 0.446, 0.455],  # 17
                [0.209, 0.250, 0.296, 0.320, 0.341, 0.366, 0.379, 0.391, 0.399, 0.407, 0.417, 0.430, 0.445, 0.461, 0.472],  # 25
                [0.209, 0.250, 0.296, 0.326, 0.349, 0.375, 0.389, 0.401, 0.409, 0.416, 0.429, 0.442, 0.458, 0.472, 0.486],  # 35
                [0.209, 0.250, 0.296, 0.330, 0.355, 0.382, 0.397, 0.411, 0.418, 0.428, 0.441, 0.455, 0.472, 0.488, 0.501],  # 50
                [0.209, 0.250, 0.295, 0.338, 0.362, 0.389, 0.405, 0.418, 0.428, 0.436, 0.450, 0.466, 0.484, 0.501, 0.514],  # 85
                [0.209, 0.250, 0.295, 0.342, 0.368, 0.399, 0.416, 0.429, 0.439, 0.449, 0.463, 0.480, 0.500, 0.518, 0.533],  # 170
                [0.209, 0.250, 0.296, 0.346, 0.374, 0.404, 0.422, 0.437, 0.447, 0.458, 0.472, 0.491, 0.511, 0.530, 0.546],  # 1000
            ]
        )  # fmt: off

        from scipy.interpolate import RegularGridInterpolator

        fp = RegularGridInterpolator((y, x), a, method="linear")
        return float(fp((other_N, N)))

    def m_N() -> float:
        """Tooth Load-Sharing Ratio

        Assumptions:
            - Spur gears

        """
        return 1.0

    def C_p(
        *,
        mu_P: float,
        E_P: Stress,
        mu_G: float,
        E_G: Stress,
    ) -> SqrtPressure:
        """Elastic Coefficient

        Args:
            mu_P (float): Poisson's ratio of pinion
            E_P (float): Modulus of elasticity of pinion
            mu_G (float): Poisson's ratio of gear
            E_G (float): Modulus of elasticity of gear

        Equation:
            14-13

        Table:
            Table 14-8
        """

        E_P_mag = E_P.to(StressUnit.PSI).magnitude
        E_G_mag = E_G.to(StressUnit.PSI).magnitude

        return SqrtPressure(
            sqrt(1 / (np.pi * ((1 - mu_P**2) / E_P_mag + (1 - mu_G**2) / E_G_mag))),
            SqrtPressureUnit.SQRT_PSI,
        )

    def C_f() -> float:
        """Surface Condition Factor"""
        return 1.0

    def m_G(
        *,
        N_P: int,
        N_G: int,
    ) -> float:
        """Gear Ratio"""
        return N_G / N_P

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

        if is_pinion:
            return 1.0

        if H_B_P is None:
            if f_P is None:
                raise ValueError(
                    "Either H_BP or f_P must be provided when is_pinion is False."
                )

            B_prime = 0.00075 * exp(-0.0112 * f_P)
            return 1 + B_prime * (450 - H_B_G)
        else:
            _hardness_ratio = H_B_P / H_B_G
            if _hardness_ratio < 1.2:
                A_prime = 0
            elif 1.2 <= _hardness_ratio <= 1.7:
                A_prime = 8.98 * 10**-3 * (H_B_P / H_B_G) - 8.29 * 10**-3
            elif _hardness_ratio > 1.7:
                A_prime = 0.00698

            return 1 + A_prime * (m_G - 1)

    def I(  # noqa: E743
        *,
        m_G: float,
        m_N: float,
        phi: Angle,
        gear_mode: Literal["external", "internal"],
    ):
        """Geometry Factor for Pitting Resistance

        Args:
            m_G (float): Gear ratio
            m_N (float): Tooth load-sharing ratio
        """
        phi_mag = phi.to(AngleUnit.RADIAN).magnitude

        if gear_mode == "external":
            _c = m_G / (m_G + 1)
        elif gear_mode == "internal":
            _c = m_G / (m_G - 1)
        return (cos(phi_mag) * sin(phi_mag) * _c) / (2 * m_N)

    def Z(
        *,
        r_P: Length,
        r_G: Length,
        phi: Angle,
        a: Length,
    ) -> float:
        """Contact Ratio

        Args:
            r_P (Length): Pitch radius of pinion
            r_G (Length): Pitch radius of gear
            phi (Angle): Pressure angle
            a (Length): Addendum radius
        """

        r_P_mag = r_P.to(LengthUnit.INCH).magnitude
        r_G_mag = r_G.to(LengthUnit.INCH).magnitude
        phi_mag = phi.to(AngleUnit.RADIAN).magnitude
        r_bP = r_P_mag * cos(phi_mag)
        r_bG = r_G_mag * cos(phi_mag)
        a_mag = a.to(LengthUnit.INCH).magnitude

        _term1 = sqrt((r_P_mag + a_mag) ** 2 - r_bP**2)
        _term2 = sqrt((r_G_mag + a_mag) ** 2 - r_bG**2)
        _term3 = (r_P_mag + r_G_mag) * sin(phi_mag)

        return _term1 + _term2 - _term3

    def S_t(
        *,
        H_B: float,
        grade: int,
        is_through_hardened: bool,
        is_nitrided: bool,
        material: Literal["chrome", "nitralloy"],
    ) -> Stress:
        """Gear Bending Strength

        Args:
            H_B (float): Brinell Hardness Number
            grade (int): Gear quality grade
            is_through_hardened (bool): True if through-hardened, False otherwise
            is_nitrided (bool): True if nitrided, False otherwise
            material (Literal["chrome", "nitralloy"]): Material type if nitrided

        Assumptions:
            - 10 million stress cycles
            - Unidirectional loading (0.7 times for reversed loading)
            - 99% reliability

        """

        if is_through_hardened and is_nitrided:
            if grade == 1:
                m, c = 82.3, 12150
            elif grade == 2:
                m, c = 108.6, 15890
        elif is_through_hardened:
            if grade == 1:
                m, c = 77.3, 12800
            elif grade == 2:
                m, c = 102, 16400
        elif is_nitrided:
            if material == "chrome":
                if grade == 1:
                    m, c = 105.2, 9280
                elif grade == 2:
                    m, c = 105.2, 22280
                elif grade == 3:
                    m, c = 105.2, 29280
            elif material == "nitralloy":
                if grade == 1:
                    m, c = 86.2, 12730
                elif grade == 2:
                    m, c = 113.8, 16650

        return Stress(m * H_B + c, StressUnit.PSI)

    def s_b_all(
        *,
        S_t: Stress,
        Y_N: float,
        K_T: float,
        K_R: float,
        S_F: float,
    ) -> Stress:
        """Allowable Bending Stress"""

        S_t_mag = S_t.to(StressUnit.PSI).magnitude

        s_b_all_mag = S_t_mag * Y_N / (K_T * K_R * S_F)
        return Stress(s_b_all_mag, StressUnit.PSI)

    def s_b(
        *,
        W_t: Force,
        K_o: float,
        K_v: float,
        K_s: float,
        K_m: float,
        K_B: float,
        P: InverseLength,
        F: Length,
        J: float,
    ) -> Stress:
        """Bending Stress

        Args:
            W_t (Force): Tangential load
            K_o (float): Overload factor
            K_v (float): Dynamic factor
            K_s (float): Size factor
            K_m (float): Load distribution factor
            K_B (float): Rim thickness factor
            P (InverseLength): Diametral pitch
            F (Length): Face width
            J (float): Geometry factor for bending strength
        """
        W_t_mag = W_t.to(ForceUnit.LBF).magnitude
        F_mag = F.to(LengthUnit.INCH).magnitude
        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

        s_b_mag = (W_t_mag * K_o * K_v * K_s * K_m * K_B * P_mag) / (F_mag * J)

        return Stress(s_b_mag, StressUnit.PSI)

    def s_f_b(
        *,
        s_b_all: Stress,
        s_b: Stress,
    ):
        """Bending Fatigue Safety Factor"""

        s_b_all_mag = s_b_all.to(StressUnit.PSI).magnitude
        s_b_mag = s_b.to(StressUnit.PSI).magnitude

        return s_b_all_mag / s_b_mag

    def S_c(
        *,
        H_B: float,
        grade: int,
    ) -> Stress:
        """Gear Contact Strength

        Args:
            H_B (float): Brinell Hardness Number
            grade (int): Gear quality grade

        Assumptions:
            - 10 million stress cycles
            - 99% reliability

        Source:
            ANSI/AGMA 2001-D04 and 2101-D04
        """

        if grade == 1:
            m, c = 322, 29100
        elif grade == 2:
            m, c = 349, 34300

        return Stress(m * H_B + c, StressUnit.PSI)

    def s_c_all(
        *,
        S_c: Stress,
        Z_N: float,
        C_H: float,
        K_T: float,
        K_R: float,
        S_H: float,
    ) -> Stress:
        """Allowable Contact Stress"""

        S_c_mag = S_c.to(StressUnit.PSI).magnitude

        s_c_all_mag = (S_c_mag * Z_N * C_H) / (K_T * K_R * S_H)
        return Stress(s_c_all_mag, StressUnit.PSI)

    def s_c(
        *,
        C_p: SqrtPressure,
        W_t: Force,
        K_o: float,
        K_v: float,
        K_s: float,
        K_m: float,
        C_f: float,
        d_P: Length,
        F: Length,
        I: float,
    ) -> Stress:
        C_p_mag = C_p.to(SqrtPressureUnit.SQRT_PSI).magnitude
        W_t_mag = W_t.to(ForceUnit.LBF).magnitude
        d_mag = d_P.to(LengthUnit.INCH).magnitude
        F_mag = F.to(LengthUnit.INCH).magnitude

        s_c_mag = C_p_mag * sqrt(
            (W_t_mag * K_o * K_v * K_s * K_m * C_f) / (d_mag * F_mag * I)
        )
        return Stress(s_c_mag, StressUnit.PSI)

    def s_f_c(
        *,
        s_c_all: Stress,
        s_c: Stress,
    ):
        """Contact Fatigue Safety Factor"""

        s_c_all_mag = s_c_all.to(StressUnit.PSI).magnitude
        s_c_mag = s_c.to(StressUnit.PSI).magnitude

        return s_c_all_mag / s_c_mag

    def bending_safety_factor(
        *,
        gear: Gear,
        other_gear: Gear,
        operating_conditions: dict,
        is_pinion: bool,
    ) -> tuple[int, int, int, int]:
        if is_pinion:
            pinion, driven = gear, other_gear
        else:
            pinion, driven = other_gear, gear

        V = Spur.V(d=gear.get_pitch_diameter, n=gear.rpm)

        H = operating_conditions["H"]
        W_t = Spur.W_t(H=H, V=V)

        power_source = operating_conditions["power_source"]
        driven_machine = operating_conditions["driven_machine"]
        K_o = Spur.K_o(power_source=power_source, driven_machine=driven_machine)

        K_v = Spur.K_v(V=V, Q_v=gear.quality_number)

        Y = Spur.Y(N=gear.n_teeth)
        K_s = Spur.K_s(F=gear.face_width, Y=Y, P=gear.get_diametral_pitch)

        C_mc = Spur.C_mc(is_crowned=gear.is_crowned)
        C_pf = Spur.C_pf(F=gear.face_width, d_P=pinion.get_pitch_diameter)
        C_pm = Spur.C_pm(S_1=None, S=None)
        gearing_condition = operating_conditions["gearing_condition"]
        C_ma = Spur.C_ma(F=gear.face_width, gearing_condition=gearing_condition)
        C_e = Spur.C_e(is_gear_adjusted=False)
        K_m = Spur.K_m(C_mc=C_mc, C_pf=C_pf, C_pm=C_pm, C_ma=C_ma, C_e=C_e)

        K_B = Spur.K_B(t_R=None, h_t=None)
        J = Spur.J(N=gear.n_teeth, other_N=other_gear.n_teeth)

        s_b = Spur.s_b(
            W_t=W_t,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            K_B=K_B,
            P=gear.get_diametral_pitch,
            F=gear.face_width,
            J=J,
        )

        S_t = Spur.S_t(
            H_B=gear.material.H_B,
            grade=gear.material.grade,
            is_through_hardened=gear.is_through_hardened,
            is_nitrided=True if gear.surface_finish == "nitrided" else False,
            material=None,
        )
        Y_N = Spur.Y_N(
            L=gear.desired_cycles,
            H_B=gear.material.H_B,
            upper=True,
        )

        T = operating_conditions["T"]
        K_T = Spur.K_T(T=T)

        R = operating_conditions["R"]
        K_R = Spur.K_R(R=R)

        S_F = operating_conditions["S_F"]
        s_b_all = Spur.s_b_all(
            S_t=S_t,
            Y_N=Y_N,
            K_T=K_T,
            K_R=K_R,
            S_F=S_F,
        )

        s_f_b = Spur.s_f_b(s_b=s_b, s_b_all=s_b_all)
        return s_f_b

    def contact_safety_factor(
        *,
        gear: Gear,
        other_gear: Gear,
        operating_conditions: dict,
        is_pinion: bool,
    ) -> tuple[int, int, int, int]:
        if is_pinion:
            pinion, driven = gear, other_gear
        else:
            pinion, driven = other_gear, gear

        C_p = Spur.C_p(
            mu_P=pinion.material.poissons_ratio,
            E_P=pinion.material.modulus_of_elasticity,
            mu_G=driven.material.poissons_ratio,
            E_G=driven.material.modulus_of_elasticity,
        )

        V = Spur.V(d=gear.get_pitch_diameter, n=gear.rpm)

        H = operating_conditions["H"]
        W_t = Spur.W_t(H=H, V=V)

        K_o = Spur.K_o(
            power_source=operating_conditions["power_source"],
            driven_machine=operating_conditions["driven_machine"],
        )

        K_v = Spur.K_v(V=V, Q_v=gear.quality_number)
        Y = Spur.Y(N=gear.n_teeth)
        K_s = Spur.K_s(F=gear.face_width, Y=Y, P=gear.get_diametral_pitch)
        C_mc = Spur.C_mc(is_crowned=gear.is_crowned)
        C_pf = Spur.C_pf(F=gear.face_width, d_P=pinion.get_pitch_diameter)
        C_pm = Spur.C_pm(S_1=None, S=None)
        C_ma = Spur.C_ma(
            F=gear.face_width,
            gearing_condition=operating_conditions["gearing_condition"],
        )
        C_e = Spur.C_e(is_gear_adjusted=False)
        K_m = Spur.K_m(C_mc=C_mc, C_pf=C_pf, C_pm=C_pm, C_ma=C_ma, C_e=C_e)
        C_f = Spur.C_f()
        d_P = pinion.get_pitch_diameter
        F = gear.face_width

        m_G = Spur.m_G(N_P=pinion.n_teeth, N_G=driven.n_teeth)
        m_N = Spur.m_N()
        I = Spur.I(
            m_G=m_G,
            m_N=m_N,
            phi=gear.pressure_angle,
            gear_mode=operating_conditions["gear_mode"],
        )

        s_c = Spur.s_c(
            C_p=C_p,
            W_t=W_t,
            K_o=K_o,
            K_v=K_v,
            K_s=K_s,
            K_m=K_m,
            C_f=C_f,
            d_P=d_P,
            F=F,
            I=I,
        )

        S_c = Spur.S_c(H_B=gear.material.H_B, grade=gear.material.grade)
        Z_N = Spur.Z_N(
            L=gear.desired_cycles,
            is_nitrided=True if gear.surface_finish == "nitrided" else False,
        )
        C_H = Spur.C_H(
            m_G=m_G,
            H_B_P=pinion.material.H_B,
            f_P=None,
            H_B_G=driven.material.H_B,
            is_pinion=is_pinion,
        )

        T = operating_conditions["T"]
        K_T = Spur.K_T(T=T)

        R = operating_conditions["R"]
        K_R = Spur.K_R(R=R)

        S_H = operating_conditions["S_H"]
        s_c_all = Spur.s_c_all(S_c=S_c, Z_N=Z_N, C_H=C_H, K_T=K_T, K_R=K_R, S_H=S_H)

        s_f_c = Spur.s_f_c(s_c=s_c, s_c_all=s_c_all)
        return s_f_c


def report(pinion: Gear, driven: Gear, operating_conditions: dict):
    pinion_bending_sf = Spur.bending_safety_factor(
        gear=pinion,
        other_gear=driven,
        operating_conditions=operating_conditions,
        is_pinion=True,
    )
    print(f"Pinion Bending Safety Factor: {pinion_bending_sf:.2f}")

    gear_bending_sf = Spur.bending_safety_factor(
        gear=driven,
        other_gear=pinion,
        operating_conditions=operating_conditions,
        is_pinion=False,
    )
    print(f"Gear Bending Safety Factor: {gear_bending_sf:.2f}")

    pinion_contact_sf = Spur.contact_safety_factor(
        gear=pinion,
        other_gear=driven,
        operating_conditions=operating_conditions,
        is_pinion=True,
    )
    print(f"Pinion Contact Safety Factor: {pinion_contact_sf:.2f}")

    gear_contact_sf = Spur.contact_safety_factor(
        gear=driven,
        other_gear=pinion,
        operating_conditions=operating_conditions,
        is_pinion=False,
    )
    print(f"Gear Contact Safety Factor: {gear_contact_sf:.2f}")


def demo():
    from pydome.materials import Steel

    pinion = Gear(
        n_teeth=20,
        rpm=AngularVelocity(1200, AngularVelocityUnit.RPM),
        diametral_pitch=InverseLength(1, InverseLengthUnit.PER_MM),
        face_width=Length(2, LengthUnit.INCH),
        desired_cycles=1e8,
        material=Steel,
        quality_number=6,
        is_crowned=False,
        is_through_hardened=True,
        pressure_angle=Angle(20, AngleUnit.DEGREE),
    )

    gear = Gear(
        n_teeth=40,
        rpm=AngularVelocity(600, AngularVelocityUnit.RPM),
        diametral_pitch=InverseLength(1, InverseLengthUnit.PER_MM),
        face_width=Length(2, LengthUnit.INCH),
        desired_cycles=5e7,
        material=Steel,
        quality_number=6,
        is_crowned=False,
        is_through_hardened=True,
        pressure_angle=Angle(20, AngleUnit.DEGREE),
    )

    operating_conditions = dict(
        H=Power(600, PowerUnit.WATT),
        power_source="uniform",
        driven_machine="moderate_shock",
        gearing_condition="enclosed_commercial",
        T=Temperature(30, TemperatureUnit.CELSIUS),
        R=0.99,
        S_F=1.0,
        S_H=1.0,
        gear_mode="external",
    )

    report(pinion, gear, operating_conditions=operating_conditions)
