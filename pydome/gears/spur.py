from math import cos, exp, log, sin, sqrt
from typing import Literal

import numpy as np

from pydome.constants import pi
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
        N: int,
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
        Y_mag = Spur.Y(N=N)
        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

        return 1.192 * ((F_mag * sqrt(Y_mag)) / P_mag) ** 0.0535

    def K_m(
        *,
        is_crowned: bool,
        F: Length,
        d_P: Length,
        S_1: float,
        S: float,
        gearing_condition: Literal[
            "open", "enclosed_commercial", "enclosed_precise", "enclosed_extra_precise"
        ],
        is_gear_adjusted: bool = False,
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

        return 1 + Spur.C_mc(is_crowned=is_crowned) * (
            Spur.C_pf(F=F, d_P=d_P) * Spur.C_pm(S_1=S_1, S=S)
            + Spur.C_ma(F=F, gearing_condition=gearing_condition)
            * Spur.C_e(is_gear_adjusted=is_gear_adjusted)
        )

    def C_mc(
        *,
        is_crowned: bool,
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
        """Mesh Proportion Modifier"""
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
        x = np.array([17, 19, 21, 26, 29, 35, 55, 135])
        y = np.array([17, 19, 21, 26, 29, 35, 55, 135])

        # a[row,col] is the J for N1=x[col] and N2=y[row]
        a = np.array(
            [  #   17      19     21     26     29     35     55    135
                [0.289, 0.315, 0.326, 0.347, 0.356, 0.378, 0.415, 0.447],  # 17
                [0.289, 0.316, 0.326, 0.348, 0.357, 0.380, 0.417, 0.448],  # 19
                [0.289, 0.317, 0.327, 0.349, 0.358, 0.381, 0.418, 0.450],  # 21
                [0.289, 0.319, 0.330, 0.351, 0.360, 0.384, 0.422, 0.454],  # 26
                [0.289, 0.321, 0.332, 0.353, 0.363, 0.387, 0.425, 0.458],  # 29
                [0.289, 0.325, 0.336, 0.358, 0.367, 0.392, 0.431, 0.465],  # 35
                [0.289, 0.333, 0.343, 0.366, 0.376, 0.402, 0.444, 0.480],  # 55
                [0.289, 0.339, 0.351, 0.376, 0.387, 0.414, 0.460, 0.499],  # 135
            ]
        )

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
        N_P: int,
        N_G: int,
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
            m_G = Spur.m_G(N_P=N_P, N_G=N_G)

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
        N_P: int,
        N_G: int,
        phi: Angle,
        gear_mode: Literal["external", "internal"],
    ):
        """Geometry Factor for Pitting Resistance"""

        m_G = Spur.m_G(N_P=N_P, N_G=N_G)
        phi_mag = phi.to(AngleUnit.RADIAN).magnitude
        m_N = Spur.m_N()

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
        H_B: float,
        grade: int,
        L: int,
        surface: Literal["case", "nitrided"] | None,
        upper: bool,
        T: Temperature,
        R: float,
        is_through_hardened: bool,
        material: Literal["chrome", "nitralloy"] | None,
        S_F: float,
    ) -> Stress:
        """Allowable Bending Stress"""

        S_t_mag = (
            Spur.S_t(
                H_B=H_B,
                grade=grade,
                is_through_hardened=is_through_hardened,
                is_nitrided=(surface == "nitrided"),
                material=material,
            )
            .to(StressUnit.PSI)
            .magnitude
        )

        Y_N_mag = Spur.Y_N(L=L, H_B=surface or H_B, upper=upper)
        K_T_mag = Spur.K_T(T=T)
        K_R_mag = Spur.K_R(R=R)

        s_b_all_mag = S_t_mag * Y_N_mag / (K_T_mag * K_R_mag * S_F)
        return Stress(s_b_all_mag, StressUnit.PSI)

    def s_b(
        *,
        N: int,
        P: InverseLength,
        n: AngularVelocity,
        H: Power,
        power_source: Literal["uniform", "light_shock", "medium_shock"],
        driven_machine: Literal["uniform", "moderate_shock", "heavy_shock"],
        Q_v: int,
        F: Length,
        is_crowned: bool,
        d_P: Length,
        S_1: float,
        S: float,
        gearing_condition: Literal[
            "open", "enclosed_commercial", "enclosed_precise", "enclosed_extra_precise"
        ],
        is_gear_adjusted: bool,
        t_R: Length,
        h_t: Length,
        other_N: int,
        J: float | None = None,
    ) -> Stress:
        """Bending Stress

        Args:
            N (int): Number of teeth
            P (InverseLength): Diametral pitch
            n (AngularVelocity): Rotational speed
            H (Power): Power transmitted
            power_source (Literal): Type of power source
            driven_machine (Literal): Type of driven machine
            Q_v (int): Quality number
            F (Length): Face width
            is_crowned (bool): Whether the gear is crowned
            d_P (Length): Pinion pitch diameter
            S_1 (float):
            S (float):
            gearing_condition (Literal): Gear environment.
            is_gear_adjusted (bool): Whether the gearing has been adjusted at assembly
            t_R (Length): Rim thickness below the tooth
            h_t (Length): Tooth height
            other_N (int): Number of teeth of mating gear
            J (float | None, default=None): Geometry factor for bending strength. If None, it will be calculated.
        """

        d_value = Spur.d(N=N, P=P)
        V_value = Spur.V(d=d_value, n=n)
        W_t_value = Spur.W_t(H=H, V=V_value)

        W_t_mag = W_t_value.to(ForceUnit.LBF).magnitude
        K_o_mag = Spur.K_o(power_source=power_source, driven_machine=driven_machine)
        K_v_mag = Spur.K_v(V=V_value, Q_v=Q_v)
        K_s_mag = Spur.K_s(F=F, N=N, P=P)
        K_m_mag = Spur.K_m(
            is_crowned=is_crowned,
            F=F,
            d_P=d_P,
            S_1=S_1,
            S=S,
            gearing_condition=gearing_condition,
            is_gear_adjusted=is_gear_adjusted,
        )
        K_B_mag = Spur.K_B(t_R=t_R, h_t=h_t)

        if J is not None:
            J_mag = J
        else:
            J_mag = Spur.J(N=N, other_N=other_N)

        F_mag = F.to(LengthUnit.INCH).magnitude
        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

        s_b_mag = (
            W_t_mag * K_o_mag * K_v_mag * K_s_mag * K_m_mag * K_B_mag * P_mag
        ) / (F_mag * J_mag)

        return Stress(s_b_mag, StressUnit.PSI)

    def s_f_b(
        *,
        N: int,
        P: InverseLength,
        F: Length,
        H: Power,
        n: AngularVelocity,
        Q_v: int,
        is_crowned: bool,
        d_P: Length,
        power_source: Literal["uniform", "light_shock", "medium_shock"],
        driven_machine: Literal["uniform", "moderate_shock", "heavy_shock"],
        gearing_condition: Literal[
            "open", "enclosed_commercial", "enclosed_precise", "enclosed_extra_precise"
        ],
        other_N: int,
        H_B: float,
        grade: int,
        L: int,
        T: Temperature,
        R: float,
        is_through_hardened: bool,
        surface: Literal["case", "nitrided"] | None,
        material: Literal["chrome", "nitralloy"] | None,
        S_F: float,
        upper: bool = True,
        is_gear_adjusted: bool = False,
        S_1: float = 0,
        S: float = 1,
        t_R: Length | None = None,
        h_t: Length | None = None,
    ):
        s_b = Spur.s_b(
            N=N,
            P=P,
            n=n,
            H=H,
            power_source=power_source,
            driven_machine=driven_machine,
            Q_v=Q_v,
            F=F,
            is_crowned=is_crowned,
            d_P=d_P,
            S_1=S_1,
            S=S,
            gearing_condition=gearing_condition,
            is_gear_adjusted=is_gear_adjusted,
            t_R=t_R,
            h_t=h_t,
            other_N=other_N,
        )

        s_b_all = Spur.s_b_all(
            H_B=H_B,
            grade=grade,
            L=L,
            surface=surface,
            upper=upper,
            T=T,
            R=R,
            is_through_hardened=is_through_hardened,
            material=material,
            S_F=S_F,
        )

        return s_b_all.magnitude / s_b.magnitude

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
        H_B: float,
        grade: int,
        L: int,
        surface: Literal["nitrided"],
        T: Temperature,
        R: float,
        N_P: int,
        N_G: int,
        H_B_P: float | None,
        f_P: float | None,
        H_B_G: float,
        is_pinion: bool,
        S_H: float,
        Z_N: float | None = None,
    ) -> Stress:
        """Allowable Contact Stress"""

        S_c_mag = Spur.S_c(H_B=H_B, grade=grade).to(StressUnit.PSI).magnitude

        if Z_N is not None:
            Z_N_mag = Z_N
        else:
            Z_N_mag = Spur.Z_N(L=L, is_nitrided=(surface == "nitrided"))

        K_T_mag = Spur.K_T(T=T)
        K_R_mag = Spur.K_R(R=R)
        C_H_mag = Spur.C_H(
            N_P=N_P,
            N_G=N_G,
            H_B_P=H_B_P,
            f_P=f_P,
            H_B_G=H_B_G,
            is_pinion=is_pinion,
        )

        s_c_all_mag = (S_c_mag * Z_N_mag * C_H_mag) / (K_T_mag * K_R_mag * S_H)
        return Stress(s_c_all_mag, StressUnit.PSI)

    def s_c(
        *,
        N: int,
        P: InverseLength,
        n: AngularVelocity,
        H: Power,
        power_source: Literal["uniform", "light_shock", "medium_shock"],
        driven_machine: Literal["uniform", "moderate_shock", "heavy_shock"],
        Q_v: int,
        F: Length,
        mu_P: float,
        E_P: float,
        mu_G: float,
        E_G: float,
        is_crowned: bool,
        d_P: Length,
        S_1: float,
        S: float,
        gearing_condition: Literal[
            "open", "enclosed_commercial", "enclosed_precise", "enclosed_extra_precise"
        ],
        is_gear_adjusted: bool,
        other_N: int,
        phi: Angle,
        gear_mode: Literal["external", "internal"],
        C_p: SqrtPressure | None = None,
    ):
        d_value = Spur.d(N=N, P=P)
        V_value = Spur.V(d=d_value, n=n)
        W_t_value = Spur.W_t(H=H, V=V_value)
        F_mag = F.to(LengthUnit.INCH).magnitude

        W_t_mag = W_t_value.to(ForceUnit.LBF).magnitude
        d_mag = d_value.to(LengthUnit.INCH).magnitude

        if C_p is not None:
            C_p_mag = C_p.to(SqrtPressureUnit.SQRT_PSI).magnitude
        else:
            C_p_mag = (
                Spur.C_p(mu_P=mu_P, E_P=E_P, mu_G=mu_G, E_G=E_G)
                .to(SqrtPressureUnit.SQRT_PSI)
                .magnitude
            )

        K_o_mag = Spur.K_o(power_source=power_source, driven_machine=driven_machine)
        K_v_mag = Spur.K_v(V=V_value, Q_v=Q_v)
        K_s_mag = Spur.K_s(F=F, N=N, P=P)
        K_m_mag = Spur.K_m(
            is_crowned=is_crowned,
            F=F,
            d_P=d_P,
            S_1=S_1,
            S=S,
            gearing_condition=gearing_condition,
            is_gear_adjusted=is_gear_adjusted,
        )
        C_f_mag = Spur.C_f()
        I_mag = Spur.I(
            N_P=N,
            N_G=other_N,
            phi=phi,
            gear_mode=gear_mode,
        )

        s_c_mag = C_p_mag * sqrt(
            (W_t_mag * K_o_mag * K_v_mag * K_s_mag * K_m_mag * C_f_mag)
            / (d_mag * F_mag * I_mag)
        )
        return Stress(s_c_mag, StressUnit.PSI)
