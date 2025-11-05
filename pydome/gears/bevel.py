from math import atan, log10, pi, sin, sqrt, tan
from typing import Literal

import numpy as np
import scipy

from pydome.gears.spur import Spur
from pydome.units import *  # noqa: F403


class Bevel(Spur):
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

        gamma, _ = Bevel.pitch_angle(
            N_p=1, N_g=d_G_mag / d_P_mag, shaft_angle=shaft_angle
        )

        A_0_mag = d_P_mag / (2 * sin(gamma.to(AngleUnit.RADIAN).magnitude))
        return Length(A_0_mag, LengthUnit.INCH)

    def pitch_angle(*, N_p: int, N_g: int, shaft_angle: Angle) -> tuple[Angle, Angle]:
        """
        Cone angles (pitch angles) for pinion and gear

        Args:
            N_p: Pinion teeth
            N_g: Gear teeth
            shaft_angle: Shaft angle
        """
        shaft_angle_mag = shaft_angle.to(AngleUnit.RADIAN).magnitude
        gamma_p_mag = atan(N_p / N_g * tan(shaft_angle_mag / 2))
        gamma_g_mag = atan(N_g / N_p * tan(shaft_angle_mag / 2))
        return (
            Angle(gamma_p_mag, AngleUnit.RADIAN),
            Angle(gamma_g_mag, AngleUnit.RADIAN),
        )

    def face_width_max(A_0: Length, P: InverseLength) -> Length:
        """
        Maximum recommended face width

        Args:
            A_0: Outer cone distance
            P: Outer transverse diametral pitch
        """
        A_0_mag = A_0.to(LengthUnit.INCH).magnitude
        P_mag = P.to(InverseLengthUnit.PER_INCH).magnitude
        return Length(min(0.3 * A_0_mag, 10 / P_mag), LengthUnit.INCH)

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

    def K_s(*, P: InverseLength) -> float:
        """
        Size factor for bending K_s

        Args:
            P: Outer transverse diametral pitch
        """
        P_d_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

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
            T_f: Gear blank or oil temperature
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
            R: Reliability
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

    def C_R(*, R: float) -> float:
        """
        Reliability factor for pitting CR

        Args:
            R: Reliability
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

    # arr[gear] = [(mate, J), ...]
    _arr_J = {
        13: [(29.733333333333334, 0.16720554272517324), (37.599999999999994, 0.1696997690531178), (44.8, 0.17108545034642034), (49.733333333333334, 0.17163972286374135), (59.86666666666667, 0.17191685912240187), (64.4, 0.17136258660508086), (71.19999999999999, 0.17773672055427253), (78.4, 0.18300230946882218), (84.93333333333334, 0.1868822170900693), (90.0, 0.18882217090069287), (94.93333333333334, 0.19076212471131643), (99.86666666666666, 0.19187066974595846)],
        15: [(16.0, 0.17551963048498848), (18.266666666666666, 0.17468822170900694), (20.53333333333333, 0.17496535796766746), (24.133333333333333, 0.17829099307159355), (34.266666666666666, 0.18106235565819864), (38.8, 0.18161662817551966), (54.93333333333333, 0.18189376443418015), (65.19999999999999, 0.18050808314087763), (74.93333333333334, 0.19020785219399539), (82.4, 0.19491916859122405), (92.53333333333333, 0.19990762124711317), (99.86666666666666, 0.20212471131639725)],
        20: [(13.733333333333334, 0.19824480369515013), (16.933333333333334, 0.20184757505773673), (18.266666666666666, 0.20184757505773673), (22.666666666666664, 0.1965819861431871), (33.46666666666667, 0.19879907621247114), (43.333333333333336, 0.20240184757505775), (58.53333333333333, 0.20212471131639725), (64.13333333333333, 0.20129330254041572), (74.93333333333334, 0.21515011547344112), (84.93333333333334, 0.22318706697459587), (93.2, 0.22762124711316398), (99.86666666666666, 0.23011547344110855)],
        25: [(14.0, 0.20628175519630487), (15.866666666666667, 0.21431870669745962), (18.53333333333333, 0.21986143187066975), (20.0, 0.2206928406466513), (21.333333333333336, 0.2206928406466513), (26.4, 0.21515011547344112), (31.333333333333332, 0.2140415704387991), (37.86666666666667, 0.21431870669745962), (44.266666666666666, 0.21709006928406469), (60.666666666666664, 0.21542725173210164), (80.8, 0.23759815242494228), (99.86666666666666, 0.2500692840646651)],
        30: [(12.933333333333334, 0.21542725173210164), (21.2, 0.23260969976905313), (22.933333333333334, 0.23399538106235568), (26.266666666666666, 0.23371824480369516), (29.066666666666666, 0.2312240184757506), (40.0, 0.22706697459584296), (49.86666666666667, 0.22595842956120094), (57.733333333333334, 0.22512702078521943), (69.86666666666667, 0.23926096997690532), (79.86666666666666, 0.2481293302540416), (90.0, 0.25588914549653585), (99.86666666666666, 0.26198614318706703)],
        35: [(13.6, 0.2212471131639723), (22.8, 0.24286374133949193), (25.866666666666667, 0.24563510392609703), (28.266666666666666, 0.24563510392609703), (33.33333333333333, 0.2412009237875289), (39.86666666666667, 0.23759815242494228), (49.86666666666667, 0.23676674364896075), (55.06666666666666, 0.2381524249422633), (57.733333333333334, 0.23842956120092382), (74.93333333333334, 0.2556120092378753), (90.0, 0.26669745958429564), (99.86666666666666, 0.2725173210161663)],
        40: [(13.466666666666667, 0.22374133949191688), (22.8, 0.24757505773672056), (26.8, 0.2522863741339492), (30.266666666666666, 0.25394919168591223), (32.93333333333334, 0.2542263279445728), (39.86666666666667, 0.24923787528868363), (49.86666666666667, 0.24646651270207853), (54.13333333333333, 0.24674364896073905), (64.93333333333334, 0.2572748267898384), (74.93333333333334, 0.2658660508083141), (87.46666666666667, 0.2750115473441109), (99.86666666666666, 0.2822170900692841)],
        45: [(12.533333333333333, 0.22401847575057737), (16.0, 0.23316397228637414), (17.733333333333334, 0.2348267898383372), (18.53333333333333, 0.23953810623556585), (20.0, 0.24397228637413396), (26.266666666666666, 0.25588914549653585), (29.866666666666667, 0.25893764434180144), (38.8, 0.25949191685912243), (46.93333333333333, 0.2553348729792148), (48.8, 0.2556120092378753), (74.93333333333334, 0.2763972286374134), (99.86666666666666, 0.29163972286374135)],
        50: [(12.8, 0.22678983833718247), (16.133333333333333, 0.23538106235565823), (18.4, 0.23981524249422634), (21.46666666666667, 0.25034642032332566), (26.266666666666666, 0.26032332563510396), (31.333333333333332, 0.2658660508083141), (36.66666666666667, 0.2675288683602771), (44.93333333333333, 0.26836027713625865), (52.266666666666666, 0.27030023094688227), (64.93333333333334, 0.2797228637413395), (80.0, 0.2894226327944573), (99.86666666666666, 0.3002309468822171)],
        60: [(12.533333333333333, 0.22706697459584296), (14.933333333333334, 0.23648960739030025), (18.133333333333333, 0.24314087759815245), (21.2, 0.2520092378752887), (24.933333333333334, 0.2597690531177829), (30.0, 0.26697459584295613), (35.06666666666666, 0.2763972286374134), (40.0, 0.2833256351039261), (44.93333333333333, 0.2894226327944573), (49.86666666666667, 0.29413394919168595), (54.93333333333333, 0.298013856812933), (60.0, 0.3002309468822171)],
        70: [(12.533333333333333, 0.22734411085450348), (14.933333333333334, 0.23759815242494228), (20.0, 0.25478060046189377), (24.933333333333334, 0.2686374133949192), (30.0, 0.2797228637413395), (34.93333333333334, 0.2894226327944573), (39.86666666666667, 0.29718244803695154), (44.93333333333333, 0.30438799076212475), (50.0, 0.31048498845265593), (56.53333333333333, 0.3171362586605081), (64.0, 0.32378752886836026), (70.0, 0.3284988452655889)],
        80: [(12.666666666666666, 0.23981524249422634), (19.2, 0.2617090069284065), (26.4, 0.2802771362586605), (34.93333333333334, 0.298013856812933), (42.13333333333333, 0.3102078521939954), (49.2, 0.3201847575057737), (54.93333333333333, 0.3271131639722864), (60.0, 0.33237875288683605), (64.93333333333334, 0.3370900692840647), (68.93333333333334, 0.3401385681293303), (74.0, 0.3437413394919169), (79.46666666666667, 0.34678983833718247)],
        90: [(12.666666666666666, 0.24729792147806007), (17.733333333333334, 0.2658660508083141), (21.733333333333334, 0.2775057736720554), (30.0, 0.29718244803695154), (39.86666666666667, 0.3157505773672056), (49.86666666666667, 0.330715935334873), (60.0, 0.34235565819861435), (68.13333333333333, 0.3503926096997691), (74.93333333333334, 0.35538106235565825), (79.86666666666666, 0.35870669745958433), (84.93333333333334, 0.3614780600461894), (90.0, 0.3636951501154735)],
        100: [(12.8, 0.2531177829099307), (17.866666666666667, 0.2719630484988453), (23.733333333333334, 0.2891454965357968), (32.93333333333334, 0.3107621247113164), (41.733333333333334, 0.32655889145496536), (49.86666666666667, 0.3387528868360277), (54.93333333333333, 0.3451270207852194), (64.93333333333334, 0.35593533487297924), (75.06666666666666, 0.3642494226327945), (84.93333333333334, 0.37034642032332565), (92.66666666666667, 0.3742263279445728), (99.86666666666666, 0.3767205542725173)],
    }  # fmt:off

    _points_J = []
    _values_J = []
    for N_gear_val, vals in _arr_J.items():
        for N_other_gear_val, J_val in vals:
            _points_J.append((N_gear_val, N_other_gear_val))
            _values_J.append(J_val)

    J_interpolator = scipy.interpolate.LinearNDInterpolator(_points_J, _values_J)
    J_ckdtree = scipy.spatial.cKDTree(_points_J)

    def J(*, N_gear: int, N_mate: int) -> float:
        """
        Bending strength geometry factor J for pinion and gear.
        Interpolated from AGMA 2003-B97 Figure 15-7 (20° pressure angle, 90° shaft).

        Args:
            N_gear: Number of teeth for which geometry factor is desired
            N_gear: Number of teeth in mate
        """

        J_interp = Bevel.J_interpolator((N_gear, N_mate))
        if J_interp is None:
            _, idx = Bevel.J_ckdtree.query((N_gear, N_mate))
            J_interp = Bevel._values_J[idx[0]]
        return float(J_interp)

    # arr[gear_teeth] = [(pinion_teeth, I), ...]
    _arr_I = {
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

    _points_I = []
    _values_I = []
    for N_gear_val, vals in _arr_I.items():
        for N_pinion_val, I_val in vals:
            _points_I.append((N_pinion_val, N_gear_val))
            _values_I.append(I_val)

    I_interpolator = scipy.interpolate.LinearNDInterpolator(_points_I, _values_I)
    I_ckdtree = scipy.spatial.cKDTree(_points_I)

    def I(*, N_pinion: int, N_gear: int) -> float:  # noqa: E743
        """
        Pitting resistance geometry factor I for straight-bevel gears.
        Simplified from AGMA 2003-B97 Figure 15-6 (20° pressure angle, 90° shaft).

        Args:
            N_pinion: Number of pinion teeth
            N_gear: Number of gear teeth
        """

        I_interp = Bevel.I_interpolator((N_pinion, N_gear))
        if I_interp is None:
            _, idx = Bevel.I_ckdtree.query([(N_pinion, N_gear)], k=1)
            I_interp = Bevel._values_I[idx[0]]
        return float(I_interp)

    def S_c(
        *,
        H_B: float | None,
        grade: int,
        is_through_hardened: bool,
        is_case_hardened: bool,
    ) -> Stress:
        """
        Contact strength number S_c for steel gears

        Args:
            H_B: Brinell Hardness Number
            grade: Gear quality grade
            is_through_hardened: True for through-hardened,
            is_case_hardened: True for case-hardened,
            If both false, assumes nitrided steel
        """
        if is_through_hardened:
            if grade == 1:
                m, c = 341, 23620
            elif grade == 2:
                m, c = 263.6, 29560
            return Stress((m * H_B + c), StressUnit.PSI)
        elif is_case_hardened:
            if grade == 1:
                return Stress(200000, StressUnit.PSI)
            elif grade == 2:
                return Stress(225000, StressUnit.PSI)
            elif grade == 3:
                return Stress(250000, StressUnit.PSI)
        else:
            return Stress(145000, StressUnit.PSI)

    def S_t(
        *,
        H_B: float | None,
        grade: int,
        is_through_hardened: bool,
        is_case_hardened: bool,
    ) -> Stress:
        """
        Bending strength number S_t for steel gears

        Args:
            H_B: Brinell Hardness Number
            grade: Gear quality grade
            is_through_hardened: True for through-hardened,
            is_case_hardened: True for case-hardened,
            If both false, assumes nitrided steel
        """
        if is_through_hardened:
            if grade == 1:
                m, c = 44, 2100
            elif grade == 2:
                m, c = 48, 5980
            return Stress((m * H_B + c), StressUnit.PSI)
        elif is_case_hardened:
            if grade == 1:
                return Stress(30000, StressUnit.PSI)
            elif grade == 2:
                return Stress(35000, StressUnit.PSI)
            elif grade == 3:
                return Stress(40000, StressUnit.PSI)
        else:
            return Stress(22000, StressUnit.PSI)

    def s_t(
        *,
        W_t: Force,
        F: Length,
        P: InverseLength,
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
            P: Outer transverse diametral pitch
            K_o: Overload factor
            K_v: Dynamic factor
            K_s: Size factor for bending
            K_m: Load distribution factor
            K_x: Lengthwise curvature factor (1.0 for straight bevel)
            J: Bending strength geometry factor
        """

        W_t_mag = W_t.to(ForceUnit.LBF).magnitude
        F_mag = F.to(LengthUnit.INCH).magnitude
        P_d_mag = P.to(InverseLengthUnit.PER_INCH).magnitude

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
            W_t: Tangential load
            F: Face width
            d_p: Pinion pitch diameter at large end
            K_o: Overload factor
            K_v: Dynamic factor
            K_m: Load distribution factor
            C_s: Size factor for pitting
            C_xc: Crowning factor for pitting
            I: Geometry factor for pitting resistance
            C_p: Elastic coefficient
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


'''
    def power_rating_bending(
        N_p: int,
        N_g: int,
        P: float,
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
            P: Outer transverse diametral pitch
            F: Face width
            n_p: Pinion speed
            HB_p: Pinion Brinell Hardness
            HB_g: Gear Brinell Hardness
            grade: Gear quality grade
            is_crowned: Whether teeth are crowned
            Q_v: Transmission accuracy number
            R: Reliability
            N_L: Number of load cycles
            K_o: Overload factor
            SF: Safety factor for bending
        """
        d_p = Bevel.d(N_p, P)
        d_g = Bevel.d(N_g, P)
        V_t = Bevel.V(d_p, n_p)

        J_p, J_g = Bevel.J(N_p, N_g)

        Kv_val = Bevel.K_v(V_t, Q_v)
        Ks_val = Bevel.K_s(P)
        K_mb = 1.25
        Km_val = Bevel.K_m(F, K_mb)
        Kx_val = Bevel.K_x()
        KT_val = Bevel.K_T()
        KR_val = Bevel.K_R(R)
        KL_val = Bevel.K_L(int(N_L))

        St_p = Bevel.S_t(HB_p, grade, True)
        s_t_all = St_p * KL_val * SF * KT_val * KR_val

        W_t_perm = (s_t_all * P * F * J_p) / (K_o * Kv_val * Ks_val * Km_val * Kx_val)

        H_bending = W_t_perm * V_t / 33000

        return H_bending

    def power_rating_wear(
        N_p: int,
        N_g: int,
        P: float,
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
            P: Outer transverse diametral pitch 
            F: Face width
            n_p: Pinion speed
            HB_p: Pinion Brinell Hardness
            HB_g: Gear Brinell Hardness
            grade: Gear quality grade
            is_crowned: Whether teeth are crowned
            Q_v: Transmission accuracy number
            R: Reliability 
            N_L: Number of load cycles 
            K_o: Overload factor
            SH: Safety factor for pitting
            C_p: Elastic coefficient
        """
        d_p = Bevel.d(N_p, P)
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
