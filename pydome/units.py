from enum import Enum

import numpy as np

pi = np.pi


class Unit:
    def to_primary(self):
        _factor_to_primary = self._factors_to_primary.get(self.unit)
        if _factor_to_primary is None:
            raise ValueError(f"Unsupported unit: {self.unit}")

        self.magnitude *= _factor_to_primary
        self.unit = self.primary_unit
        return self

    def to(self, target_unit):
        if self.unit == target_unit:
            return self

        self.to_primary()
        _factor = self._factors_to_primary.get(target_unit)
        if _factor is None:
            raise ValueError(f"Unsupported target unit: {target_unit}")
        _factor_from_primary = 1 / _factor

        self.magnitude *= _factor_from_primary
        self.unit = target_unit
        return self

    def __repr__(self):
        return f"{self.magnitude} {self.unit.value}"

    def equals(self, other: "Unit", rel: float = 1e-3) -> bool:
        self.to_primary()
        other.to_primary()

        return np.isclose(self.magnitude, other.magnitude, rtol=rel)

    def copy(self) -> "Unit":
        return type(self)(self.magnitude, self.unit)

    def __add__(self, other: "Unit") -> "Unit":
        self.to_primary()
        other.to_primary()

        return type(self)(self.magnitude + other.magnitude, self.primary_unit)

    def __sub__(self, other: "Unit") -> "Unit":
        self.to_primary()
        other.to_primary()

        return type(self)(self.magnitude - other.magnitude, self.primary_unit)

    def __mul__(self, other: float) -> "Unit":
        return type(self)(self.magnitude * other, self.unit)

    def __rmul__(self, other: float) -> "Unit":
        return self.__mul__(other)

    def __truediv__(self, other: "Unit | float | int") -> "Unit | float":
        if isinstance(other, Unit):
            if type(self) is not type(other):
                raise ValueError("Can only divide units of the same type")
            self.to_primary()
            other.to_primary()
            return self.magnitude / other.magnitude
        elif isinstance(other, (float, int)):
            self.to_primary()
            return type(self)(self.magnitude / other, self.primary_unit)


class VelocityUnit(Enum):
    MPS = "m/s"
    KPH = "km/h"
    FPM = "ft/min"


class Velocity(Unit):
    primary_unit = VelocityUnit.MPS
    _factors_to_primary = {
        VelocityUnit.MPS: 1.0,
        VelocityUnit.KPH: 1 / 3.6,
        VelocityUnit.FPM: 0.00508,
    }

    def __init__(self, magnitude: float, unit: VelocityUnit):
        self.magnitude = magnitude
        self.unit = unit


class AngleUnit(Enum):
    RADIAN = "radian"
    DEGREE = "degree"


class Angle(Unit):
    primary_unit = AngleUnit.RADIAN
    _factors_to_primary = {
        AngleUnit.RADIAN: 1.0,
        AngleUnit.DEGREE: pi / 180.0,
    }

    def __init__(self, magnitude: float, unit: AngleUnit):
        self.magnitude = magnitude
        self.unit = unit


class LengthUnit(Enum):
    MM = "mm"
    INCH = "inch"
    M = "m"


class Length(Unit):
    primary_unit = LengthUnit.MM
    _factors_to_primary = {
        LengthUnit.MM: 1.0,
        LengthUnit.INCH: 25.4,
        LengthUnit.M: 1000.0,
    }

    def __init__(self, magnitude: float, unit: LengthUnit):
        self.magnitude = magnitude
        self.unit = unit


class TemperatureUnit(Enum):
    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"


class Temperature(Unit):
    primary_unit = TemperatureUnit.KELVIN
    _lambdas_to_primary = {
        TemperatureUnit.KELVIN: lambda x: x,
        TemperatureUnit.CELSIUS: lambda x: x + 273.15,
        TemperatureUnit.FAHRENHEIT: lambda x: (x - 32) * 5 / 9 + 273.15,
    }

    _lamdas_from_primary = {
        TemperatureUnit.KELVIN: lambda x: x,
        TemperatureUnit.CELSIUS: lambda x: x - 273.15,
        TemperatureUnit.FAHRENHEIT: lambda x: (x - 273.15) * 9 / 5 + 32,
    }

    def to_primary(self) -> "Temperature":
        _lambda = Temperature._lambdas_to_primary.get(self.unit)
        if _lambda is None:
            raise ValueError(f"Unsupported unit: {self.unit}")

        self.magnitude = _lambda(self.magnitude)
        self.unit = TemperatureUnit.KELVIN
        return self

    def to(self, target_unit: TemperatureUnit) -> "Temperature":
        if self.unit == target_unit:
            return self

        self.to_primary()
        _lambda = Temperature._lamdas_from_primary.get(target_unit)
        if _lambda is None:
            raise ValueError(f"Unsupported target unit: {target_unit}")

        self.magnitude = _lambda(self.magnitude)
        self.unit = target_unit
        return self

    def __init__(self, magnitude: float, unit: TemperatureUnit):
        self.magnitude = magnitude
        self.unit = unit


class StressUnit(Enum):
    PSI = "psi"
    PA = "Pa"
    MPA = "MPa"


class Stress(Unit):
    primary_unit = StressUnit.PSI
    _factors_to_primary = {
        StressUnit.PSI: 1.0,
        StressUnit.PA: 1 / 6894.757,
        StressUnit.MPA: 145.038,
    }

    def __init__(self, magnitude: float, unit: StressUnit):
        self.magnitude = magnitude
        self.unit = unit


class InverseLengthUnit(Enum):
    PER_INCH = "1/inch"
    PER_MM = "1/mm"


class InverseLength(Unit):
    primary_unit = InverseLengthUnit.PER_MM
    _factors_to_primary = {
        InverseLengthUnit.PER_INCH: 1 / 25.4,
        InverseLengthUnit.PER_MM: 1.0,
    }

    def __init__(self, magnitude: float, unit: InverseLengthUnit):
        self.magnitude = magnitude
        self.unit = unit


class AngularVelocityUnit(Enum):
    RPM = "rpm"
    RPS = "rps"
    RAD_per_SEC = "radps"


class AngularVelocity(Unit):
    primary_unit = AngularVelocityUnit.RPS
    _factors_to_primary = {
        AngularVelocityUnit.RPM: 1 / 60,
        AngularVelocityUnit.RPS: 1.0,
        AngularVelocityUnit.RAD_per_SEC: 1 / (2 * pi),
    }

    def __init__(self, magnitude: float, unit: AngularVelocityUnit):
        self.magnitude = magnitude
        self.unit = unit


class AccelerationUnit(Enum):
    MPS2 = "m/s²"
    FPS2 = "ft/s²"


class Acceleration(Unit):
    primary_unit = AccelerationUnit.MPS2
    _factors_to_primary = {
        AccelerationUnit.MPS2: 1.0,
        AccelerationUnit.FPS2: 0.3048,
    }

    def __init__(self, magnitude: float, unit: AccelerationUnit):
        self.magnitude = magnitude
        self.unit = unit


class PowerUnit(Enum):
    WATT = "W"
    HP = "hp"


class Power(Unit):
    primary_unit = PowerUnit.WATT
    _factors_to_primary = {
        PowerUnit.WATT: 1.0,
        PowerUnit.HP: 745.7,
    }

    def __init__(self, magnitude: float, unit: PowerUnit):
        self.magnitude = magnitude
        self.unit = unit


class ForceUnit(Enum):
    LBF = "lbf"
    NEWTON = "N"


class Force(Unit):
    primary_unit = ForceUnit.NEWTON
    _factors_to_primary = {
        ForceUnit.LBF: 4.44822,
        ForceUnit.NEWTON: 1.0,
    }

    def __init__(self, magnitude: float, unit: ForceUnit):
        self.magnitude = magnitude
        self.unit = unit


class SqrtPressureUnit(Enum):
    SQRT_PSI = "sqrt(psi)"
    SQRT_PASCAL = "sqrt(Pa)"


class SqrtPressure(Unit):
    primary_unit = SqrtPressureUnit.SQRT_PASCAL
    _factors_to_primary = {
        SqrtPressureUnit.SQRT_PSI: np.sqrt(6894.757),
        SqrtPressureUnit.SQRT_PASCAL: 1.0,
    }

    def __init__(self, magnitude: float, unit: SqrtPressureUnit):
        self.magnitude = magnitude
        self.unit = unit


class TorqueUnit(Enum):
    LBF_IN = "lbf-in"
    Nm = "N·m"


class Torque(Unit):
    primary_unit = TorqueUnit.Nm
    _factors_to_primary = {
        TorqueUnit.LBF_IN: 0.113,
        TorqueUnit.Nm: 1.0,
    }

    def __init__(self, magnitude: float, unit: TorqueUnit):
        self.magnitude = magnitude
        self.unit = unit


class DensityUnit(Enum):
    KG_PER_M3 = "kg/m³"
    LB_PER_IN3 = "lb/in³"


class Density(Unit):
    primary_unit = DensityUnit.KG_PER_M3
    _factors_to_primary = {
        DensityUnit.KG_PER_M3: 1.0,
        DensityUnit.LB_PER_IN3: 27679.9,
    }

    def __init__(self, magnitude: float, unit: DensityUnit):
        self.magnitude = magnitude
        self.unit = unit


class TimeUnit(Enum):
    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"
    DAY = "day"
    YEAR = "year"


class Time(Unit):
    primary_unit = TimeUnit.SECOND
    _factors_to_primary = {
        TimeUnit.SECOND: 1.0,
        TimeUnit.MINUTE: 60.0,
        TimeUnit.HOUR: 3600.0,
        TimeUnit.DAY: 86400.0,
        TimeUnit.YEAR: 31536000.0,
    }

    def __init__(self, magnitude: float, unit: TimeUnit):
        self.magnitude = magnitude
        self.unit = unit


class MassUnit(Enum):
    KG = "kg"
    LB = "lb"


class Mass(Unit):
    primary_unit = MassUnit.KG
    _factors_to_primary = {
        MassUnit.KG: 1.0,
        MassUnit.LB: 0.453592,
    }

    def __init__(self, magnitude: float, unit: MassUnit):
        self.magnitude = magnitude
        self.unit = unit
