from pydome.solver import solve_for_parameters
from pydome.units import Length, LengthUnit
from pydome.utilities import approx


def test_solve_for_parameter_simple():
    def f(x: float) -> float:
        return x * 2 + 3

    known_params = {}
    unknown_params = {"x": float}
    target_value = 13
    bounds = {"x": (0, 10)}

    result, params = solve_for_parameters(
        func=f,
        known_params=known_params,
        unknown_params=unknown_params,
        target_value=target_value,
        bounds=bounds,
    )

    assert params["x"] == approx(5.0)


def test_solve_for_parameter_with_unit():
    def f(x: Length) -> Length:
        x_mag = x.to(LengthUnit.MM).magnitude
        return Length(x_mag * 3 + 5, LengthUnit.MM)

    known_params = {}
    unknown_params = {"x": Length}
    target_value = Length(20, LengthUnit.MM)
    bounds = {"x": (Length(0, LengthUnit.MM), Length(10, LengthUnit.MM))}

    result, params = solve_for_parameters(
        func=f,
        known_params=known_params,
        unknown_params=unknown_params,
        target_value=target_value,
        bounds=bounds,
    )

    assert params["x"].equals(Length(5, LengthUnit.MM))


def test_solve_for_multiple_parameters():
    def f(x: float, y: float) -> float:
        return x**2 + y**2

    known_params = {}
    unknown_params = {"x": float, "y": float}
    target_value = 50
    bounds = {"x": (0, 10), "y": (0, 10)}

    result, params = solve_for_parameters(
        func=f,
        known_params=known_params,
        unknown_params=unknown_params,
        target_value=target_value,
        bounds=bounds,
    )

    x = params["x"]
    y = params["y"]
    assert approx(x**2 + y**2) == 50


def test_solve_for_parameter_nested():
    from dataclasses import dataclass

    @dataclass
    class Adder:
        a: float
        b: float

    def f(adder: Adder) -> float:
        return adder.a + adder.b

    known_params = {"adder": Adder(a=2, b=3)}
    unknown_params = {"adder.b": float}
    target_value = 11
    bounds = {"adder.b": (0, 10)}

    result, params = solve_for_parameters(
        func=f,
        known_params=known_params,
        unknown_params=unknown_params,
        target_value=target_value,
        bounds=bounds,
    )

    assert params["adder.b"] == approx(9.0)
