from typing import Callable
from enum import Enum

import optimagic as om
from pydome.units import Unit


class ValueType(Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"


UNKNOWN = ValueType.UNKNOWN


def solve_for_parameters(
    *,
    func: Callable,
    known_params: dict[str, Unit | float],
    unknown_params: dict[str],
    target_value: Unit | float,
    bounds: dict[str, tuple],
    constraints: list | None = None,
) -> tuple[om.OptimizeResult, dict[str, Unit | float]]:
    """
    Numerically solve for the values of `unknown_params` that make
    func(**known_params, **unknown_params=?) == target_value using scipy.optimize.minimize.

    Args:
        func (Callable): Function to evaluate with
        known_params (dict[str, Unit | float]): Known parameters for the function.
        unknown_params (dict[str]): Mapping from unknown_params to their types (Unit or float).
        target_value (Unit | float): Desired target value of the function.
        bounds (dict[str, tuple]): Bounds for each unknown parameter as {param: (min, max)}.

    Returns:
        tuple[om.OptimizeResult, dict[str, Unit | float]]: The optimization result
        and a dictionary of the solved parameters converted back to their original units.
    """

    target_value_mag = _get_magnitude(target_value)

    lower_bounds = {}
    upper_bounds = {}
    for param in unknown_params:
        lb, ub = bounds[param]
        lower_bounds[param] = _get_magnitude(lb)
        upper_bounds[param] = _get_magnitude(ub)
    om_bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

    initial_guess = {
        param: (lower_bounds[param] + upper_bounds[param]) / 2
        for param in unknown_params
    }

    def objective(x: dict[str, float]) -> float:
        for param, unit_type in unknown_params.items():
            if unit_type is int:
                value = int(x[param])
            elif unit_type is float:
                value = x[param]
            else:
                value = unit_type(x[param], unit_type.primary_unit)

            parts = param.split(".")
            obj = known_params
            for part in parts[:-1]:
                if isinstance(obj, dict):
                    obj = obj[part]
                else:
                    obj = getattr(obj, part)

            last_part = parts[-1]

            if isinstance(obj, dict):
                obj[last_part] = value
            else:
                setattr(obj, last_part, value)

        result = func(**known_params)
        result_mag = _get_magnitude(result)
        return (result_mag - target_value_mag) ** 2

    res = om.minimize(
        objective,
        params=initial_guess,
        bounds=om_bounds,
        constraints=constraints,
        algorithm="scipy_slsqp",
    )
    if not res.success:
        raise RuntimeError(f"Minimization did not converge: {res.message}")

    return (
        res,
        {
            param: unknown_params[param](
                res.params[param], unknown_params[param].primary_unit
            )
            if unknown_params[param] is not float
            else res.params[param]
            for param in res.params
        },
    )


def _get_magnitude(value: Unit | float) -> float:
    if isinstance(value, Unit):
        return value.to_primary().magnitude
    return value
