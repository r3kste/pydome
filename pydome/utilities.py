import pytest


def approx(val, rel=1e-3):
    return pytest.approx(val, rel=rel)


def vector_magnitude(*args) -> float:
    S = sum(arg**2 for arg in args)
    return S**0.5
