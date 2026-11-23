import numpy as np
import pytest

from memfit.physics import C, Source, observe, memory_prediction
from memfit.hfunc import compile_expr, validate
from memfit.fitting import Data, fit


def test_parser_rejects_code():
    for bad in ["__import__('os')", "s.__class__", "open('x')", "[1][0]", "lambda: 1", "'a'"]:
        with pytest.raises(ValueError):
            compile_expr(bad)


def test_validate_requires_0_to_1():
    with pytest.raises(ValueError):
        validate(compile_expr("tanh(s)")[0])          # goes from -1 to 1
    validate(compile_expr("smooth")[0])


def test_memory_independent_of_shape():
    x = np.array([200.0, 0, 0])
    for name in ["smooth", "oscillating", "sharp", "two-stage"]:
        H = compile_expr(name)[0]
        o = observe(Source(H), x, (-10, 10), 4000)
        dA, _ = memory_prediction(1.0, np.zeros(3), np.array([0, 0.5 * C, 0]), x)
        assert abs((o["A"][-1, 1] - o["A"][0, 1]) / dA[1] - 1) < 0.01


@pytest.mark.parametrize("observable", ["A", "E"])
def test_fit_recovers_memory(observable):
    x = np.array([10.0, 0, 0])
    o = observe(Source(compile_expr("smooth")[0]), x)
    Y = o[observable]
    rng = np.random.default_rng(1)
    Yn = Y + 0.05 * np.abs(Y - Y[0]).max() * rng.standard_normal(Y.shape)
    res = fit(Data(o["t"], Yn, observable, x))
    _, Delta = memory_prediction(1.0, np.zeros(3), np.array([0, 0.5 * C, 0]), x)
    assert abs(res["Delta"][1] - Delta[1]) < 4 * res["Delta_err"][1]
    assert 0.9 < res["red_chi2"] < 1.1
    assert res["dchi2"] > 100


def test_no_memory_when_velocity_unchanged():
    x = np.array([10.0, 0, 0])
    v = np.array([0, 0.5 * C, 0])
    o = observe(Source(compile_expr("smooth")[0], v_in=v, v_out=v), x)
    Y = o["A"]
    rng = np.random.default_rng(2)
    Yn = Y + 0.05 * 0.15 * rng.standard_normal(Y.shape)
    res = fit(Data(o["t"], Yn, "A", x))
    assert abs(res["Delta"][1]) < 5 * res["Delta_err"][1]
    assert res["dchi2"] < 40


def test_erf_transition_with_E_data():
    """Regression: with logistic-only tails, E data from an erf step gave +5% (8 sigma) bias."""
    x = np.array([10.0, 0, 0])
    o = observe(Source(compile_expr("erf")[0]), x)
    Y = o["E"]
    rng = np.random.default_rng(0)
    Yn = Y + 0.05 * np.abs(Y - Y[0]).max() * rng.standard_normal(Y.shape)
    res = fit(Data(o["t"], Yn, "E", x))
    _, Delta = memory_prediction(1.0, np.zeros(3), np.array([0, 0.5 * C, 0]), x)
    assert abs(res["Delta"][1] - Delta[1]) < 3 * res["Delta_err"][1]
    assert abs(res["Delta"][1] / Delta[1] - 1) < 0.025
