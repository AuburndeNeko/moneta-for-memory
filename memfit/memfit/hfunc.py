"""
Transition functions H(s), s = t/T, given as presets or as safe math expressions.

An expression may use the variable s, numbers, + - * / ** %, comparisons, and these
functions: sin cos tan exp log log10 sqrt abs tanh sinh cosh arctan arcsin arccos
erf erfc sigmoid heaviside where clip minimum maximum sign; constants pi and e.
Anything else (attributes, imports, indexing, lambdas, ...) is rejected.
"""
import ast
import numpy as np
from scipy.special import erf, erfc, expit

PRESETS = {
    "smooth": ("0.5*(1 + tanh(2*s))",
               "monotone smooth step (logistic, 10-90% rise = 1.10 T); the paper's omega = 0 style"),
    "oscillating": ("0.5*(1 + tanh(2*s)) + 0.15*sin(10*s)*exp(-s**2)",
                    "smooth step with ringing during the rise (stand-in for the paper's omega = 10)"),
    "erf": ("0.5*(1 + erf(s))", "Gaussian-integral step"),
    "sharp": ("0.5*(1 + tanh(20*s))", "nearly instantaneous step (10-90% rise = 0.11 T)"),
    "ramp": ("clip(s + 0.5, 0, 1)", "constant acceleration for a time T (piecewise linear)"),
    "two-stage": ("0.25*(1 + tanh(3*(s + 1.5))) + 0.25*(1 + tanh(3*(s - 1.5)))",
                  "two kicks of half the velocity change, 3 T apart"),
}

_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh, "sinh": np.sinh,
    "cosh": np.cosh, "arctan": np.arctan, "arcsin": np.arcsin, "arccos": np.arccos,
    "erf": erf, "erfc": erfc, "sigmoid": expit, "sign": np.sign,
    "heaviside": lambda x: np.heaviside(x, 0.5), "where": np.where, "clip": np.clip,
    "minimum": np.minimum, "maximum": np.maximum,
}
_CONSTS = {"pi": np.pi, "e": np.e}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
            ast.Constant, ast.Compare, ast.BoolOp, ast.IfExp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or)


def compile_expr(expr):
    """Return (H, expression string). H maps an array of s to an array."""
    if not isinstance(expr, str) or len(expr) > 500:
        raise ValueError("H must be a string of at most 500 characters.")
    expr = PRESETS.get(expr, (expr,))[0]
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Could not parse H expression: {err.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"Not allowed in H expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id not in _CONSTS \
                and node.id != "s":
            raise ValueError(f"Unknown name in H expression: '{node.id}'. "
                             "Use the variable s and the listed functions.")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in _FUNCS):
            raise ValueError("Only the listed functions can be called in H.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed in H.")
    code = compile(tree, "<H>", "eval")
    env = {"__builtins__": {}, **_FUNCS, **_CONSTS}

    def H(s):
        s = np.asarray(s, float)
        with np.errstate(all="ignore"):
            out = eval(code, env, {"s": s})
        return np.broadcast_to(np.asarray(out, float), s.shape).copy()

    return H, expr


def validate(H, tol=1e-3):
    """H must run from 0 (far past) to 1 (far future) and be finite."""
    s = np.linspace(-60, 60, 24001)
    h = H(s)
    if not np.all(np.isfinite(h)):
        raise ValueError("H is not finite everywhere (check for division by zero or log of <= 0).")
    if abs(h[0]) > tol or abs(h[-1] - 1) > tol:
        raise ValueError(f"H must go from 0 to 1: H(-60) = {h[0]:.4g}, H(60) = {h[-1]:.4g}. "
                         "Memory is defined by the initial and final velocities, so H must "
                         "settle at 0 before and 1 after the transition.")
    return h


def transition_metrics(t, g):
    """Time of first 0.5 crossing and 10-90% rise time of a step g(t)."""
    def first_cross(level):
        i = int(np.argmax(g >= level))
        if i == 0 or g[i] < level:
            return float("nan")
        return float(np.interp(level, g[i - 1:i + 1], t[i - 1:i + 1]))
    return first_cross(0.5), first_cross(0.9) - first_cross(0.1)
