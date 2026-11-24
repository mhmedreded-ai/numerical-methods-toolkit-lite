"""Least-squares regression and curve fitting powered by NumPy."""

import math
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


def _validate_xy(x: List[float], y: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xa) != len(ya):
        raise ValueError(f"عدد قيم X ({len(xa)}) لا يساوي عدد قيم Y ({len(ya)}).")
    if len(xa) < 2:
        raise ValueError("يلزم نقطتان على الأقل لحساب الانحدار ومواءمة المنحنى.")
    return xa, ya


def _fit_line(u: np.ndarray, v: np.ndarray, err_msg: str) -> Tuple[float, float]:
    if float(np.sum((u - u.mean()) ** 2)) < 1e-14:
        raise ValueError(err_msg)
    slope, intercept = np.polyfit(u, v, 1)
    return float(intercept), float(slope)


def _pack(method: str, name: str, eq: str, coeffs: List[float], xa: np.ndarray, ya: np.ndarray,
          predict: Callable[[float], float], num_params: int, sign_r: int = 1) -> Dict[str, Any]:
    y_pred = [float(predict(float(xi))) for xi in xa]
    residuals = [float(yi - yp) for yi, yp in zip(ya, y_pred)]
    st, sr = float(np.sum((ya - np.mean(ya)) ** 2)), sum(e ** 2 for e in residuals)
    r2 = max(0.0, min(1.0, 1.0 - sr / st)) if st > 1e-14 else (1.0 if sr < 1e-14 else 0.0)
    df, x_min, x_max = len(xa) - num_params, float(xa.min()), float(xa.max())
    x_curve = [x_min] if x_min == x_max else np.linspace(x_min, x_max, 200).tolist()
    return {
        "method": method, "method_name": name, "equation": eq, "coefficients": coeffs,
        "y_pred": y_pred, "residuals": residuals, "r2": r2, "r": sign_r * math.sqrt(r2),
        "syx": math.sqrt(max(0.0, sr / df)) if df > 0 else 0.0, "sr": sr, "st": st,
        "x_curve": x_curve, "y_curve": [float(predict(xi)) for xi in x_curve],
    }


def linear_regression(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Fit linear model y = a0 + a1 * x."""
    xa, ya = _validate_xy(x, y)
    a0, a1 = _fit_line(xa, ya, "جميع قيم X متطابقة، لا يمكن حساب الانحدار الخطي.")
    eq = f"y = {a0:.5f} {'+' if a1 >= 0 else '-'} {abs(a1):.5f} * x"
    return _pack("Linear", "الانحدار الخطي (Linear Regression)", eq, [a0, a1],
                 xa, ya, lambda v: a0 + a1 * v, 2, sign_r=1 if a1 >= 0 else -1)


def polynomial_regression(x: List[float], y: List[float], degree: int = 2) -> Dict[str, Any]:
    """Fit polynomial model y = a0 + a1*x + ... + am*x^m."""
    xa, ya = _validate_xy(x, y)
    m = int(degree)
    if m < 1:
        raise ValueError("درجة كثير الحدود يجب أن تكون 1 على الأقل.")
    if len(xa) <= m:
        raise ValueError(f"عدد النقاط ({len(xa)}) يجب أن يكون أكبر من درجة كثير الحدود ({m}).")
    poly_desc = np.polyfit(xa, ya, m)
    coeffs = [float(c) for c in poly_desc[::-1]]
    terms = [f"{coeffs[0]:.5f}"] + [
        f"{'+' if c >= 0 else '-'} {abs(c):.5f}*x" + (f"^{p}" if p > 1 else "")
        for p, c in enumerate(coeffs[1:], 1)
    ]
    return _pack("Polynomial", f"انحدار متعدد الحدود من الدرجة {m} (Polynomial Degree {m})",
                 "y = " + " ".join(terms), coeffs, xa, ya, lambda v: float(np.polyval(poly_desc, v)), m + 1)


def exponential_regression(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Fit exponential model y = a * e^(b * x) (requires y > 0)."""
    xa, ya = _validate_xy(x, y)
    if np.any(ya <= 0):
        raise ValueError("المواءمة الأسية تتطلب أن تكون جميع قيم Y موجبة تماماً (Y > 0).")
    ln_a, b = _fit_line(xa, np.log(ya), "جميع قيم X متطابقة، لا يمكن حساب المواءمة الأسية.")
    a = math.exp(ln_a)
    return _pack("Exponential", "المواءمة الأسية (Exponential Fit: y = a*e^(bx))",
                 f"y = {a:.5f} * e^({b:.5f} * x)", [a, b], xa, ya, lambda v: a * math.exp(b * v), 2)


def power_regression(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Fit power model y = a * x^b (requires x > 0 and y > 0)."""
    xa, ya = _validate_xy(x, y)
    if np.any(xa <= 0) or np.any(ya <= 0):
        raise ValueError("مواءمة دالة القوى تتطلب أن تكون جميع قيم X و Y موجبة تماماً (> 0).")
    ln_a, b = _fit_line(np.log(xa), np.log(ya), "جميع قيم ln(X) متطابقة، لا يمكن حساب مواءمة القوى.")
    a = math.exp(ln_a)
    return _pack("Power", "مواءمة دالة القوى (Power Fit: y = a*x^b)",
                 f"y = {a:.5f} * x^({b:.5f})", [a, b], xa, ya, lambda v: a * (v ** b), 2)


def fit_curve(x: List[float], y: List[float], method: str = "linear", degree: int = 2) -> Dict[str, Any]:
    """Unified curve-fitting dispatcher."""
    dispatch = {
        "linear": linear_regression, "خطي": linear_regression,
        "polynomial": lambda xs, ys: polynomial_regression(xs, ys, degree=degree),
        "متعدد الحدود": lambda xs, ys: polynomial_regression(xs, ys, degree=degree),
        "exponential": exponential_regression, "أسي": exponential_regression, "اسي": exponential_regression,
        "power": power_regression, "قوى": power_regression,
    }
    if method.lower() not in dispatch:
        raise ValueError(f"طريقة غير معروفة: {method}. الطرق المتاحة: linear, polynomial, exponential, power.")
    return dispatch[method.lower()](x, y)
