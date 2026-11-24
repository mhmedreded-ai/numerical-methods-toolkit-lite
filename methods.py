"""Math 3 Numerical Methods Engine: Parsing, Roots, Linear Systems, and Regression."""

import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCAL_DICT = {"e": sp.E, "E": sp.E, "pi": sp.pi, "ln": sp.log}
StepList = List[Dict[str, Any]]
RootResult = Tuple[float, StepList, bool, str]
LinearResult = Tuple[List[float], StepList, bool, str]


# =============================================================================
# 1. Symbolic Expression Parsing, Derivatives & Matrix Diagonal Dominance
# =============================================================================
def _wrap_safe(func: Callable[[float], float], label: str = "f") -> Callable[[float], float]:
    """Wrap a lambdified function with float conversion and NaN/Inf/domain guards."""
    def safe(val: float) -> float:
        try:
            res = float(func(float(val)))
            if math.isnan(res) or math.isinf(res):
                raise ValueError("قيمة غير معرفة أو غير منتهية (NaN / Infinity)")
            return res
        except (ValueError, OverflowError, ZeroDivisionError, TypeError) as err:
            raise ValueError(f"خطأ رياضي عند حساب {label}({val}): {err}") from err
    return safe


def parse_math_expression(expr_str: str) -> Tuple[sp.Expr, Callable[[float], float], sp.Symbol]:
    """Parse a formula string in variable 'x' into (sympy_expr, safe_callable, x_symbol)."""
    clean = expr_str.strip()
    if not clean:
        raise ValueError("تعبير رياضي فارغ! يرجى إدخال معادلة صالحة بدلالة x.")
    x = sp.Symbol("x")
    expr = parse_expr(clean, transformations=TRANSFORMATIONS, local_dict=LOCAL_DICT)
    extra = expr.free_symbols - {x}
    if extra:
        raise ValueError(f"تم العثور على رموز غير معروفة: {', '.join(map(str, extra))} - استخدم المتغير x فقط.")
    return expr, _wrap_safe(sp.lambdify(x, expr, modules=["math"]), "f"), x


def make_derivative(expr: sp.Expr, x: sp.Symbol) -> Tuple[sp.Expr, Callable[[float], float]]:
    """Differentiate expr symbolically with respect to x and return (diff_expr, safe_callable)."""
    diff_expr = sp.diff(expr, x)
    return diff_expr, _wrap_safe(sp.lambdify(x, diff_expr, modules=["math"]), "f'")


def check_diagonally_dominant(A: List[List[float]]) -> Tuple[bool, bool]:
    """Return (is_dominant, is_strict) for square matrix A."""
    strict = True
    for i, row in enumerate(A):
        diag, off_diag = abs(row[i]), sum(abs(v) for j, v in enumerate(row) if j != i)
        if diag < off_diag:
            return False, False
        if diag == off_diag:
            strict = False
    return True, strict


def make_diagonally_dominant(A: List[List[float]], b: List[float]) -> Tuple[List[List[float]], List[float], bool, str]:
    """Reorder rows of A x = b to achieve diagonal dominance when possible."""
    is_dom, is_strict = check_diagonally_dominant(A)
    if is_dom:
        return A, b, True, ("المصفوفة مهيمنة قطرياً بشكل صارم" if is_strict else "المصفوفة مهيمنة قطرياً بشكل ضعيف")
    n = len(A)
    if n <= 8:
        for perm in itertools.permutations(range(n)):
            perm_A = [A[i] for i in perm]
            if check_diagonally_dominant(perm_A)[0]:
                return perm_A, [b[i] for i in perm], True, "تمت إعادة ترتيب المعادلات تلقائياً لتحقيق السيادة القطرية"
    perm, used = [-1] * n, set()
    for col in range(n):
        cands = [(abs(A[r][col]), r) for r in range(n) if r not in used and abs(A[r][col]) > 0]
        if cands:
            best_row = max(cands)[1]
            perm[col] = best_row
            used.add(best_row)
    if len(used) == n:
        new_A, new_b = [A[i] for i in perm], [b[i] for i in perm]
        if check_diagonally_dominant(new_A)[0]:
            return new_A, new_b, True, "تمت إعادة ترتيب المعادلات لتحقيق السيادة القطرية"
        return new_A, new_b, False, "تم ترتيب المعادلات لتقليل أصفار القطر، لكن السيادة القطرية غير محققة بالكامل"
    return A, b, False, "تنبيه: المصفوفة ليست مهيمنة قطرياً، قد تتباعد الطريقة (Diverge)"


# =============================================================================
# 2. Root-Finding Algorithms (Bisection, Newton-Raphson, Secant, Modified Secant)
# =============================================================================
def _diverged(val: float) -> bool:
    return math.isnan(val) or math.isinf(val) or abs(val) > 1e12


def _max_iter_msg(max_iter: int) -> str:
    return f"تم الوصول للحد الأقصى للتكرارات ({max_iter}) دون الوصول للدقة المطلوبة."


def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-3, max_iter: int = 50) -> RootResult:
    """Find a root of f(x) in [a, b] via interval halving."""
    if a > b:
        a, b = b, a
    elif a == b:
        raise ValueError("بداية الفترة ونهايتها متطابقتان (a == b).")
    fa, fb = f(a), f(b)
    for name, pt, fval in (("a", a, fa), ("b", b, fb)):
        if abs(fval) < 1e-12:
            return pt, [], True, f"النقطة {name} = {pt} هي جذر دقيق للمعادلة: f({pt}) = 0"
    if fa * fb > 0:
        raise ValueError(f"f({a}) = {fa:.4f} و f({b}) = {fb:.4f} لهما نفس الإشارة! شرط الطريقة f(a)*f(b) < 0 غير محقق.")
    steps, c = [], (a + b) / 2.0
    for i in range(1, max_iter + 1):
        c, err = (a + b) / 2.0, (b - a) / 2.0
        fc = f(c)
        steps.append({"iter": i, "a": a, "b": b, "c": c, "fc": fc, "error": err})
        if abs(fc) < 1e-12 or err < tol:
            return c, steps, True, f"تم الوصول للجذر: c = {c:.6f} بعد {i} دورة (الخطأ = {err:.6e})"
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, steps, False, _max_iter_msg(max_iter)


def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-3, max_iter: int = 50) -> RootResult:
    """Find a root of f(x) using tangent projections with derivative df(x)."""
    if abs(f(x0)) < 1e-12:
        return x0, [], True, f"النقطة الابتدائية x0 = {x0} هي جذر دقيق للمعادلة: f({x0}) = 0"
    steps, x = [], x0
    for i in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if abs(dfx) < 1e-14:
            return x, steps, False, f"توقف: المشتقة تساوي صفراً عند x = {x:.5f}، لا يمكن إكمال القسمة."
        x_next = x - fx / dfx
        err = abs(x_next - x)
        steps.append({"iter": i, "xn": x, "fx": fx, "dfx": dfx, "x_next": x_next, "error": err})
        if _diverged(x_next):
            return x, steps, False, "تنبيه: الطريقة تتباعد (Diverges)! تحقق من اختيار نقطة بداية أقرب للجذر."
        if err < tol or abs(f(x_next)) < 1e-12:
            return x_next, steps, True, f"تم الوصول للجذر: x = {x_next:.6f} بعد {i} دورة (الخطأ = {err:.6e})"
        x = x_next
    return x, steps, False, _max_iter_msg(max_iter)


def secant(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-3, max_iter: int = 50) -> RootResult:
    """Find a root of f(x) using two-point secant finite differences."""
    if x0 == x1:
        raise ValueError("يجب إدخال نقطتين مختلفتين (x0 != x1).")
    f0, f1 = f(x0), f(x1)
    for name, pt, fval in (("x0", x0, f0), ("x1", x1, f1)):
        if abs(fval) < 1e-12:
            return pt, [], True, f"النقطة {name} = {pt} هي جذر دقيق للمعادلة: f({pt}) = 0"
    steps = []
    for i in range(1, max_iter + 1):
        denom = f1 - f0
        if abs(denom) < 1e-14:
            return x1, steps, False, "توقف: فرق قيم الدالة (f1 - f0) قريب جداً من الصفر، لا يمكن القسمة."
        x2 = x1 - f1 * (x1 - x0) / denom
        err = abs(x2 - x1)
        f2 = float("nan") if _diverged(x2) else f(x2)
        steps.append({"iter": i, "x_prev": x0, "x_curr": x1, "x_next": x2, "fx_next": f2, "error": err})
        if _diverged(x2):
            return x1, steps, False, "تنبيه: الطريقة تتباعد (Diverges)! تحقق من اختيار نقاط بداية أكثر ملاءمة."
        if err < tol or abs(f2) < 1e-12:
            return x2, steps, True, f"تم الوصول للجذر: x = {x2:.6f} بعد {i} دورة (الخطأ = {err:.6e})"
        x0, f0, x1, f1 = x1, f1, x2, f2
    return x1, steps, False, _max_iter_msg(max_iter)


def modified_secant(f: Callable[[float], float], x0: float, delta: float = 0.01, tol: float = 1e-3, max_iter: int = 50) -> RootResult:
    """Find a root of f(x) using single-point fractional perturbation delta."""
    if abs(delta) < 1e-15:
        return x0, [], False, "قيمة delta صغيرة جداً أو تساوي صفراً."
    f0 = f(x0)
    if abs(f0) < 1e-12:
        return x0, [], True, f"النقطة الابتدائية x0 = {x0} هي جذر دقيق للمعادلة: f({x0}) = 0"
    steps, x = [], x0
    for i in range(1, max_iter + 1):
        h = delta * x if abs(x) > 1e-12 else delta
        f_pert = f(x + h)
        denom = f_pert - f0
        if abs(denom) < 1e-14:
            return x, steps, False, "مقام قانون القاطع المعدل f(x+h) - f(x) يساوي صفراً، لا يمكن إكمال القسمة."
        x_next = x - (h * f0) / denom
        err = abs(x_next - x)
        steps.append({"iter": i, "xi": x, "fxi": f0, "f_pert": f_pert, "x_next": x_next, "error": err})
        if _diverged(x_next):
            return x_next, steps, False, "النظام متباعد (Divergent)! قيم x تتزايد بشكل غير محدود."
        f_next = f(x_next)
        if err < tol or abs(f_next) < 1e-12:
            return x_next, steps, True, f"تم الوصول للجذر: x = {x_next:.6f} بعد {i} دورة (الخطأ = {err:.6e})"
        x, f0 = x_next, f_next
    return x, steps, False, _max_iter_msg(max_iter)


# =============================================================================
# 3. Iterative Linear System Solvers (Jacobi & Gauss-Seidel)
# =============================================================================
def _iterative_solve(
    A: List[List[float]], b: List[float], in_place: bool,
    x0: Optional[List[float]] = None, tol: float = 1e-3, max_iter: int = 50,
) -> LinearResult:
    """Solve A x = b iteratively (simultaneously if in_place=False, Gauss-Seidel if True)."""
    n = len(A)
    if n < 2:
        raise ValueError("يجب أن يحتوي النظام على معادلتين على الأقل.")
    if n != len(b) or any(len(row) != n for row in A):
        raise ValueError("أبعاد المصفوفة A ومتجه الثوابت b غير متطابقة.")
    if x0 is not None and len(x0) != n:
        raise ValueError(f"التخمين الابتدائي يجب أن يحتوي على {n} أرقام.")

    A_c, b_c, _, _ = make_diagonally_dominant(A, b)
    for i in range(n):
        if abs(A_c[i][i]) < 1e-12:
            raise ValueError(f"العنصر القطري A[{i + 1}][{i + 1}] يساوي صفراً، لا يمكن القسمة عليه.")

    x, steps = (list(x0) if x0 is not None else [0.0] * n), []
    for it in range(1, max_iter + 1):
        x_old = list(x)
        ref = x if in_place else x_old
        x_next = x if in_place else [0.0] * n
        for i in range(n):
            off_sum = sum(A_c[i][j] * ref[j] for j in range(n) if j != i)
            x_next[i] = (b_c[i] - off_sum) / A_c[i][i]
        x = x_next
        err = max(abs(x[i] - x_old[i]) for i in range(n))
        steps.append({"iter": it, "x": list(x), "error": err})
        if math.isnan(err) or math.isinf(err) or err > 1e12:
            return x, steps, False, "النظام متباعد (Divergent)! تحقق من شرط السيادة القطرية."
        if err < tol:
            return x, steps, True, f"تم الوصول للحل بنجاح بعد {it} دورة (الخطأ = {err:.6e})."
    return x, steps, False, _max_iter_msg(max_iter)


def jacobi(A: List[List[float]], b: List[float], x0: Optional[List[float]] = None, tol: float = 1e-3, max_iter: int = 50) -> LinearResult:
    """Solve linear system A x = b using simultaneous Jacobi relaxation."""
    return _iterative_solve(A, b, in_place=False, x0=x0, tol=tol, max_iter=max_iter)


def gauss_seidel(A: List[List[float]], b: List[float], x0: Optional[List[float]] = None, tol: float = 1e-3, max_iter: int = 50) -> LinearResult:
    """Solve linear system A x = b using in-place Gauss-Seidel successive displacements."""
    return _iterative_solve(A, b, in_place=True, x0=x0, tol=tol, max_iter=max_iter)


# =============================================================================
# 4. Curve Fitting & Least-Squares Regression (Linear, Polynomial, Exp, Power)
# =============================================================================
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
