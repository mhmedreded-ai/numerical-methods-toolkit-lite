"""Root-finding algorithms: Bisection, Newton-Raphson, Secant, and Modified Secant."""

import math
from typing import Any, Callable, Dict, List, Tuple

StepList = List[Dict[str, Any]]
RootResult = Tuple[float, StepList, bool, str]


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
