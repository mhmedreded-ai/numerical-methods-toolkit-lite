"""Symbolic expression parsing, safe numerical evaluation, and matrix conditioning."""

import itertools
import math
from typing import Callable, List, Tuple
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCAL_DICT = {"e": sp.E, "E": sp.E, "pi": sp.pi, "ln": sp.log}


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
