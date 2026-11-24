"""Iterative linear system solvers: Jacobi and Gauss-Seidel."""

import math
from typing import Any, Dict, List, Optional, Tuple
from core.math_utils import make_diagonally_dominant

LinearResult = Tuple[List[float], List[Dict[str, Any]], bool, str]


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
    return x, steps, False, f"تم الوصول للحد الأقصى للتكرارات ({max_iter}) دون الوصول للدقة المطلوبة."


def jacobi(A: List[List[float]], b: List[float], x0: Optional[List[float]] = None, tol: float = 1e-3, max_iter: int = 50) -> LinearResult:
    """Solve linear system A x = b using simultaneous Jacobi relaxation."""
    return _iterative_solve(A, b, in_place=False, x0=x0, tol=tol, max_iter=max_iter)


def gauss_seidel(A: List[List[float]], b: List[float], x0: Optional[List[float]] = None, tol: float = 1e-3, max_iter: int = 50) -> LinearResult:
    """Solve linear system A x = b using in-place Gauss-Seidel successive displacements."""
    return _iterative_solve(A, b, in_place=True, x0=x0, tol=tol, max_iter=max_iter)
