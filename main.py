"""Math 3 Numerical Methods Suite - Unified CLI Runner & Desktop Launcher."""

import os
import subprocess
import sys

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import core

ROOT_CLI = {
    "1": (("Iter", "a", "b", "c", "f(c)", "Error"), ("a", "b", "c", "fc")),
    "2": (("Iter", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"), ("xn", "fx", "dfx", "x_next")),
    "3": (("Iter", "x_(n-1)", "x_n", "x_(n+1)", "f(x_(n+1))", "Error"), ("x_prev", "x_curr", "x_next", "fx_next")),
    "4": (("Iter", "x_i", "f(x_i)", "f(x_i+h)", "x_(i+1)", "Error"), ("xi", "fxi", "f_pert", "x_next")),
}


def _ask(prompt: str, cast=str, default=None):
    raw = input(prompt).strip()
    return default if (not raw and default is not None) else cast(raw)


def _ask_nums(prompt: str):
    return [float(v) for v in input(prompt).replace(",", " ").split()]


def _print_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows), 8) + 2 for i, h in enumerate(headers)]
    sep = "-" * sum(widths)
    print("\n" + sep + "\n" + "".join(str(h).ljust(w) for h, w in zip(headers, widths)) + "\n" + sep)
    for row in rows:
        print("".join(str(val).ljust(w) for val, w in zip(row, widths)))
    print(sep)


def _run_root(choice: str) -> None:
    expr, f, x_sym = core.parse_math_expression(_ask("أدخل المعادلة بدلالة x (مثال: x^3 - x - 2): "))
    print(f"المعادلة المقبولة: f(x) = {expr}")
    if choice == "1":
        args, fn = (_ask("بداية الفترة (a): ", float), _ask("نهاية الفترة (b): ", float)), core.bisection
    elif choice == "2":
        diff_expr, df = core.make_derivative(expr, x_sym)
        print(f"المشتقة المحسوبة: f'(x) = {diff_expr}")
        args, fn = (df, _ask("النقطة الابتدائية (x0): ", float)), core.newton_raphson
    elif choice == "3":
        args, fn = (_ask("النقطة الأولى (x0): ", float), _ask("النقطة الثانية (x1): ", float)), core.secant
    else:
        args, fn = (_ask("التخمين الابتدائي (x0): ", float), _ask("نسبة الاضطراب delta (افتراضي 0.01): ", float, 0.01)), core.modified_secant
    tol, max_it = _ask("نسبة الخطأ (افتراضي 0.001): ", float, 0.001), _ask("أقصى تكرار (افتراضي 50): ", int, 50)
    _, steps, conv, msg = fn(f, *args, tol, max_it)
    headers, keys = ROOT_CLI[choice]
    if steps:
        _print_table(headers, [[s["iter"], *(f"{s[k]:.5f}" for k in keys), f"{s['error']:.6e}"] for s in steps])
    print(f"{'[+]' if conv else '[!]'} {msg}")


def _run_linear(solver) -> None:
    n = _ask("أدخل عدد المعادلات/المتغيرات (n): ", int)
    A = [_ask_nums(f"معاملات المعادلة {i + 1}: ") for i in range(n)]
    b = _ask_nums(f"أدخل متجه الثوابت b ({n} أرقام): ")
    x0_raw = _ask("التخمين الابتدائي (Enter للأصفار): ", str, "")
    x0 = [float(v) for v in x0_raw.replace(",", " ").split()] if x0_raw else None
    tol, max_it = _ask("نسبة الخطأ (افتراضي 0.001): ", float, 0.001), _ask("أقصى تكرار (افتراضي 50): ", int, 50)
    sol, steps, conv, msg = solver(A, b, x0, tol, max_it)
    if steps:
        _print_table(["Iter", "المتجه x", "Max Error"], [[s["iter"], "[" + ", ".join(f"{v:.5f}" for v in s["x"]) + "]", f"{s['error']:.6e}"] for s in steps])
    print(f"{'[+]' if conv else '[!]'} {msg}")
    if conv:
        print("\n".join(f"x{i} = {v:.6f}" for i, v in enumerate(sol, 1)))


def _run_regression() -> None:
    x, y = _ask_nums("قيم X (مفصولة بمسافات أو فواصل): "), _ask_nums("قيم Y (مفصولة بمسافات أو فواصل): ")
    models = {"1": "linear", "2": "polynomial", "3": "exponential", "4": "power"}
    choice = _ask("النموذج [1: Linear, 2: Polynomial, 3: Exponential, 4: Power]: ")
    if choice not in models:
        raise ValueError("اختيار نموذج غير صحيح!")
    deg = _ask("درجة كثير الحدود (افتراضي 2): ", int, 2) if choice == "2" else 2
    res = core.fit_curve(x, y, models[choice], degree=deg)
    print(f"\nالنموذج: {res['method_name']}\nالمعادلة: {res['equation']}\nR^2 = {res['r2']:.6f} ({res['r2']*100:.2f}%) | Sy/x = {res['syx']:.6f} | SSE = {res['sr']:.6f}")
    _print_table(["i", "X", "Y", "Y_pred", "Residual"], [[i, f"{xi:.4f}", f"{yi:.4f}", f"{yp:.4f}", f"{e:.4f}"] for i, (xi, yi, yp, e) in enumerate(zip(x, y, res["y_pred"], res["residuals"]), 1)])


def menu() -> None:
    """Interactive CLI menu for Math 3 Numerical Methods."""
    while True:
        print("\n" + "=" * 50 + "\n     حاسبة طرق Math 3 (التحليل العددي الشامل)\n" + "=" * 50)
        print("1. Bisection   2. Newton-Raphson   3. Secant   4. Modified Secant")
        print("5. Jacobi      6. Gauss-Seidel     7. Regression & Curve Fitting")
        print("8. تشغيل واجهة سطح المكتب (Desktop GUI)       0. خروج\n" + "=" * 50)
        choice = input("اختر رقم الطريقة: ").strip()
        try:
            if choice in ROOT_CLI:
                _run_root(choice)
            elif choice in ("5", "6"):
                _run_linear(core.jacobi if choice == "5" else core.gauss_seidel)
            elif choice == "7":
                _run_regression()
            elif choice == "8":
                subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "app.py")], cwd=BASE_DIR)
            elif choice == "0":
                print("شكراً لاستخدامك حاسبة التحليل العددي. بالتوفيق!")
                break
            else:
                print("اختيار غير صحيح، يرجى إدخال رقم من 0 إلى 8.")
        except KeyboardInterrupt:
            print("\nتم إلغاء العملية.")
        except Exception as err:
            print(f"\nحدث خطأ: {err}")


if __name__ == "__main__":
    menu()
