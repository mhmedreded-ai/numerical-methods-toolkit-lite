"""Math 3 - Numerical Analysis Desktop Application (CustomTkinter + Matplotlib)."""

import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

if sys.platform.startswith("linux"):
    venv_lib = os.path.join(BASE_DIR, ".venv", "lib")
    if os.path.exists(os.path.join(venv_lib, "libtk8.6.so")):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ.setdefault("TCL_LIBRARY", os.path.join(venv_lib, "tcl8.6"))
        os.environ.setdefault("TK_LIBRARY", os.path.join(venv_lib, "tk8.6"))
        if venv_lib not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{venv_lib}:{ld}" if ld else venv_lib
            if sys.argv and sys.argv[0] != "-c":
                os.execv(sys.executable, [sys.executable] + sys.argv)

import arabic_reshaper
import customtkinter as ctk
import matplotlib
import tkinter.font as tkfont
from bidi.algorithm import get_display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk
import methods

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Tahoma", "Arial", "Noto Sans Arabic", "Geeza Pro", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LTR_TRAILING_SYMBOLS_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9\s\u00b2\+\-\*\/\^\=\|\(\)\.\,\%]*[\-\=\+\*\/\|]+)")
DEFAULT_METRICS = "مقاييس الجودة: R² = -- | Sy/x = -- | SSE (Sr) = --"


def ar(text: str) -> str:
    """Reshape and reorder Arabic text for bidirectional rendering in Tkinter/Matplotlib."""
    if not text or not ARABIC_CHAR_RE.search(str(text)):
        return str(text) if text else ""
    try:
        use_lro = sys.platform in ("win32", "darwin")
        lines = []
        for line in str(text).split("\n"):
            if not ARABIC_CHAR_RE.search(line):
                lines.append(line)
                continue
            bidi = get_display(arabic_reshaper.reshape(LTR_TRAILING_SYMBOLS_RE.sub("\\1\u200e", line)), base_dir="R")
            lines.append(f"\u202d{bidi}\u202c" if use_lro else bidi)
        return "\n".join(lines)
    except Exception:
        return str(text)


ROOT_CFG = {
    "Bisection": ("بداية الفترة (a):", "نهاية الفترة (b):", "2.0", ("a", "b", "c (المنتصف)", "f(c)"), ("a", "b", "c", "fc")),
    "Newton-Raphson": ("النقطة الابتدائية (x0):", None, None, ("x_n", "f(x_n)", "f'(x_n)", "x_(n+1)"), ("xn", "fx", "dfx", "x_next")),
    "Secant": ("النقطة الأولى (x0):", "النقطة الثانية (x1):", "2.0", ("x_(n-1)", "x_n", "x_(n+1)", "f(x_(n+1))"), ("x_prev", "x_curr", "x_next", "fx_next")),
    "Modified Secant": ("النقطة الابتدائية (x0):", "نسبة التغيير (Delta / \u03b4):", "0.01", ("x_i", "f(x_i)", "f(x_i + \u03b4*x_i)", "x_(i+1)"), ("xi", "fxi", "f_pert", "x_next")),
}
REG_SAMPLES = {
    "Linear": ("1, 2, 3, 4, 5, 6, 7", "0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5"),
    "Polynomial": ("0, 1, 2, 3, 4, 5", "2.1, 7.7, 13.6, 27.2, 40.9, 61.1"),
    "Exponential": ("1, 2, 3, 4, 5", "0.5, 1.7, 5.0, 14.8, 44.0"),
    "Power": ("1, 2, 3, 4, 5", "0.5, 2.0, 4.5, 8.0, 12.5"),
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
if "CTK_SCALING" in os.environ:
    try:
        sc = float(os.environ["CTK_SCALING"])
        ctk.set_widget_scaling(sc)
        ctk.set_window_scaling(sc)
    except ValueError:
        pass


def _root_cols(mid_headers):
    return [("c1", "الدورة", 70), *zip(("c2", "c3", "c4", "c5"), mid_headers, (125, 130, 145, 135)), ("c6", "الخطأ (Error)", 120)]


class NumericalApp(ctk.CTk):
    """Desktop GUI for Root Finding, Linear Systems, and Curve Fitting."""

    def __init__(self):
        super().__init__()
        self.title("Math 3 - Numerical Analysis Suite (التحليل العددي)")
        self.geometry("1020x820")
        self.minsize(900, 700)
        self.font_family, self.font_mono = self._pick_fonts()
        self._style_treeview()
        self._lbl(self, "حاسبة طرق التحليل العددي المتكاملة (Math 3)", size=22, bold=True, pack={"pady": (12, 4)})
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=15, pady=8, fill="both", expand=True)
        tabs = ("جذور المعادلات (Root Finding)", "الأنظمة الخطية (Linear Systems)", "الانحدار ومواءمة المنحنيات (Regression)")
        self.tab_roots, self.tab_linear, self.tab_reg = (self.tabview.add(ar(t)) for t in tabs)
        self._setup_roots_tab()
        self._setup_linear_tab()
        self._setup_regression_tab()

    def _pick_fonts(self):
        try:
            avail = set(tkfont.families(self))
        except Exception:
            avail = set()
        ui = next((f for f in ("Segoe UI", "Noto Sans Arabic", "Geeza Pro", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial") if f in avail), "Segoe UI")
        mono = next((f for f in ("Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "Courier New", "monospace") if f in avail), "Consolas")
        return ui, mono

    def _style_treeview(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#23272e", foreground="#dcdfe4", rowheight=26, fieldbackground="#23272e", font=(self.font_family, 10))
        style.configure("Treeview.Heading", background="#1f538d", foreground="#ffffff", font=(self.font_family, 10, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", "#2a5d9f")], foreground=[("selected", "#ffffff")])

    def _lbl(self, parent, text="", size=12, bold=False, italic=False, pack=None, **kw):
        lbl = ctk.CTkLabel(parent, text=ar(text), font=(self.font_family, size, "bold" if bold else "normal", "italic" if italic else "roman"), **kw)
        if pack is not None:
            lbl.pack(**pack)
        return lbl

    def _seg(self, parent, values, var, cmd=None, pady=8):
        ctk.CTkSegmentedButton(parent, values=values, variable=var, command=cmd, font=(self.font_family, 13, "bold"), height=32).pack(pady=pady)

    def _field(self, parent, label, default, row, col, width=120, bold=False):
        lbl = self._lbl(parent, label, bold=bold)
        lbl.grid(row=row, column=col + 1, padx=8, pady=4, sticky="e")
        entry = ctk.CTkEntry(parent, width=width)
        if default:
            entry.insert(0, default)
        entry.grid(row=row, column=col, padx=8, pady=4, sticky="w")
        return lbl, entry

    def _table(self, parent, cols, pack=True):
        wrap = ctk.CTkFrame(parent)
        if pack:
            wrap.pack(fill="both", expand=True, padx=10, pady=4)
        scroll = ttk.Scrollbar(wrap, orient="vertical")
        tree = ttk.Treeview(wrap, columns=[c[0] for c in cols], show="headings", yscrollcommand=scroll.set)
        scroll.config(command=tree.yview)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        for cid, title, w in cols:
            tree.heading(cid, text=ar(title))
            tree.column(cid, width=w, anchor="center")
        return tree if pack else (wrap, tree)

    def _btn(self, parent, text, cmd, fg="#1f538d", hover="#14375e", fsize=13, pack=None, **kw):
        btn = ctk.CTkButton(parent, text=ar(text), command=cmd, fg_color=fg, hover_color=hover, font=(self.font_family, fsize, "bold"), **kw)
        if pack is not None:
            btn.pack(**pack)
        return btn

    def _set_entry(self, entry, value) -> None:
        entry.delete(0, "end")
        entry.insert(0, str(value))

    def _set_status(self, label, msg: str, conv: bool) -> None:
        color = "#2ecc71" if conv else ("#e67e22" if "الحد الأقصى" in msg else "#e74c3c")
        label.configure(text=ar(msg), text_color=color)

    # ---------------- 1. Root Finding Tab ----------------
    def _setup_roots_tab(self):
        self.root_method_var = ctk.StringVar(value="Bisection")
        self._seg(self.tab_roots, list(ROOT_CFG), self.root_method_var, self.update_root_inputs)
        frame = ctk.CTkFrame(self.tab_roots)
        frame.pack(pady=4, padx=12, fill="x")
        _, self.entry_eq = self._field(frame, "المعادلة f(x):", "x^3 - x - 2", 0, 2, width=280, bold=True)
        self.lbl_diff_info = self._lbl(frame, size=11, italic=True, text_color="#3498db")
        self.lbl_diff_info.grid(row=0, column=0, columnspan=2, padx=10, pady=6, sticky="w")
        self.lbl_p1, self.entry_p1 = self._field(frame, "بداية الفترة (a):", "1.0", 1, 2)
        self.lbl_p2, self.entry_p2 = self._field(frame, "نهاية الفترة (b):", "2.0", 2, 2)
        _, self.entry_tol = self._field(frame, "نسبة الخطأ (Tol):", "0.001", 1, 0, width=100)
        _, self.entry_max_iter = self._field(frame, "أقصى تكرار:", "50", 2, 0, width=100)
        self._btn(self.tab_roots, "احسب الحل الآن", self.calculate_roots, height=34, pack={"pady": 6})
        self.res_roots = self._lbl(self.tab_roots, size=13, bold=True, wraplength=850, pack={"pady": 2})
        self.tree_roots = self._table(self.tab_roots, _root_cols(ROOT_CFG["Bisection"][3]))

    def update_root_inputs(self, choice):
        p1_lbl, p2_lbl, p2_def, mid_headers, _ = ROOT_CFG[choice]
        for cid, title, w in _root_cols(mid_headers):
            self.tree_roots.heading(cid, text=ar(title))
            self.tree_roots.column(cid, width=w, anchor="center")
        self.lbl_diff_info.configure(text="")
        self.lbl_p1.configure(text=ar(p1_lbl))
        if p2_lbl:
            self.lbl_p2.configure(text=ar(p2_lbl))
            self._set_entry(self.entry_p2, p2_def)
            self.lbl_p2.grid()
            self.entry_p2.grid()
        else:
            self.lbl_p2.grid_remove()
            self.entry_p2.grid_remove()

    def calculate_roots(self):
        self.tree_roots.delete(*self.tree_roots.get_children())
        self.res_roots.configure(text="")
        try:
            expr, f, x_sym = methods.parse_math_expression(self.entry_eq.get())
            tol, max_iter, method = float(self.entry_tol.get()), int(self.entry_max_iter.get()), self.root_method_var.get()
            p1 = float(self.entry_p1.get())
            if method == "Bisection":
                p2 = float(self.entry_p2.get())
                if p1 > p2:
                    p1, p2 = p2, p1
                    self._set_entry(self.entry_p1, p1)
                    self._set_entry(self.entry_p2, p2)
                _, steps, conv, msg = methods.bisection(f, p1, p2, tol, max_iter)
            elif method == "Newton-Raphson":
                diff_expr, df = methods.make_derivative(expr, x_sym)
                self.lbl_diff_info.configure(text=ar(f"المشتقة المحسوبة: f'(x) = {diff_expr}"))
                _, steps, conv, msg = methods.newton_raphson(f, df, p1, tol, max_iter)
            elif method == "Secant":
                _, steps, conv, msg = methods.secant(f, p1, float(self.entry_p2.get()), tol, max_iter)
            else:
                _, steps, conv, msg = methods.modified_secant(f, p1, float(self.entry_p2.get().strip() or 0.01), tol, max_iter)
            keys = ROOT_CFG[method][4]
            for s in steps:
                self.tree_roots.insert("", "end", values=(s["iter"], *(f"{s[k]:.5f}" for k in keys), f"{s['error']:.6f}"))
            self._set_status(self.res_roots, msg, conv)
        except Exception as err:
            self._set_status(self.res_roots, f"خطأ: {err}", False)

    # ---------------- 2. Linear Systems Tab ----------------
    def _setup_linear_tab(self):
        self.linear_method_var = ctk.StringVar(value="Gauss-Seidel")
        self._seg(self.tab_linear, ["Gauss-Seidel", "Jacobi"], self.linear_method_var)
        frame = ctk.CTkFrame(self.tab_linear)
        frame.pack(pady=4, padx=12, fill="x")
        self._lbl(frame, "مصفوفة المعاملات A (كل سطر يمثل معادلة، الأرقام مفصولة بمسافات):", bold=True, pack={"anchor": "e", "padx": 10, "pady": (4, 0)})
        self.txt_matrix = ctk.CTkTextbox(frame, height=75, font=(self.font_mono, 12))
        self.txt_matrix.insert("1.0", "4 1 -1\n2 7 1\n1 -3 12")
        self.txt_matrix.pack(fill="x", padx=10, pady=3)
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=4)
        _, self.entry_b = self._field(grid, "الحدود المطلقة b:", "3 19 31", 0, 4, width=140)
        _, self.entry_x0 = self._field(grid, "التخمين X0 (اختياري):", "", 0, 2, width=120)
        _, self.entry_lin_tol = self._field(grid, "نسبة الخطأ:", "0.001", 0, 0, width=75)
        self.lbl_dom_status = self._lbl(self.tab_linear, size=11, italic=True, text_color="#3498db", pack={"pady": 2})
        self._btn(self.tab_linear, "حل النظام الخطي الآن", self.calculate_linear, height=34, pack={"pady": 4})
        self.res_linear = self._lbl(self.tab_linear, size=13, bold=True, wraplength=850, pack={"pady": 2})
        self.tree_linear = self._table(self.tab_linear, [("Iter", "الدورة", 80), ("Values", "قيم المجاهيل [x1, x2, x3, ...]", 550), ("Error", "أقصى خطأ (Max Error)", 160)])

    def calculate_linear(self):
        self.tree_linear.delete(*self.tree_linear.get_children())
        self.res_linear.configure(text="")
        self.lbl_dom_status.configure(text="")
        try:
            A = [[float(v) for v in line.split()] for line in self.txt_matrix.get("1.0", "end").splitlines() if line.strip()]
            b = [float(v) for v in self.entry_b.get().split()]
            tol, n = float(self.entry_lin_tol.get()), len(A)
            if n < 2 or n != len(b) or any(len(r) != n for r in A):
                raise ValueError("يجب إدخال مصفوفة مربعة (n>=2) متوافقة مع متجه الثوابت b.")
            raw_x0 = self.entry_x0.get().strip()
            x0 = [float(v) for v in raw_x0.split()] if raw_x0 else None
            self.lbl_dom_status.configure(text=ar(f"[معاينة السيادة القطرية]: {methods.make_diagonally_dominant(A, b)[3]}"))
            solver = methods.jacobi if self.linear_method_var.get() == "Jacobi" else methods.gauss_seidel
            sol, steps, conv, msg = solver(A, b, x0=x0, tol=tol, max_iter=100)
            for s in steps:
                self.tree_linear.insert("", "end", values=(s["iter"], "[" + ", ".join(f"{v:.5f}" for v in s["x"]) + "]", f"{s['error']:.6f}"))
            sol_str = "[" + ", ".join(f"{v:.5f}" for v in sol) + "]"
            self._set_status(self.res_linear, f"تم الوصول للحل بنجاح: {sol_str} بعد {len(steps)} دورة" if conv else msg, conv)
        except Exception as err:
            self._set_status(self.res_linear, f"خطأ في الإدخال: {err}", False)

    # ---------------- 3. Regression & Curve Fitting Tab ----------------
    def _setup_regression_tab(self):
        self.reg_method_var = ctk.StringVar(value="Linear")
        self._seg(self.tab_reg, list(REG_SAMPLES), self.reg_method_var, self.update_regression_inputs, pady=(6, 4))
        frame = ctk.CTkFrame(self.tab_reg)
        frame.pack(fill="x", padx=12, pady=4)
        _, self.entry_reg_x = self._field(frame, "قيم X (مفصولة بمسافات أو فواصل):", REG_SAMPLES["Linear"][0], 0, 1, width=380, bold=True)
        _, self.entry_reg_y = self._field(frame, "قيم Y (مفصولة بمسافات أو فواصل):", REG_SAMPLES["Linear"][1], 1, 1, width=380, bold=True)
        opts = ctk.CTkFrame(frame, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, padx=10, pady=(2, 6), sticky="ew")
        self.deg_container = ctk.CTkFrame(opts, fg_color="transparent")
        self._lbl(self.deg_container, "درجة كثير الحدود (Degree m):", pack={"side": "right", "padx": (5, 0)})
        self.entry_reg_deg = ctk.CTkEntry(self.deg_container, width=65)
        self.entry_reg_deg.insert(0, "2")
        self.entry_reg_deg.pack(side="right", padx=(0, 5))
        self._btn(opts, "بيانات تجريبية (Sample)", self.load_sample_regression_data, fg="#2b5278", hover="#1e3b57", width=130, height=28, fsize=11, pack={"side": "left", "padx": 6})
        self._btn(opts, "مسح الحقول (Clear)", self.clear_regression_inputs, fg="#5c6370", hover="#4b5263", width=110, height=28, fsize=11, pack={"side": "left", "padx": 6})
        self._btn(self.tab_reg, "احسب وطابق المنحنى الآن (Fit Curve)", self.calculate_regression, height=34, pack={"pady": 4})

        box = ctk.CTkFrame(self.tab_reg, fg_color="#1e2227")
        box.pack(fill="x", padx=12, pady=2)
        self.lbl_reg_eq = self._lbl(box, "معادلة المنحنى: اضغط على زر الحساب لعرض المعادلة والمخطط", size=13, bold=True, text_color="#61afef", pack={"pady": (4, 2)})
        self.lbl_reg_metrics = self._lbl(box, DEFAULT_METRICS, size=11, text_color="#abb2bf", pack={"pady": (0, 4)})
        self.lbl_reg_error = self._lbl(self.tab_reg, bold=True, text_color="#e74c3c", pack={"pady": (0, 2)})

        split = ctk.CTkFrame(self.tab_reg)
        split.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        split.columnconfigure(0, weight=4)
        split.columnconfigure(1, weight=5)
        split.rowconfigure(0, weight=1)
        tbl_wrap, self.tree_reg = self._table(split, [("pt", "ن", 35), ("x", "X", 65), ("y_act", "Y الفعلي", 75), ("y_pred", "Y النموذج", 75), ("res", "الباقي (e)", 75)], pack=False)
        tbl_wrap.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        plot_box = ctk.CTkFrame(split)
        plot_box.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.fig_reg = Figure(figsize=(4.6, 3.2), dpi=95)
        self.fig_reg.patch.set_facecolor("#23272e")
        self.ax_reg = self.fig_reg.add_subplot(111)
        self.canvas_reg = FigureCanvasTkAgg(self.fig_reg, master=plot_box)
        self.canvas_reg.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)
        self._style_plot(clear=True)
        self.canvas_reg.draw()

    def _style_plot(self, clear=False):
        if clear:
            self.ax_reg.clear()
        self.ax_reg.set_facecolor("#1e2227")
        self.ax_reg.tick_params(colors="#abb2bf", labelsize=8)
        for sp in self.ax_reg.spines.values():
            sp.set_color("#5c6370")
        self.ax_reg.grid(True, color="#5c6370", linestyle="--", alpha=0.35)
        self.ax_reg.set_xlabel("X", color="#abb2bf", fontsize=9)
        self.ax_reg.set_ylabel("Y", color="#abb2bf", fontsize=9)

    def update_regression_inputs(self, choice=None):
        if (choice or self.reg_method_var.get()) == "Polynomial":
            self.deg_container.pack(side="right", padx=10)
        else:
            self.deg_container.pack_forget()

    def load_sample_regression_data(self):
        sx, sy = REG_SAMPLES[self.reg_method_var.get()]
        self._set_entry(self.entry_reg_x, sx)
        self._set_entry(self.entry_reg_y, sy)

    def clear_regression_inputs(self):
        self._set_entry(self.entry_reg_x, "")
        self._set_entry(self.entry_reg_y, "")
        self.lbl_reg_eq.configure(text=ar("معادلة المنحنى: في انتظار الحساب..."))
        self.lbl_reg_metrics.configure(text=ar(DEFAULT_METRICS))
        self.lbl_reg_error.configure(text="")
        self.tree_reg.delete(*self.tree_reg.get_children())
        self._style_plot(clear=True)
        self.canvas_reg.draw()

    def calculate_regression(self):
        self.lbl_reg_error.configure(text="")
        method = self.reg_method_var.get()
        try:
            x_vals = [float(v) for v in self.entry_reg_x.get().replace(",", " ").split()]
            y_vals = [float(v) for v in self.entry_reg_y.get().replace(",", " ").split()]
            if not x_vals or not y_vals:
                raise ValueError("يرجى إدخال قيم X وقيم Y!")
            deg = int(self.entry_reg_deg.get()) if method == "Polynomial" else 2
            res = methods.fit_curve(x_vals, y_vals, method=method, degree=deg)
            self.lbl_reg_eq.configure(text=ar(f"معادلة المنحنى:  {res['equation']}"))
            self.lbl_reg_metrics.configure(text=ar(f"R² = {res['r2']:.5f} ({res['r2']*100:.2f}%)   |   Sy/x = {res['syx']:.5f}   |   SSE (Sr) = {res['sr']:.5f}   |   St = {res['st']:.5f}"))
            self.tree_reg.delete(*self.tree_reg.get_children())
            for i, (xi, yi, yp, e) in enumerate(zip(x_vals, y_vals, res["y_pred"], res["residuals"]), 1):
                self.tree_reg.insert("", "end", values=(i, f"{xi:.4g}", f"{yi:.4g}", f"{yp:.4g}", f"{e:+.4g}"))
            self._style_plot(clear=True)
            self.ax_reg.scatter(x_vals, y_vals, color="#e06c75", s=50, zorder=5, label=ar("نقاط البيانات (Data)"), edgecolors="#ffffff", linewidths=0.8)
            self.ax_reg.plot(res["x_curve"], res["y_curve"], color="#61afef", linewidth=2.2, zorder=4, label=ar(f"منحنى المواءمة ({method})"))
            self.ax_reg.set_title(f"Curve Fit: {res['method_name']} (R² = {res['r2']:.4f})", color="#dcdfe4", fontsize=10, fontweight="bold", pad=8)
            self.ax_reg.legend(facecolor="#282c34", edgecolor="#5c6370", labelcolor="#dcdfe4", fontsize=8, loc="best")
            self.fig_reg.tight_layout()
            self.canvas_reg.draw()
        except Exception as err:
            self.lbl_reg_error.configure(text=ar(f"خطأ أثناء الحساب: {err}"))


if __name__ == "__main__":
    app = NumericalApp()
    app.mainloop()
