# Math 3 - Numerical Analysis Suite (Lite Edition)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Arch%20Linux%20%7C%20Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](https://github.com/mhmedreded-ai/numerical-methods-toolkit-lite)
[![GUI Framework](https://img.shields.io/badge/GUI-CustomTkinter-darkblue.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![Math Engine](https://img.shields.io/badge/Engine-SymPy%20%2B%20NumPy-orange.svg)](https://www.sympy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A clean, streamlined, and easy-to-read **CLI + Desktop GUI** toolkit for engineering mathematics (**Math 3 / Numerical Methods**), featuring a pure mathematical core, interactive iteration tables, real-time Matplotlib curve plotting, and bidirectional Arabic text rendering.

---

## Available Releases

- **`v1.0.0-modular`** (`main` branch — this structure): Organized modular `core/` package (`core/math_utils.py`, `core/roots.py`, `core/linear.py`, `core/regression.py`) + `app.py` + `main.py`.
- **`v1.0.0-3files`** (`3files` branch): Minimal 3-file flat root layout (`methods.py` + `app.py` + `main.py`).

Download either archive from the [GitHub Releases](https://github.com/mhmedreded-ai/numerical-methods-toolkit-lite/releases) page.

---

## Architecture & Directory Structure

```text
numerical-methods-toolkit-lite/
├── core/                   # Pure mathematical engines (zero UI/print dependencies)
│   ├── __init__.py         # Public math API exports
│   ├── math_utils.py       # SymPy expression parser, symbolic derivative, diagonal dominance
│   ├── roots.py            # Bisection, Newton-Raphson, Secant, Modified Secant
│   ├── linear.py           # Unified iterative engine for Jacobi & Gauss-Seidel
│   └── regression.py       # NumPy least-squares fitting (Linear, Polynomial, Exponential, Power)
├── app.py                  # CustomTkinter Desktop GUI + Matplotlib plot + Arabic BiDi shaping
├── main.py                 # Interactive Terminal CLI + Desktop GUI launcher
├── run.sh                  # Linux / macOS launcher (Desktop GUI default, --cli for terminal)
├── run.bat                 # Windows launcher (Desktop GUI default, --cli for terminal)
├── requirements.txt        # Core math & desktop dependencies
├── README.md               # English documentation
├── README_AR.md            # Arabic documentation (النسخة العربية)
└── LICENSE                 # MIT License
```

### Included Numerical Methods

1. **Root Finding (`core/roots.py`)**
   - **Bisection Method**: Bracketing method using Bolzano's theorem ($c = (a+b)/2$).
   - **Newton-Raphson Method**: Quadratic tangent method ($x_{n+1} = x_n - f(x_n)/f'(x_n)$) with automatic symbolic differentiation via SymPy.
   - **Secant Method**: Two-point finite-difference secant solver.
   - **Modified Secant Method**: Single-point perturbation solver using fractional step $\delta$.
2. **Linear Systems (`core/linear.py`)**
   - **Jacobi Iteration**: Simultaneous coordinate updates.
   - **Gauss-Seidel Iteration**: In-place successive displacements.
   - Includes automatic **diagonal dominance** detection and row permutation (`make_diagonally_dominant`).
3. **Curve Fitting & Regression (`core/regression.py`)**
   - **Linear** ($y = a_0 + a_1 x$), **Polynomial** (degree $m$), **Exponential** ($y = a e^{bx}$), and **Power** ($y = a x^b$).
   - Computes $R^2$, $r$, $S_{y/x}$, $\text{SSE}$ ($S_r$), $S_t$, residuals, and 200-point smooth curves for plotting.

---

## Installation & Running

### 1. System Prerequisites
- **Python 3.10+**
- **Arch Linux**: `sudo pacman -S tk python`
- **Ubuntu / Debian**: `sudo apt install python3 python3-pip python3-tk`
- **Fedora**: `sudo dnf install python3 python3-tkinter`

### 2. Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Desktop GUI or CLI
```bash
# Desktop GUI (Linux / macOS)
./run.sh
# or directly:
.venv/bin/python app.py

# Interactive Terminal CLI
./run.sh --cli
# or directly:
.venv/bin/python main.py
```

On Windows:
```bat
run.bat
run.bat --cli
```

### Wayland / HiDPI Scaling
```bash
CTK_SCALING=1.0 .venv/bin/python app.py
```

---

## License

Licensed under the [MIT License](LICENSE).
