# حاسبة طرق التحليل العددي (Math 3) - النسخة المبسطة (Lite)

[![Python Version](https://img.shields.io/badge/بايثون-3.10%2B-blue.svg)](https://www.python.org/)
[![المنصة](https://img.shields.io/badge/المنصة-آرش%20لينكس%20%7C%20لينكس%20%7C%20ويندوز%20%7C%20ماك-lightgrey.svg)](https://github.com/mhmedreded-ai/numerical-methods-toolkit-lite)
[![الواجهة الرسومية](https://img.shields.io/badge/الواجهة-CustomTkinter-darkblue.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![المحرك الرياضي](https://img.shields.io/badge/المحرك-SymPy%20%2B%20NumPy-orange.svg)](https://www.sympy.org/)
[![الرخصة](https://img.shields.io/badge/الرخصة-MIT-green.svg)](LICENSE)

حقيبة متكاملة ومبسطة لطرق **التحليل العددي (Math 3)** تعمل عبر **واجهة الطرفية (CLI)** و**واجهة سطح المكتب (CustomTkinter Desktop GUI)** مع فصل تام للخوارزميات الرياضية عن الواجهات، وجداول تكرار تفاعلية، ورسم بياني عبر Matplotlib، ودعم كامل للغة العربية (BiDi).

---

## الإصدارات المتاحة (Releases)

- **`v1.0.0-modular`** (فرع `main` — هذا التنظيم): هيكلة معيارية في مجلد `core/` (`math_utils.py`, `roots.py`, `linear.py`, `regression.py`) + `app.py` + `main.py`.
- **`v1.0.0-3files`** (فرع `3files`): هيكلة مصغّرة من 3 ملفات بايثون فقط في المجلد الرئيسي (`methods.py` + `app.py` + `main.py`).

يمكنك تحميل أي من الإصدارين من صفحة [GitHub Releases](https://github.com/mhmedreded-ai/numerical-methods-toolkit-lite/releases).

---

## هيكل المشروع ودور كل ملف

```text
numerical-methods-toolkit-lite/
├── core/                   # المحركات الرياضية الصافية (بدون أي طباعة أو كود واجهات)
│   ├── __init__.py         # تصدير الدوال الأساسية
│   ├── math_utils.py       # تحليل المعادلات بـ SymPy، المشتقة الآمنة، والسيادة القطرية
│   ├── roots.py            # طرق الجذور: Bisection, Newton-Raphson, Secant, Modified Secant
│   ├── linear.py           # محرك مشترك لطريقتي Jacobi و Gauss-Seidel
│   └── regression.py       # مواءمة المنحنيات والانحدار باستخدام NumPy
├── app.py                  # واجهة سطح المكتب (CustomTkinter + Matplotlib + دعم العربي BiDi)
├── main.py                 # القائمة التفاعلية للطرفية (CLI) وتشغيل واجهة سطح المكتب
├── run.sh                  # سكربت تشغيل لينكس وماك (الواجهة افتراضياً، و --cli للطرفية)
├── run.bat                 # سكربت تشغيل ويندوز (الواجهة افتراضياً، و --cli للطرفية)
├── requirements.txt        # مكتبات المشروع الأساسية فقط
├── README.md               # التوثيق باللغة الإنجليزية
├── README_AR.md            # التوثيق باللغة العربية (هذا الملف)
└── LICENSE                 # رخصة MIT مفتوحة المصدر
```

---

## الطرق العددية المدعومة

1. **طرق إيجاد الجذور (`core/roots.py`)**:
   - **تنصيف الفترات (Bisection)**
   - **نيوتن-رافسون (Newton-Raphson)** مع حساب المشتقة رمزياً تلقائياً
   - **طريقة القاطع (Secant)**
   - **طريقة القاطع المعدلة (Modified Secant)** بمعامل الاضطراب $\delta$
2. **طرق حل الأنظمة الخطية (`core/linear.py`)**:
   - **جاكوبي (Jacobi)** و**جاوس-سيدل (Gauss-Seidel)**
   - فحص وإعادة ترتيب صفوف المصفوفة تلقائياً لتحقيق **السيادة القطرية (Diagonal Dominance)**
3. **الانحدار ومواءمة المنحنيات (`core/regression.py`)**:
   - **الخطي (Linear)**، **متعدد الحدود (Polynomial)**، **الأسي (Exponential)**، و**دالة القوى (Power)**
   - حساب $R^2$، $r$، $S_{y/x}$، $\text{SSE}$ ($S_r$)، وجدول البواقي مع الرسم البياني التفاعلي عبر Matplotlib

---

## التثبيت والتشغيل

### 1. متطلبات النظام
- **Python 3.10+**
- **Arch Linux**: `sudo pacman -S tk python`
- **Ubuntu / Debian**: `sudo apt install python3 python3-pip python3-tk`
- **Fedora**: `sudo dnf install python3 python3-tkinter`

### 2. إعداد البيئة الافتراضية وتثبيت المكتبات
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. تشغيل البرنامج
```bash
# تشغيل واجهة سطح المكتب (لينكس / ماك)
./run.sh
# أو مباشرة:
.venv/bin/python app.py

# تشغيل قائمة الطرفية (CLI)
./run.sh --cli
# أو مباشرة:
.venv/bin/python main.py
```

على **Windows**:
```bat
run.bat
run.bat --cli
```

### ضبط مقياس الشاشة على Wayland / Hyprland
```bash
CTK_SCALING=1.0 .venv/bin/python app.py
```

---

## الرخصة

منشور تحت رخصة **[MIT License](LICENSE)** مفتوحة المصدر.
