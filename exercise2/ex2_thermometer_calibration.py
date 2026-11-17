"""
Exercise 2 - Calibration of a novel thermometer against a mercury thermometer
  3. Plot the raw data set
  4. Fit a straight line: RMSE, R^2, adjusted R^2, residual plot, comment on adequacy
  5. Fit a 2nd-order polynomial and repeat
Extras: fit equations with parameter uncertainties, residual randomness checks (runs of signs,
Durbin-Watson, Shapiro-Wilk), F-test for the quadratic term, AIC, hysteresis between the
cooling and heating sweeps, and the inverse calibration T = f(reading) that a user would need.
Run:  python ex2_thermometer_calibration.py   (needs Exercise2_thermometer_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- settings ----------------
DATA_FILE = "Exercise2_thermometer_data.xlsx"
OUT_DIR = "ex2_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})
SWEEPS = {"cooling": ("tab:blue", "o", "Cooling sweep (80 → 4 °C)"),
          "heating": ("tab:orange", "^", "Heating sweep (4 → 80 °C)")}

# ---------------- load data ----------------
df = pd.read_excel(DATA_FILE, sheet_name="Data")
df = df[df["Sweep"].isin(["cooling", "heating"])].copy()       # drop the note row
T = df["Hg_temperature_C"].to_numpy(float)                     # x: reference temperature (°C)
Y = df["New_thermometer_reading"].to_numpy(float)              # y: new thermometer reading (a.u.)
sweep = df["Sweep"].to_numpy()
n = len(T)
print(f"Data points: {n}  (cooling {np.sum(sweep == 'cooling')}, heating {np.sum(sweep == 'heating')})")


def poly_fit(x, y, deg):
    """Least-squares polynomial fit with fit statistics and parameter uncertainties."""
    X = np.vander(x, deg + 1)                       # columns: x^deg ... x^0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    res = y - yhat
    n_, p = len(y), deg                              # p = number of predictors (excluding intercept)
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n_ - 1) / (n_ - p - 1)
    rmse = np.sqrt(ss_res / n_)
    S = np.sqrt(ss_res / (n_ - p - 1))               # standard error of the regression
    cov = S ** 2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    # residual randomness checks
    signs = np.sign(res[np.argsort(x)])
    runs = 1 + np.sum(signs[1:] != signs[:-1])       # number of runs of same-sign residuals
    n_pos, n_neg = np.sum(signs > 0), np.sum(signs < 0)
    mu = 2 * n_pos * n_neg / (n_pos + n_neg) + 1
    var = (mu - 1) * (mu - 2) / (n_pos + n_neg - 1)
    z_runs = (runs - mu) / np.sqrt(var)
    r_sorted = res[np.argsort(x)]
    dw = np.sum(np.diff(r_sorted) ** 2) / ss_res     # Durbin-Watson (≈2 means no pattern)
    aic = n_ * np.log(ss_res / n_) + 2 * (deg + 1)
    return {"deg": deg, "coef": coef, "se": se, "yhat": yhat, "res": res, "r2": r2,
            "adj_r2": adj_r2, "rmse": rmse, "S": S, "ss_res": ss_res, "runs": runs,
            "runs_expected": mu, "runs_p": 2 * stats.norm.sf(abs(z_runs)), "dw": dw,
            "shapiro_p": stats.shapiro(res).pvalue, "aic": aic}


def equation(coef):
    """Readable equation string, highest power first."""
    deg = len(coef) - 1
    terms = []
    for i, c in enumerate(coef):
        pw = deg - i
        t = f"{abs(c):.4g}" + ("" if pw == 0 else "T" if pw == 1 else f"T$^{pw}$")
        terms.append(("−" if c < 0 else "+") + " " + t)
    s = " ".join(terms)
    return "y = " + (s[2:] if s.startswith("+") else "−" + s[2:])


fits = {1: poly_fit(T, Y, 1), 2: poly_fit(T, Y, 2)}
names = {1: "Linear (1st order)", 2: "Quadratic (2nd order)"}

# ---------------- print results ----------------
for d, f in fits.items():
    print(f"\n========== {names[d]} ==========")
    print("Equation:", equation(f["coef"]).replace("$", "").replace("^", "^"))
    for i, (c, s) in enumerate(zip(f["coef"], f["se"])):
        print(f"   coefficient of T^{d - i}: {c:+.6g} ± {s:.3g}")
    print(f"   RMSE            = {f['rmse']:.3f}")
    print(f"   S (std error)   = {f['S']:.3f}")
    print(f"   R^2             = {f['r2']:.5f}")
    print(f"   adjusted R^2    = {f['adj_r2']:.5f}")
    print(f"   residual runs   = {f['runs']} (expected {f['runs_expected']:.0f} if random, p = {f['runs_p']:.2g})")
    print(f"   Durbin-Watson   = {f['dw']:.2f} (≈2 = random, <<2 = systematic pattern)")
    print(f"   Shapiro-Wilk p  = {f['shapiro_p']:.3g}")
    print(f"   AIC             = {f['aic']:.1f} (lower is better)")

# F-test: does the quadratic term significantly improve the fit?
F = (fits[1]["ss_res"] - fits[2]["ss_res"]) / (fits[2]["ss_res"] / (n - 3))
p_F = stats.f.sf(F, 1, n - 3)
print(f"\nF-test linear vs quadratic: F = {F:.1f}, p = {p_F:.3g}")

# sensitivity (slope of the calibration curve) for the quadratic
a, b, c = fits[2]["coef"]
for t0 in (4, 20, 40, 60, 80):
    print(f"   sensitivity dy/dT at {t0:2d} °C = {2 * a * t0 + b:+.2f} units/°C")

# hysteresis: fit each sweep separately with a quadratic
tt = np.linspace(T.min(), T.max(), 300)
sweep_fit = {s: poly_fit(T[sweep == s], Y[sweep == s], 2) for s in SWEEPS}
gap = np.polyval(sweep_fit["cooling"]["coef"], tt) - np.polyval(sweep_fit["heating"]["coef"], tt)
print(f"\nHysteresis (cooling − heating reading): mean {gap.mean():.1f}, range {gap.min():.1f} to {gap.max():.1f} units")
for s, f in sweep_fit.items():
    print(f"   {s}: {equation(f['coef']).replace('$', '')}   RMSE {f['rmse']:.2f}")

# inverse calibration: temperature from a reading (what the user of the thermometer needs)
inv = poly_fit(Y, T, 2)
print(f"\nInverse calibration: {equation(inv['coef']).replace('$', '').replace('T', 'R').replace('y =', 'T =')}  "
      f"-> RMSE {inv['rmse']:.2f} °C, R^2 {inv['r2']:.4f}")

# ======================= FIGURE 1: raw data =======================
fig, ax = plt.subplots(figsize=(8, 5.5))
for s, (col, mk, lab) in SWEEPS.items():
    m = sweep == s
    ax.scatter(T[m], Y[m], color=col, marker=mk, s=18, alpha=0.8, label=f"{lab}, n = {m.sum()}")
ax.set(xlabel="Temperature from Hg thermometer, $T$ (°C)", ylabel="New thermometer reading, $y$ (a.u.)",
       title="Raw calibration data")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex2_fig1_raw_data.png", bbox_inches="tight")

# ============ FIGURES 2 & 3: each fit with its residual plot ============
for d, f in fits.items():
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1.2]})
    for s, (col, mk, lab) in SWEEPS.items():
        m = sweep == s
        ax.scatter(T[m], Y[m], color=col, marker=mk, s=16, alpha=0.7, label=lab)
        axr.scatter(T[m], f["res"][m], color=col, marker=mk, s=14, alpha=0.7)
    ax.plot(tt, np.polyval(f["coef"], tt), "k-", lw=2, label=f"Fit: {equation(f['coef'])}")
    ax.text(0.97, 0.97, f"RMSE = {f['rmse']:.2f}\n$R^2$ = {f['r2']:.4f}\nadj. $R^2$ = {f['adj_r2']:.4f}",
            transform=ax.transAxes, ha="right", va="top", bbox=dict(fc="white", ec="0.7"))
    ax.set(ylabel="New thermometer reading, $y$ (a.u.)", title=f"{names[d]} fit")
    ax.legend(loc="lower left")
    axr.axhline(0, color="k", lw=1)
    for k in (-2, 2):
        axr.axhline(k * f["S"], color="0.5", ls=":", lw=1)
    axr.set(xlabel="Temperature from Hg thermometer, $T$ (°C)", ylabel="Residual $y-\\hat{y}$ (a.u.)",
            title=f"Residuals (dotted: ±2S;  runs = {f['runs']}, expected {f['runs_expected']:.0f};  "
                  f"Durbin–Watson = {f['dw']:.2f})")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ex2_fig{d + 1}_{['', 'linear', 'quadratic'][d]}_fit.png", bbox_inches="tight")

# ============ FIGURE 4 (extra): residual distributions compared ============
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lim = np.abs(fits[1]["res"]).max() * 1.05
bins = np.linspace(-lim, lim, 30)
for a_, d in zip(ax, (1, 2)):
    f = fits[d]
    a_.hist(f["res"], bins=bins, color="0.6", edgecolor="k", lw=0.4, density=True)
    xx = np.linspace(-lim, lim, 200)
    a_.plot(xx, stats.norm.pdf(xx, 0, f["S"]), "r-", lw=1.5, label=f"normal, σ = S = {f['S']:.1f}")
    a_.set(xlabel="Residual (a.u.)", ylabel="Probability density",
           title=f"{names[d]}: Shapiro–Wilk p = {f['shapiro_p']:.2g}")
    a_.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex2_fig4_residual_histograms.png", bbox_inches="tight")

# ============ FIGURE 5 (extra): hysteresis between sweeps ============
fig, (ax, axg) = plt.subplots(1, 2, figsize=(12, 4.6))
for s, (col, mk, lab) in SWEEPS.items():
    m = sweep == s
    ax.scatter(T[m], Y[m], color=col, marker=mk, s=12, alpha=0.5)
    ax.plot(tt, np.polyval(sweep_fit[s]["coef"], tt), color=col, lw=2, label=f"{lab} quadratic fit")
ax.set(xlabel="Temperature, $T$ (°C)", ylabel="Reading, $y$ (a.u.)", title="(a) Separate fits for each sweep")
ax.legend()
axg.plot(tt, gap, "k-", lw=2)
axg.axhline(gap.mean(), color="tab:red", ls="--", label=f"mean gap = {gap.mean():.1f} a.u.")
axg.axhline(0, color="0.5", lw=0.8)
axg.set(xlabel="Temperature, $T$ (°C)", ylabel="Cooling − heating reading (a.u.)",
        title="(b) Hysteresis: difference between the two sweeps")
axg.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex2_fig5_hysteresis.png", bbox_inches="tight")

# ============ FIGURE 6 (extra): inverse calibration curve ============
fig, ax = plt.subplots(figsize=(7.5, 5))
yy = np.linspace(Y.min(), Y.max(), 300)
ax.scatter(Y, T, s=10, color="0.5", alpha=0.6, label="data")
ax.plot(yy, np.polyval(inv["coef"], yy), "k-", lw=2,
        label=equation(inv["coef"]).replace("T", "R").replace("y =", "T ="))
ax.set(xlabel="New thermometer reading, $R$ (a.u.)", ylabel="Temperature, $T$ (°C)",
       title=f"Inverse calibration curve (RMSE = {inv['rmse']:.2f} °C)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex2_fig6_inverse_calibration.png", bbox_inches="tight")

# ---------------- save statistics ----------------
rows = {}
for d, f in fits.items():
    rows[names[d]] = {"equation": equation(f["coef"]).replace("$", ""),
                      **{f"coef T^{d - i}": c for i, c in enumerate(f["coef"])},
                      **{f"SE coef T^{d - i}": s for i, s in enumerate(f["se"])},
                      "RMSE": f["rmse"], "S": f["S"], "R^2": f["r2"], "adjusted R^2": f["adj_r2"],
                      "residual runs": f["runs"], "runs expected (random)": f["runs_expected"],
                      "runs test p": f["runs_p"], "Durbin-Watson": f["dw"],
                      "Shapiro-Wilk p": f["shapiro_p"], "AIC": f["aic"]}
rows["Linear vs quadratic"] = {"F statistic": F, "F-test p": p_F}
rows["Hysteresis"] = {"mean gap (a.u.)": gap.mean(), "min gap": gap.min(), "max gap": gap.max()}
rows["Inverse calibration T(R)"] = {"equation": equation(inv["coef"]).replace("$", "").replace("T", "R").replace("y =", "T ="),
                                    "RMSE (°C)": inv["rmse"], "R^2": inv["r2"]}
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/ex2_statistics.csv")
print(f"\nSaved figures and statistics to ./{OUT_DIR}")
plt.show()
