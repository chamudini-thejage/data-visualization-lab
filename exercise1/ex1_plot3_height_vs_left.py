"""
Exercise 1 - Plot 3: Height vs left pointer length
- One graph: male data + fit, female data + fit, and the fit for all students together
- Fit equations in the legend; r, R^2, m, b, S, RMSE (+ standard errors, 95 % CI, p-values)
- Residual plots (y - y_hat) for the separate gender fits and the all-student fit
- Extras: test whether male and female slopes differ; multiple regression with a gender term
Run:  python ex1_plot3_height_vs_left.py   (needs Exercise1_finger_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- settings ----------------
DATA_FILE = "Exercise1_finger_data.xlsx"   # keep in the same folder as this script
OUT_DIR = "ex1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})

# ---------------- load data ----------------
df = pd.read_excel(DATA_FILE, sheet_name="Data")
df = df[df["Gender"].isin(["Male", "Female"])].copy()          # drop the note row
df["L"] = df[[f"L{i}_cm" for i in range(1, 6)]].mean(axis=1)   # mean of 5 left readings
df["R"] = df[[f"R{i}_cm" for i in range(1, 6)]].mean(axis=1)   # mean of 5 right readings
df["sdL"] = df[[f"L{i}_cm" for i in range(1, 6)]].std(axis=1, ddof=1)
df["sdR"] = df[[f"R{i}_cm" for i in range(1, 6)]].std(axis=1, ddof=1)
df["D"] = df["L"] - df["R"]                                    # left minus right
df["H"] = df["Height_cm"]


def regress(x, y):
    """Least-squares line y = m x + b with the full set of fit statistics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    res = y - yhat
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    sxx = np.sum((x - x.mean()) ** 2)
    r = np.corrcoef(x, y)[0, 1]
    S = np.sqrt(ss_res / (n - 2))                 # standard error of the regression
    se_m = S / np.sqrt(sxx)
    se_b = S * np.sqrt(1 / n + x.mean() ** 2 / sxx)
    tcrit = stats.t.ppf(0.975, n - 2)
    p_slope = 2 * stats.t.sf(abs(m / se_m), n - 2)
    return {"n": n, "m": m, "b": b, "r": r, "R2": 1 - ss_res / ss_tot, "S": S,
            "RMSE": np.sqrt(ss_res / n), "se_m": se_m, "se_b": se_b,
            "m_CI95": (m - tcrit * se_m, m + tcrit * se_m),
            "b_CI95": (b - tcrit * se_b, b + tcrit * se_b), "p_slope": p_slope,
            "shapiro_p": stats.shapiro(res).pvalue if n >= 3 else np.nan,
            "yhat": yhat, "res": res, "x": x, "y": y, "tcrit": tcrit, "sxx": sxx}


def ci_band(fit, xx):
    """95 % confidence band of the fitted line at points xx."""
    x = fit["x"]
    se = fit["S"] * np.sqrt(1 / fit["n"] + (xx - x.mean()) ** 2 / fit["sxx"])
    yy = fit["m"] * xx + fit["b"]
    return yy - fit["tcrit"] * se, yy + fit["tcrit"] * se


def stats_table(fits):
    """Turn a dict of fits into a printable / saveable table."""
    rows = {}
    for name, f in fits.items():
        rows[name] = {"n": f["n"], "r": f["r"], "R^2": f["R2"],
                      "slope m": f["m"], "SE(m)": f["se_m"],
                      "m 95% CI low": f["m_CI95"][0], "m 95% CI high": f["m_CI95"][1],
                      "intercept b": f["b"], "SE(b)": f["se_b"],
                      "S (std err of regression)": f["S"], "RMSE": f["RMSE"],
                      "p-value (slope = 0)": f["p_slope"],
                      "Shapiro-Wilk p (residual normality)": f["shapiro_p"]}
    return pd.DataFrame(rows)


# =====================================================================
# Plot 3: height vs left pointer length
# =====================================================================
Y, YLABEL, TAG = "L", "Left pointer length (cm)", "left"
groups = {"Male": df[df["Gender"] == "Male"], "Female": df[df["Gender"] == "Female"], "All students": df}
fits = {g: regress(s["H"], s[Y]) for g, s in groups.items()}
tab = stats_table(fits)
print(f"Height vs {YLABEL}:  fit y = m*height + b")
print(tab.round(5).to_string())

# extra 1: do the male and female slopes differ?
fm, ff = fits["Male"], fits["Female"]
t_slope = (fm["m"] - ff["m"]) / np.sqrt(fm["se_m"] ** 2 + ff["se_m"] ** 2)
p_slope_diff = 2 * stats.t.sf(abs(t_slope), fm["n"] + ff["n"] - 4)
print(f"\nMale vs female slope difference: t = {t_slope:.3f}, p = {p_slope_diff:.4f}")

# extra 2: is there a gender effect once height is accounted for?  y = b0 + b1*height + b2*female
X = np.column_stack([np.ones(len(df)), df["H"], (df["Gender"] == "Female").astype(float)])
yv = df[Y].values
beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
res_mr = yv - X @ beta
dof = len(yv) - 3
cov = (res_mr @ res_mr / dof) * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(cov))
p_beta = 2 * stats.t.sf(np.abs(beta / se_beta), dof)
r2_mr = 1 - np.sum(res_mr ** 2) / np.sum((yv - yv.mean()) ** 2)
print("Multiple regression y = b0 + b1*height + b2*female:")
for nm, bv, se, pv in zip(["b0", "b1 (height)", "b2 (female)"], beta, se_beta, p_beta):
    print(f"   {nm:12s} = {bv:+.5f} ± {se:.5f}   p = {pv:.4g}")
print(f"   R^2 = {r2_mr:.4f}  (height alone: {fits['All students']['R2']:.4f})")

# ---------- figure: scatter + three fits (top), residuals (bottom) ----------
STY = {"Male": ("tab:blue", "o", "-"), "Female": ("tab:red", "s", "-"), "All students": ("k", None, "--")}
fig = plt.figure(figsize=(11, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.3])
ax = fig.add_subplot(gs[0, :])
for g in ("Male", "Female"):
    c, mk, _ = STY[g]
    s = groups[g]
    ax.scatter(s["H"], s[Y], color=c, marker=mk, s=28, alpha=0.75, edgecolor="k", lw=0.4,
               label=f"{g} data (n = {len(s)})")
for g, f in fits.items():
    c, _, ls = STY[g]
    xx = np.linspace(f["x"].min(), f["x"].max(), 200)
    sign = "+" if f["b"] >= 0 else "−"
    ax.plot(xx, f["m"] * xx + f["b"], color=c, ls=ls, lw=2,
            label=f"{g} fit: y = {f['m']:.4f}x {sign} {abs(f['b']):.3f}  "
                  f"(r = {f['r']:.3f}, $R^2$ = {f['R2']:.3f})")
    if g == "All students":
        lo, hi = ci_band(f, xx)
        ax.fill_between(xx, lo, hi, color="0.5", alpha=0.15, label="95 % confidence band (all students)")
ax.set_xlabel("Height (cm)")
ax.set_ylabel(YLABEL)
ax.set_title(f"Height vs {YLABEL.split(' (')[0].lower()} by gender")
ax.legend(fontsize=8, loc="best")

ax1 = fig.add_subplot(gs[1, 0])
for g in ("Male", "Female"):
    c, mk, _ = STY[g]
    ax1.scatter(fits[g]["x"], fits[g]["res"], color=c, marker=mk, s=18, alpha=0.75, label=g)
ax1.axhline(0, color="k", lw=1)
ax1.set(xlabel="Height (cm)", ylabel="Residual $y-\\hat{y}$ (cm)", title="(b) Residuals of separate gender fits")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
fa = fits["All students"]
for g in ("Male", "Female"):
    c, mk, _ = STY[g]
    msk = (df["Gender"] == g).values
    ax2.scatter(fa["x"][msk], fa["res"][msk], color=c, marker=mk, s=18, alpha=0.75, label=g)
ax2.axhline(0, color="k", lw=1)
for k in (-2, 2):
    ax2.axhline(k * fa["S"], color="0.5", ls=":", lw=1)
ax2.set(xlabel="Height (cm)", title="(c) Residuals of all-student fit (dotted: ±2S)")
fig.text(0.01, 0.98, "(a)", fontsize=12, va="top")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/plot3_height_vs_{TAG}.png", bbox_inches="tight")

# ---------- save numbers ----------
extra = pd.DataFrame({"All students": {
    "Slope difference M-F t": t_slope, "Slope difference M-F p": p_slope_diff,
    "MR b0": beta[0], "MR b1 height": beta[1], "MR b2 female": beta[2],
    "MR p(b1)": p_beta[1], "MR p(b2)": p_beta[2], "MR R^2": r2_mr}})
pd.concat([tab, extra]).to_csv(f"{OUT_DIR}/plot3_statistics.csv")
print(f"\nSaved figure and statistics to ./{OUT_DIR}")
plt.show()
