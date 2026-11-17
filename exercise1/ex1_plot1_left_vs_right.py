"""
Exercise 1 - Plot 1: Left vs right pointer finger length
- Scatter of all students (mean of 5 readings, ±1 SD error bars) with least-squares fit
- Fit equation in the legend; r, R^2, m, b, S, RMSE (+ standard errors, 95 % CI, p-values)
- Residual plot (y - y_hat)
- Extras: test of slope = 1 and intercept = 0, paired t-test, Bland-Altman agreement plot
Run:  python ex1_plot1_left_vs_right.py   (needs Exercise1_finger_data.xlsx in the same folder)
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
# Plot 1: left vs right pointer length
# =====================================================================
fit = regress(df["L"], df["R"])
print("Left vs right pointer length: fit R = m*L + b")
print(stats_table({"All students": fit}).round(5).to_string())

# extra: is the line consistent with perfect symmetry (m = 1, b = 0)?
t_m1 = (fit["m"] - 1) / fit["se_m"]
p_m1 = 2 * stats.t.sf(abs(t_m1), fit["n"] - 2)
t_b0 = fit["b"] / fit["se_b"]
p_b0 = 2 * stats.t.sf(abs(t_b0), fit["n"] - 2)
print(f"\nTest slope = 1:     t = {t_m1:.3f}, p = {p_m1:.4f}")
print(f"Test intercept = 0: t = {t_b0:.3f}, p = {p_b0:.4f}")

# extra: paired comparison of the two hands
t_pair, p_pair = stats.ttest_rel(df["L"], df["R"])
w_p = stats.wilcoxon(df["L"], df["R"]).pvalue
print(f"Paired t-test L vs R: t = {t_pair:.3f}, p = {p_pair:.4f};  Wilcoxon p = {w_p:.4f}")

# extra: Bland-Altman agreement analysis
mean_LR = (df["L"] + df["R"]) / 2
bias = df["D"].mean()
sd_d = df["D"].std(ddof=1)
loa = (bias - 1.96 * sd_d, bias + 1.96 * sd_d)
print(f"Bland-Altman: bias (L-R) = {bias:.4f} cm, 95% limits of agreement = {loa[0]:.3f} to {loa[1]:.3f} cm")

# ---------- Figure 1: scatter + fit (top) and residuals (bottom) ----------
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2]})
for g, mk, c in [("Male", "o", "tab:blue"), ("Female", "s", "tab:red")]:
    s = df[df["Gender"] == g]
    ax.errorbar(s["L"], s["R"], xerr=s["sdL"], yerr=s["sdR"], fmt=mk, color=c, ms=5,
                alpha=0.75, capsize=2, elinewidth=0.8, label=f"{g} (n = {len(s)})")
xx = np.linspace(df["L"].min() - 0.2, df["L"].max() + 0.2, 200)
lo, hi = ci_band(fit, xx)
ax.fill_between(xx, lo, hi, color="k", alpha=0.12, label="95 % confidence band")
ax.plot(xx, fit["m"] * xx + fit["b"], "k-", lw=2,
        label=f"Fit: $R$ = {fit['m']:.3f}$L$ {fit['b']:+.3f}\n$r$ = {fit['r']:.3f}, $R^2$ = {fit['R2']:.3f}")
ax.plot(xx, xx, "g--", lw=1.2, label="$R = L$ (perfect symmetry)")
ax.set_ylabel("Right pointer length, $R$ (cm)")
ax.set_title("Right vs left pointer finger length (mean of 5 readings, ±1 SD)")
ax.legend(loc="upper left")
ax.set_aspect("equal", adjustable="datalim")

axr.scatter(df["L"], fit["res"], c=df["Gender"].map({"Male": "tab:blue", "Female": "tab:red"}),
            s=18, alpha=0.75)
axr.axhline(0, color="k", lw=1)
axr.axhline(2 * fit["S"], color="0.5", ls=":", lw=1)
axr.axhline(-2 * fit["S"], color="0.5", ls=":", lw=1, label="±2S")
axr.set_xlabel("Left pointer length, $L$ (cm)")
axr.set_ylabel("Residual $y-\\hat{y}$ (cm)")
axr.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/plot1_left_vs_right.png", bbox_inches="tight")

# ---------- Figure 1b (extra): Bland-Altman plot ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
for g, mk, c in [("Male", "o", "tab:blue"), ("Female", "s", "tab:red")]:
    s = df["Gender"] == g
    ax.scatter(mean_LR[s], df["D"][s], marker=mk, color=c, s=22, alpha=0.75, label=g)
ax.axhline(bias, color="k", lw=1.5, label=f"Bias = {bias:+.3f} cm")
ax.axhline(loa[0], color="k", ls="--", lw=1, label=f"95 % limits: {loa[0]:+.3f} to {loa[1]:+.3f} cm")
ax.axhline(loa[1], color="k", ls="--", lw=1)
ax.axhline(0, color="g", ls=":", lw=1)
ax.set_xlabel("Mean of left and right, $(L+R)/2$ (cm)")
ax.set_ylabel("Difference $L-R$ (cm)")
ax.set_title(f"Bland–Altman plot (paired t-test p = {p_pair:.3f})")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/plot1b_bland_altman.png", bbox_inches="tight")

# ---------- save numbers ----------
tab = stats_table({"All students": fit})
extra = pd.DataFrame({"All students": {
    "t (slope = 1)": t_m1, "p (slope = 1)": p_m1, "t (intercept = 0)": t_b0, "p (intercept = 0)": p_b0,
    "paired t": t_pair, "paired t p-value": p_pair, "Wilcoxon p-value": w_p,
    "Bland-Altman bias L-R (cm)": bias, "LoA low (cm)": loa[0], "LoA high (cm)": loa[1]}})
pd.concat([tab, extra]).to_csv(f"{OUT_DIR}/plot1_statistics.csv")
print(f"\nSaved figures and statistics to ./{OUT_DIR}")
plt.show()
