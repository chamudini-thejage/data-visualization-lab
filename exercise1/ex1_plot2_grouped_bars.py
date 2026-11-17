"""
Exercise 1 - Plot 2: Grouped (clustered) bar charts, male vs female
  (A) left pointer length, (B) right pointer length, (C) left - right difference
- Bar chart: top = number of people per bin, bottom = percentage of each gender
  (the percentage panel is the fair comparison: 62 males vs 24 females)
- ECDF plot: cumulative fraction of each group, independent of group size, with the KS gap marked
- Bootstrap: 95 % confidence intervals for the mean difference, median difference and Cohen's d
- Per bin: absolute delta, percentage discrepancy, ratio, category sums, % of total, dominant gender
  -> main versions use % of own gender; count versions kept as descriptive only
- Summary: directional shifts (flips), correlation of gaps, Welch t, Mann-Whitney, KS test, Cohen's d
Run:  python ex1_plot2_grouped_bars.py   (needs Exercise1_finger_data.xlsx in the same folder)
"""
import os
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
# Plot 2: grouped (clustered) bar charts, male vs female, for
#         (A) left length, (B) right length, (C) difference L - R
# =====================================================================
CASES = [("L", "Left pointer length (cm)", 0.25, "plot2a_left_grouped_bar"),
         ("R", "Right pointer length (cm)", 0.25, "plot2b_right_grouped_bar"),
         ("D", "Difference, left − right (cm)", 0.05, "plot2c_difference_grouped_bar")]


def make_edges(values, width, centred_on_zero=False):
    """Bin edges of a fixed width that cover all values."""
    if centred_on_zero:                        # bins centred on 0, ±0.05, ±0.10 ...
        k = np.ceil((np.abs(values).max() + width / 2) / width)
        return np.arange(-k * width - width / 2, k * width + width / 2 + 1e-9, width)
    lo = np.floor(values.min() / width) * width
    hi = np.ceil(values.max() / width) * width
    return np.arange(lo, hi + width / 2, width)


def bin_comparison(m_counts, f_counts, centres):
    """Male vs female comparison metrics for every bin.
    Main metrics use % of each gender's own total (fair for unequal group sizes);
    count-based metrics are kept for description only."""
    M, F = m_counts.astype(float), f_counts.astype(float)
    Mp, Fp = 100 * M / M.sum(), 100 * F / F.sum()            # % of each gender's own total
    t = pd.DataFrame({"Bin centre": centres, "Male count": M, "Female count": F})
    t["Bin sum (M+F)"] = M + F                                                    # category sum
    t["% of all students"] = 100 * (M + F) / (M.sum() + F.sum())                 # aggregated % of total
    t["Male % of males"] = Mp
    t["Female % of females"] = Fp
    # --- main (group-size independent) metrics ---
    t["Absolute delta |M%-F%| (pp)"] = np.abs(Mp - Fp)
    t["Percentage discrepancy of % (%)"] = np.where(Mp + Fp > 0, 100 * np.abs(Mp - Fp) / ((Mp + Fp) / 2), np.nan)
    t["Ratio M%/F%"] = np.where(Fp > 0, Mp / np.where(Fp > 0, Fp, 1), np.nan)
    t["Dominant (% of group)"] = np.select([Mp > Fp, Fp > Mp], ["Male", "Female"], "Tie")
    # --- descriptive only (biased by 62 vs 24 group sizes) ---
    t["[counts] Absolute delta |M-F|"] = np.abs(M - F)
    t["[counts] Percentage discrepancy (%)"] = np.where(M + F > 0, 100 * np.abs(M - F) / ((M + F) / 2), np.nan)
    t["[counts] Ratio M/F"] = np.where(F > 0, M / np.where(F > 0, F, 1), np.nan)
    t["[counts] Dominant"] = np.select([M > F, F > M], ["Male", "Female"], "Tie")
    return t


def bootstrap_ci(a, b, func=np.mean, n_boot=10000, seed=1):
    """Bootstrap 95 % CI of func(a) - func(b): resample each group with replacement."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.array([func(rng.choice(a, len(a))) - func(rng.choice(b, len(b))) for _ in range(n_boot)])
    return np.percentile(diffs, [2.5, 97.5]), diffs


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def count_flips(dominant, occupied):
    """Directional shifts: how often the dominant gender switches between successive occupied bins."""
    seq = [d for d, o in zip(dominant, occupied) if o and d != "Tie"]
    flips = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    return flips, len(seq)


summary_rows = {}
writer = pd.ExcelWriter(f"{OUT_DIR}/plot2_bin_statistics.xlsx")
for col, label, width, fname in CASES:
    male, female = df.loc[df["Gender"] == "Male", col], df.loc[df["Gender"] == "Female", col]
    edges = make_edges(df[col], width, centred_on_zero=(col == "D"))
    centres = np.round((edges[:-1] + edges[1:]) / 2, 4) + 0.0   # +0.0 removes "-0.00"
    mc, _ = np.histogram(male, edges)
    fc, _ = np.histogram(female, edges)
    t = bin_comparison(mc, fc, centres)
    t.insert(0, "Bin range (cm)", [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(edges[:-1], edges[1:])])
    occupied = (mc + fc) > 0

    flips_c, nb_c = count_flips(t["[counts] Dominant"], occupied)
    flips_p, nb_p = count_flips(t["Dominant (% of group)"], occupied)
    gap_counts = mc - fc
    gap_pct = t["Male % of males"] - t["Female % of females"]
    r_gap_c, p_gap_c = stats.pearsonr(centres[occupied], gap_counts[occupied])
    r_gap_p, p_gap_p = stats.pearsonr(centres[occupied], gap_pct[occupied])
    r_mf, p_mf = stats.pearsonr(t["Male % of males"][occupied], t["Female % of females"][occupied])

    # group-level comparison (extra)
    welch = stats.ttest_ind(male, female, equal_var=False)
    mw = stats.mannwhitneyu(male, female)
    ks = stats.ks_2samp(male, female)
    cohen_d = cohens_d(male, female)
    ci_mean, boot_diffs = bootstrap_ci(male, female, np.mean)
    ci_median, _ = bootstrap_ci(male, female, np.median)
    rng = np.random.default_rng(2)
    d_boot = [cohens_d(rng.choice(male.values, len(male)), rng.choice(female.values, len(female))) for _ in range(5000)]
    ci_d = np.percentile(d_boot, [2.5, 97.5])

    s = {"Bin width (cm)": width, "Number of bins": len(centres),
         "Male total (category sum)": int(mc.sum()), "Female total (category sum)": int(fc.sum()),
         "Male % of all students": 100 * mc.sum() / (mc.sum() + fc.sum()),
         "Female % of all students": 100 * fc.sum() / (mc.sum() + fc.sum()),
         "Overall |M-F| (count)": abs(int(mc.sum() - fc.sum())),
         "Overall ratio M/F": mc.sum() / fc.sum(),
         "Sum of bin |M-F| (count) [descriptive]": t["[counts] Absolute delta |M-F|"].sum(),
         "Sum of bin |M%-F%| / 2 (distribution non-overlap, %)": t["Absolute delta |M%-F%| (pp)"].sum() / 2,
         "Directional flips (% of group) [main]": f"{flips_p} in {nb_p} bins",
         "Directional flips (counts) [descriptive]": f"{flips_c} in {nb_c} bins",
         "Correlation of gaps r (% gap vs bin) [main]": r_gap_p, "  p ": p_gap_p,
         "Correlation of gaps r (count gap vs bin) [descriptive]": r_gap_c, "  p": p_gap_c,
         "r between male% and female% profiles": r_mf, "  p  ": p_mf,
         "Male mean ± SD (cm)": f"{male.mean():.3f} ± {male.std():.3f}",
         "Female mean ± SD (cm)": f"{female.mean():.3f} ± {female.std():.3f}",
         "Male median (cm)": male.median(), "Female median (cm)": female.median(),
         "Mean difference M-F (cm)": male.mean() - female.mean(),
         "Bootstrap 95% CI of mean difference (cm)": f"{ci_mean[0]:.3f} to {ci_mean[1]:.3f}",
         "Median difference M-F (cm)": male.median() - female.median(),
         "Bootstrap 95% CI of median difference (cm)": f"{ci_median[0]:.3f} to {ci_median[1]:.3f}",
         "Percentage discrepancy of means (%)": (100 * abs(male.mean() - female.mean()) / ((male.mean() + female.mean()) / 2)
                                                 if col != "D" else "not meaningful: means are close to 0"),
         "Ratio of means M/F": male.mean() / female.mean() if col != "D" else "not meaningful: means are close to 0",
         "Welch t-test p": welch.pvalue, "Mann-Whitney U p": mw.pvalue,
         "Kolmogorov-Smirnov p (shape)": ks.pvalue, "Cohen's d (effect size)": cohen_d,
         "Bootstrap 95% CI of Cohen's d": f"{ci_d[0]:.2f} to {ci_d[1]:.2f}",
         "KS statistic D (max ECDF gap)": ks.statistic}
    summary_rows[label] = s
    t.to_excel(writer, sheet_name=fname.split("_")[0][-2:] + "_" + col + "_bins", index=False)

    print(f"\n========== {label} ==========")
    print(t.round(2).to_string(index=False))
    for k, v in s.items():
        print(f"{k:55s} {v:.4g}" if isinstance(v, (float, np.floating)) else f"{k:55s} {v}")

    # ---------- figure: counts (top) and % of each gender (bottom) ----------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    bw = width * 0.4
    for a, ym, yf, ylab in [(a1, mc, fc, "Number of people"),
                            (a2, t["Male % of males"], t["Female % of females"], "Percentage of own gender (%)")]:
        bm = a.bar(centres - bw / 2, ym, width=bw, color="tab:blue", edgecolor="k", lw=0.6,
                   label=f"Male (n = {len(male)})")
        bf = a.bar(centres + bw / 2, yf, width=bw, color="tab:red", edgecolor="k", lw=0.6,
                   label=f"Female (n = {len(female)})")
        if a is a1:
            a.bar_label(bm, labels=[str(v) if v else "" for v in ym], fontsize=7, padding=1)
            a.bar_label(bf, labels=[str(v) if v else "" for v in yf], fontsize=7, padding=1)
        a.axvline(male.mean(), color="tab:blue", ls="--", lw=1.3, label=f"Male mean = {male.mean():.2f} cm")
        a.axvline(female.mean(), color="tab:red", ls="--", lw=1.3, label=f"Female mean = {female.mean():.2f} cm")
        a.set_ylabel(ylab)
        a.grid(axis="x", alpha=0)
    a1.set_title(f"{label.split(' (')[0]}: distribution by gender "
                 f"(Welch p = {welch.pvalue:.2g}, Cohen's d = {cohen_d:.2f})")
    a1.legend(fontsize=8)
    a1.set_ylim(0, max(mc.max(), fc.max()) * 1.15)
    a2.set_xticks(centres)
    if col == "D":
        a2.set_xticklabels([f"{c:+.2f}" for c in centres], rotation=45)
        a2.set_xlabel(f"{label}  [bin centre, bin width {width} cm]")
    else:
        a2.set_xticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])],
                           rotation=45, ha="right")
        a2.set_xlabel(f"{label}  [bin range, bin width {width} cm]")
    if col == "D":
        for a in (a1, a2):
            a.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{fname}.png", bbox_inches="tight")

    # ---------- figure: ECDF (left) and bootstrap distribution of the mean difference (right) ----------
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for vals, c, g in [(male, "tab:blue", "Male"), (female, "tab:red", "Female")]:
        xs = np.sort(vals.values)
        e1.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=c, lw=2,
                label=f"{g} (n = {len(xs)})")
    # mark the largest vertical gap = KS statistic D
    grid = np.sort(df[col].values)
    Fm = np.searchsorted(np.sort(male.values), grid, side="right") / len(male)
    Ff = np.searchsorted(np.sort(female.values), grid, side="right") / len(female)
    k = int(np.argmax(np.abs(Fm - Ff)))
    e1.vlines(grid[k], min(Fm[k], Ff[k]), max(Fm[k], Ff[k]), color="k", lw=2, linestyles=":",
              label=f"max gap D = {ks.statistic:.2f} (KS p = {ks.pvalue:.2g})")
    if col == "D":
        e1.axvline(0, color="0.5", lw=0.8)
    e1.set(xlabel=label, ylabel="Fraction of group with value ≤ x",
           title="(a) Empirical cumulative distribution (ECDF)", ylim=(-0.02, 1.02))
    e1.legend(loc="lower right", fontsize=8)

    e2.hist(boot_diffs, bins=50, color="0.6", edgecolor="k", lw=0.3)
    e2.axvline(male.mean() - female.mean(), color="k", lw=2,
               label=f"observed = {male.mean() - female.mean():+.3f} cm")
    for v in ci_mean:
        e2.axvline(v, color="tab:purple", ls="--", lw=1.5)
    e2.axvline(0, color="tab:green", ls=":", lw=1.5, label="zero (no difference)")
    e2.plot([], [], color="tab:purple", ls="--", label=f"95 % CI: {ci_mean[0]:+.3f} to {ci_mean[1]:+.3f} cm")
    e2.set(xlabel="Mean difference, male − female (cm)", ylabel="Bootstrap samples",
           title="(b) Bootstrap of the mean difference (10 000 resamples)")
    e2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{fname.replace('grouped_bar', 'ecdf_bootstrap')}.png", bbox_inches="tight")

pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary")
writer.close()
print(f"\nSaved figures and bin tables to ./{OUT_DIR}")
plt.show()
