"""
Exercise 5 - Head size and brain weight, analysed step by step (size-adjusted group comparison / ANCOVA)
  Step 1  Understand the question and the variables
  Step 2  Look at the target (brain weight) and the covariate (head size) alone
  Step 3  Screen: brain vs head size; raw group differences; is head size itself different between groups?
  Step 4  Split by group and look again: one shared line or shifted lines? ratio and size-adjusted residuals
  Step 5  Build the model (ANCOVA): head size + gender + age, test for unequal slopes, allometric log-log fit
  Step 6  Check the model with residuals
  Step 7  Summarise: raw vs size-adjusted differences, effect sizes with 95 % CI
Each step prints what it found and the decision it leads to.
Run:  python ex5_brain_stepwise.py   (needs Exercise5_brain_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = "Exercise5_brain_data.xlsx"
OUT_DIR = "ex5_step_outputs"
AGE_LABELS = {1: ">18", 2: "<18"}        # as given in the lab sheet (see the Notes sheet in the Excel file)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})
pd.set_option("display.width", 200)
BOX_MEAN = dict(marker="D", markerfacecolor="white", markeredgecolor="k", markersize=5)


def header(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


def decision(text):
    print(f"--> DECISION: {text}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")


def ols(X, y, names):
    """Least squares with SE, t, p, covariance, R^2, adjusted R^2, RMSE, S."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    res = y - yhat
    n, k = X.shape
    ss_res, ss_tot = np.sum(res ** 2), np.sum((y - y.mean()) ** 2)
    S2 = ss_res / (n - k)
    cov = S2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    r2 = 1 - ss_res / ss_tot
    return {"table": pd.DataFrame({"coefficient": beta, "SE": se, "t": beta / se, "p": p},
                                  index=["intercept"] + names),
            "beta": beta, "cov": cov, "R2": r2, "adjR2": 1 - (1 - r2) * (n - 1) / (n - k),
            "RMSE": np.sqrt(ss_res / n), "S": np.sqrt(S2), "yhat": yhat, "res": res,
            "ss_res": ss_res, "k": k, "dof": n - k}


def f_test(small, big):
    """Partial F-test: does the bigger (nested) model fit significantly better?"""
    q = big["k"] - small["k"]
    F = ((small["ss_res"] - big["ss_res"]) / q) / (big["ss_res"] / big["dof"])
    return F, stats.f.sf(F, q, big["dof"])


# =============================================================================================
header("STEP 1 - Understand the question and the variables")
# =============================================================================================
df = pd.read_excel(DATA_FILE, sheet_name="Data")
df.columns = ["g", "a", "head", "brain"]
df["Gender"] = df.g.map({1: "Male", 2: "Female"})
df["Age"] = df.a.map(AGE_LABELS)
df["fem"] = (df.g == 2).astype(int)
df["age2"] = (df.a == 2).astype(int)                       # 1 = age group 2
GROUPS = {("Male", AGE_LABELS[1]): ("tab:blue", "o"), ("Male", AGE_LABELS[2]): ("tab:cyan", "^"),
          ("Female", AGE_LABELS[1]): ("tab:red", "o"), ("Female", AGE_LABELS[2]): ("tab:orange", "^")}
print(f"{len(df)} people, missing values: {int(df.isna().sum().sum())}")
print("Group sizes:\n", pd.crosstab(df.Gender, df.Age, margins=True).to_string())
print("Variables: brain weight (g) = TARGET, continuous | head size (cm^3) = continuous COVARIATE | "
      "gender, age group = categorical (2 levels each)")
print("Expectation: bigger head -> heavier brain; males have bigger heads. Question: do groups differ "
      "AFTER accounting for head size?")
decision("scatter brain vs head size coloured by group; box plots for raw group comparisons; "
         "final analysis = regression with head size + group terms (ANCOVA)")

# =============================================================================================
header("STEP 2 - Look at the target and the covariate alone")
# =============================================================================================
for v, lab in [("brain", "Brain weight (g)"), ("head", "Head size (cm³)")]:
    print(f"{lab}: mean {df[v].mean():.1f}, median {df[v].median():.1f}, SD {df[v].std():.1f}, "
          f"skew {stats.skew(df[v]):.2f}, Shapiro p = {stats.shapiro(df[v]).pvalue:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, v, lab in [(ax[0], "brain", "Brain weight (g)"), (ax[1], "head", "Head size (cm³)")]:
    a.hist(df[v], bins=22, color="0.6", edgecolor="k", lw=0.4, density=True)
    xx = np.linspace(df[v].min(), df[v].max(), 200)
    a.plot(xx, stats.norm.pdf(xx, df[v].mean(), df[v].std()), "r-", lw=1.5, label="normal curve")
    a.axvline(df[v].mean(), color="k", lw=1.5, label=f"mean = {df[v].mean():.0f}")
    a.set(xlabel=lab, ylabel="Probability density", title=f"{lab.split(' (')[0]}: skew = {stats.skew(df[v]):.2f}")
    a.legend()
save(fig, "step2_distributions")
decision("both are roughly symmetric (|skew| < 0.5, no separate peaks) -> means, t-tests and ordinary "
         "least-squares regression are appropriate (unlike the skewed insurance charges)")

# =============================================================================================
header("STEP 3 - Screen: brain vs head size, and raw group differences")
# =============================================================================================
r, p = stats.pearsonr(df["head"], df.brain)
print(f"Brain vs head size: r = {r:.3f} (R^2 = {r ** 2:.3f}), p = {p:.1e}")
raw = {}
for var in ["brain", "head"]:
    for name, col, a_, b_ in [("Male - Female", "Gender", "Male", "Female"),
                              (f"{AGE_LABELS[1]} - {AGE_LABELS[2]}", "Age", AGE_LABELS[1], AGE_LABELS[2])]:
        A, B = df[df[col] == a_][var], df[df[col] == b_][var]
        t = stats.ttest_ind(A, B, equal_var=False)
        sp = np.sqrt(((len(A) - 1) * A.var() + (len(B) - 1) * B.var()) / (len(A) + len(B) - 2))
        raw[f"{var}: {name}"] = {"difference": A.mean() - B.mean(), "% of mean": 100 * (A.mean() - B.mean()) / df[var].mean(),
                                 "Welch p": t.pvalue, "Cohen d": (A.mean() - B.mean()) / sp}
raw = pd.DataFrame(raw).T
print(raw.round(4).to_string())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax[0].scatter(df["head"], df.brain, s=12, alpha=0.6, color="0.3")
ax[0].set(xlabel="Head size (cm³)", ylabel="Brain weight (g)", title=f"(a) Brain vs head size: r = {r:.2f}")
order = list(GROUPS)
for a, v, lab in [(ax[1], "brain", "Brain weight (g)"), (ax[2], "head", "Head size (cm³)")]:
    data = [df[(df.Gender == g) & (df.Age == ag)][v] for g, ag in order]
    bp = a.boxplot(data, patch_artist=True, widths=0.6, showmeans=True, meanprops=BOX_MEAN)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(GROUPS[k][0])
        patch.set_alpha(0.5)
    a.set_xticks(range(1, 5), [f"{g[0]}\n{ag}\nn={len(d)}" for (g, ag), d in zip(order, data)])
    a.set(ylabel=lab)
ax[1].set_title("(b) Brain weight by group")
ax[2].set_title("(c) Head size by group: same pattern!")
save(fig, "step3_screening")
decision("brain weight differs by gender and age, BUT head size differs in exactly the same way -> "
         "the raw differences may be just a size effect (confounding). Compare groups at EQUAL head size.")

# =============================================================================================
header("STEP 4 - Split by group and look again")
# =============================================================================================
common = stats.linregress(df["head"], df.brain)
df["res_common"] = df.brain - (common.intercept + common.slope * df["head"])   # size-adjusted brain weight
df["ratio"] = df.brain / df["head"] * 1000                                     # mg per cm^3
slopes = {}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
for (g, ag), (c, mk) in GROUPS.items():
    s = df[(df.Gender == g) & (df.Age == ag)]
    l = stats.linregress(s["head"], s.brain)
    slopes[f"{g}, {ag}"] = {"n": len(s), "slope (g/cm^3)": l.slope, "SE": l.stderr, "r": l.rvalue,
                            "mean ratio (mg/cm^3)": s.ratio.mean(), "mean size-adjusted residual (g)": s.res_common.mean()}
    ax[0].scatter(s["head"], s.brain, s=16, alpha=0.6, color=c, marker=mk, label=f"{g}, age {ag}")
    xx = np.linspace(s["head"].min(), s["head"].max(), 50)
    ax[0].plot(xx, l.intercept + l.slope * xx, color=c, lw=1.8)
xx = np.linspace(df["head"].min(), df["head"].max(), 50)
ax[0].plot(xx, common.intercept + common.slope * xx, "k--", lw=2, label="common line (all)")
ax[0].set(xlabel="Head size (cm³)", ylabel="Brain weight (g)", title="(a) Each group's own line")
ax[0].legend(fontsize=8)
for a, v, lab, t in [(ax[1], "ratio", "Brain / head size (mg/cm³)", "(b) Ratio: size-free comparison"),
                     (ax[2], "res_common", "Residual from common line (g)", "(c) Size-adjusted brain weight")]:
    data = [df[(df.Gender == g) & (df.Age == ag)][v] for g, ag in order]
    bp = a.boxplot(data, patch_artist=True, widths=0.6, showmeans=True, meanprops=BOX_MEAN)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(GROUPS[k][0])
        patch.set_alpha(0.5)
    a.set_xticks(range(1, 5), [f"{g[0]}\n{ag}" for g, ag in order])
    a.set(ylabel=lab, title=t)
ax[2].axhline(0, color="k", lw=1)
save(fig, "step4_split_by_group")
slopes = pd.DataFrame(slopes).T
print(slopes.round(4).to_string())
for name, col, a_, b_ in [("gender", "Gender", "Male", "Female"), ("age", "Age", AGE_LABELS[1], AGE_LABELS[2])]:
    A, B = df[df[col] == a_], df[df[col] == b_]
    print(f"{name}: ratio {a_} {A.ratio.mean():.1f} vs {b_} {B.ratio.mean():.1f} mg/cm^3 "
          f"(p = {stats.ttest_ind(A.ratio, B.ratio, equal_var=False).pvalue:.3f}); "
          f"size-adjusted difference {A.res_common.mean() - B.res_common.mean():+.1f} g "
          f"(p = {stats.ttest_ind(A.res_common, B.res_common, equal_var=False).pvalue:.3f})")
decision("the four groups follow nearly parallel lines close to one common line; after removing head size the "
         "gender gap shrinks from ~113 g to a few tens of g -> model with a common slope + group shifts, "
         "and formally test whether slopes differ")

# =============================================================================================
header("STEP 5 - Build the model (ANCOVA) and test it")
# =============================================================================================
y = df.brain.to_numpy(float)
df["head_c"] = df["head"] - df["head"].mean()           # centred head size (makes intercepts meaningful)
df["head_x_fem"] = df.head_c * df.fem
df["head_x_age2"] = df.head_c * df.age2
specs = {"M1: head size only": ["head_c"],
         "M2: + gender": ["head_c", "fem"],
         "M3: + gender + age (ANCOVA)": ["head_c", "fem", "age2"],
         "M4: + unequal slopes": ["head_c", "fem", "age2", "head_x_fem", "head_x_age2"],
         "M0: gender + age only (no head size)": ["fem", "age2"]}
fits = {m: ols(df[c].to_numpy(float), y, c) for m, c in specs.items()}
comp = pd.DataFrame({m: {"terms": len(c), "R^2": f["R2"], "adjusted R^2": f["adjR2"], "RMSE (g)": f["RMSE"]}
                     for (m, c), f in zip(specs.items(), fits.values())}).T
print(comp.round(4).to_string())
for a_, b_ in [("M1: head size only", "M2: + gender"), ("M2: + gender", "M3: + gender + age (ANCOVA)"),
               ("M3: + gender + age (ANCOVA)", "M4: + unequal slopes")]:
    F, pF = f_test(fits[a_], fits[b_])
    print(f"F-test {a_[:2]} -> {b_[:2]}: F = {F:.2f}, p = {pF:.4f}")
final = fits["M3: + gender + age (ANCOVA)"]
print("\nFinal model M3 (head size centred at its mean, %.0f cm^3):" % df["head"].mean())
print(final["table"].round(4).to_string())

# adjusted means: every group evaluated at the SAME (mean) head size
adj = {}
for (g, ag) in GROUPS:
    x0 = np.array([1, 0, g == "Female", ag == AGE_LABELS[2]], float)
    m, se = x0 @ final["beta"], np.sqrt(x0 @ final["cov"] @ x0)
    rawm = df[(df.Gender == g) & (df.Age == ag)].brain.mean()
    adj[f"{g}, {ag}"] = {"raw mean (g)": rawm, "adjusted mean at mean head size (g)": m,
                         "95% CI low": m - 1.96 * se, "95% CI high": m + 1.96 * se}
adj = pd.DataFrame(adj).T
print("\nRaw vs size-adjusted group means:")
print(adj.round(1).to_string())

# allometry: brain = c * head^b  ->  log(brain) = log(c) + b log(head)
la = stats.linregress(np.log10(df["head"]), np.log10(df.brain))
tcrit = stats.t.ppf(0.975, len(df) - 2)
b_lo, b_hi = la.slope - tcrit * la.stderr, la.slope + tcrit * la.stderr
p_b1 = 2 * stats.t.sf(abs((la.slope - 1) / la.stderr), len(df) - 2)
print(f"\nAllometric fit: brain = {10 ** la.intercept:.3f} * head^{la.slope:.3f}  "
      f"(b 95% CI {b_lo:.3f} to {b_hi:.3f}; test b = 1: p = {p_b1:.4f})")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for (g, ag), (c, mk) in GROUPS.items():
    s = df[(df.Gender == g) & (df.Age == ag)]
    ax[0].scatter(s["head"], s.brain, s=14, alpha=0.5, color=c, marker=mk, label=f"{g}, age {ag}")
    x0 = np.column_stack([np.ones(50), xx - df["head"].mean(), np.full(50, g == "Female"), np.full(50, ag == AGE_LABELS[2])])
    ax[0].plot(xx, x0 @ final["beta"], color=c, lw=2)
ax[0].set(xlabel="Head size (cm³)", ylabel="Brain weight (g)",
          title=f"(a) Final ANCOVA model: parallel lines, R² = {final['R2']:.3f}")
ax[0].legend(fontsize=8)
ax[1].scatter(np.log10(df["head"]), np.log10(df.brain), s=12, alpha=0.5, color="0.3")
lx = np.linspace(np.log10(df["head"]).min(), np.log10(df["head"]).max(), 50)
ax[1].plot(lx, la.intercept + la.slope * lx, "k-", lw=2, label=f"fit: slope b = {la.slope:.2f} ({b_lo:.2f}–{b_hi:.2f})")
ax[1].plot(lx, la.intercept + la.slope * lx.mean() + (lx - lx.mean()), "g--", lw=1.5, label="b = 1 (brain ∝ head size)")
ax[1].set(xlabel="log$_{10}$ head size", ylabel="log$_{10}$ brain weight", title="(b) Allometric (log–log) fit")
ax[1].legend()
save(fig, "step5_ancova_and_allometry")
decision("unequal slopes are not needed (M3 -> M4 not significant) -> M3 (common slope + gender + age shifts) is the final model. "
         "Allometric b < 1: small heads naturally have a HIGHER brain/head ratio, so the simple ratio in step 4 is biased "
         "towards smaller-headed groups -> trust the regression adjustment, not the ratio")

# =============================================================================================
header("STEP 6 - Check the final model with residuals")
# =============================================================================================
res, S = final["res"], final["S"]
print(f"S = {S:.1f} g, RMSE = {final['RMSE']:.1f} g; Shapiro-Wilk p = {stats.shapiro(res).pvalue:.3f}; "
      f"|residual| > 2.5S: {np.sum(np.abs(res) > 2.5 * S)} people")
quad = ols(np.column_stack([df.head_c, df.head_c ** 2, df.fem, df.age2]), y, ["head_c", "head_c^2", "fem", "age2"])
print(f"Curvature check (adding head^2): p = {quad['table'].loc['head_c^2', 'p']:.3f}")
lev = stats.levene(*[res[((df.Gender == g) & (df.Age == ag)).to_numpy()] for g, ag in GROUPS])
print(f"Equal residual spread across groups (Levene): p = {lev.pvalue:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for (g, ag), (c, mk) in GROUPS.items():
    m = ((df.Gender == g) & (df.Age == ag)).to_numpy()
    ax[0].scatter(final["yhat"][m], res[m], s=14, alpha=0.6, color=c, marker=mk, label=f"{g}, {ag}")
ax[0].axhline(0, color="k", lw=1)
for k in (-2, 2):
    ax[0].axhline(k * S, color="0.5", ls=":", lw=1)
ax[0].set(xlabel="Predicted brain weight (g)", ylabel="Residual (g)", title="(a) Residuals vs prediction (dotted ±2S)")
ax[0].legend(fontsize=7)
data = [res[((df.Gender == g) & (df.Age == ag)).to_numpy()] for g, ag in GROUPS]
bp = ax[1].boxplot(data, patch_artist=True, widths=0.6, showmeans=True, meanprops=BOX_MEAN)
for patch, k in zip(bp["boxes"], GROUPS):
    patch.set_facecolor(GROUPS[k][0])
    patch.set_alpha(0.5)
ax[1].set_xticks(range(1, 5), [f"{g[0]}\n{ag}" for g, ag in GROUPS])
ax[1].axhline(0, color="k", lw=1)
ax[1].set(ylabel="Residual (g)", title=f"(b) Residuals by group (Levene p = {lev.pvalue:.2f})")
(osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
ax[2].scatter(osm, osr, s=10, color="0.3")
ax[2].plot(osm, sl * osm + ic, "r-", lw=1.5)
ax[2].set(xlabel="Theoretical normal quantiles", ylabel="Ordered residuals (g)",
          title=f"(c) Normal Q–Q plot (Shapiro p = {stats.shapiro(res).pvalue:.2f})")
save(fig, "step6_residual_check")
decision("no curvature, no group pattern, equal spread -> the ANCOVA model is adequate. Residuals are only mildly "
         "non-normal (a few large values); with n = 237 the regression estimates and CIs remain reliable")

# =============================================================================================
header("STEP 7 - Summarise: raw vs size-adjusted differences")
# =============================================================================================
tb = final["table"]
rawM = raw.loc["brain: Male - Female", "difference"]
rawA = raw.loc[f"brain: {AGE_LABELS[1]} - {AGE_LABELS[2]}", "difference"]
summary = pd.DataFrame({
    "raw difference (g)": [rawM, rawA],
    "size-adjusted difference (g)": [-tb.loc["fem", "coefficient"], -tb.loc["age2", "coefficient"]],
    "95% CI low": [-tb.loc["fem", "coefficient"] - 1.96 * tb.loc["fem", "SE"], -tb.loc["age2", "coefficient"] - 1.96 * tb.loc["age2", "SE"]],
    "95% CI high": [-tb.loc["fem", "coefficient"] + 1.96 * tb.loc["fem", "SE"], -tb.loc["age2", "coefficient"] + 1.96 * tb.loc["age2", "SE"]],
    "p (adjusted)": [tb.loc["fem", "p"], tb.loc["age2", "p"]]},
    index=["Male - Female", f"Age {AGE_LABELS[1]} - {AGE_LABELS[2]}"])
summary["% of raw gap explained by head size"] = 100 * (1 - summary["size-adjusted difference (g)"] / summary["raw difference (g)"])
print(summary.round(3).to_string())
print(f"Head size effect: +{100 * tb.loc['head_c', 'coefficient']:.1f} g brain per +100 cm^3 head "
      f"(95% CI {100 * (tb.loc['head_c', 'coefficient'] - 1.96 * tb.loc['head_c', 'SE']):.1f} to "
      f"{100 * (tb.loc['head_c', 'coefficient'] + 1.96 * tb.loc['head_c', 'SE']):.1f})")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
lab = list(summary.index)
xpos = np.arange(len(lab))
ax[0].bar(xpos - 0.2, summary["raw difference (g)"], 0.4, color="0.6", edgecolor="k", label="raw difference")
ax[0].bar(xpos + 0.2, summary["size-adjusted difference (g)"], 0.4, color="tab:blue", edgecolor="k",
          yerr=[summary["size-adjusted difference (g)"] - summary["95% CI low"],
                summary["95% CI high"] - summary["size-adjusted difference (g)"]],
          capsize=5, label="at equal head size (95 % CI)")
ax[0].axhline(0, color="k", lw=1)
ax[0].set_xticks(xpos, lab)
ax[0].set(ylabel="Difference in brain weight (g)", title="(a) Most of the raw gap is a head-size effect")
ax[0].legend()
names = list(adj.index)
ax[1].errorbar(adj["adjusted mean at mean head size (g)"], range(4),
               xerr=[adj["adjusted mean at mean head size (g)"] - adj["95% CI low"],
                     adj["95% CI high"] - adj["adjusted mean at mean head size (g)"]],
               fmt="o", color="tab:blue", capsize=4, label="adjusted mean (95 % CI)")
ax[1].scatter(adj["raw mean (g)"], range(4), marker="x", color="0.4", s=50, label="raw mean")
ax[1].set_yticks(range(4), names)
ax[1].set(xlabel="Brain weight (g)", title=f"(b) Group means at the same head size ({df['head'].mean():.0f} cm³)")
ax[1].legend(fontsize=8)
save(fig, "step7_raw_vs_adjusted")

with pd.ExcelWriter(f"{OUT_DIR}/ex5_step_results.xlsx") as w:
    raw.to_excel(w, sheet_name="3 Raw differences")
    slopes.to_excel(w, sheet_name="4 Group slopes & ratios")
    comp.to_excel(w, sheet_name="5 Model comparison")
    final["table"].to_excel(w, sheet_name="5 Final ANCOVA model")
    adj.to_excel(w, sheet_name="5 Adjusted means")
    summary.to_excel(w, sheet_name="7 Summary")
print(f"\nSaved figures and tables to ./{OUT_DIR}")
plt.show()
