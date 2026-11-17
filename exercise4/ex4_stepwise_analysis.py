"""
Exercise 4 - Insurance charges, analysed step by step with the decision framework:
  Step 1  Understand the question and the variables (target, types, expectations)
  Step 2  Look at the target alone (shape of the charge distribution)
  Step 3  Screen every factor against the target, one at a time
  Step 4  Split by the strongest factor and look again (repeat until no structure is left)
  Step 5  Build the model to match what the plots showed
  Step 6  Check the model with residuals
  Step 7  Summarise the answer in plain units
Each step prints what it found and the decision it leads to.
Run:  python ex4_stepwise_analysis.py   (needs Exercise4_insurance_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = "Exercise4_insurance_data.xlsx"
OUT_DIR = "ex4_step_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})
pd.set_option("display.width", 200)
SMK = {0: ("Non-smoker", "tab:green"), 1: ("Smoker", "tab:red")}
GROUPS = {(0, 0): ("Non-smoker, BMI < 30", "tab:green"), (0, 1): ("Non-smoker, BMI ≥ 30", "tab:olive"),
          (1, 0): ("Smoker, BMI < 30", "tab:orange"), (1, 1): ("Smoker, BMI ≥ 30", "tab:red")}
BOX_MEAN = dict(marker="D", markerfacecolor="white", markeredgecolor="k", markersize=5)


def header(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


def decision(text):
    print(f"--> DECISION: {text}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")


def ols(X, y, names):
    """Least squares with SE, t, p, R^2, adjusted R^2, RMSE."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    res = y - yhat
    n, k = X.shape
    ss_res, ss_tot = np.sum(res ** 2), np.sum((y - y.mean()) ** 2)
    se = np.sqrt(np.diag(ss_res / (n - k) * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    r2 = 1 - ss_res / ss_tot
    return {"table": pd.DataFrame({"coefficient": beta, "SE": se, "t": beta / se, "p": p},
                                  index=["intercept"] + names),
            "R2": r2, "adjR2": 1 - (1 - r2) * (n - 1) / (n - k), "RMSE": np.sqrt(ss_res / n),
            "yhat": yhat, "res": res, "ss_res": ss_res, "k": k}


# =============================================================================================
header("STEP 1 - Understand the question and the variables")
# =============================================================================================
df = pd.read_excel(DATA_FILE, sheet_name="Data")
df.columns = ["age", "sex", "bmi", "smoker", "charges"]
df["smk"] = (df.smoker == "yes").astype(int)
df["fem"] = (df.sex == "female").astype(int)
df["k$"] = df.charges / 1000
y = df.charges.to_numpy(float)
print(f"{len(df)} people, missing values: {int(df.isna().sum().sum())}")
vtab = pd.DataFrame({
    "role": ["factor", "factor", "factor", "factor", "TARGET"],
    "type": ["continuous", "categorical (2)", "continuous", "categorical (2)", "continuous"],
    "range / levels": [f"{df.age.min()}-{df.age.max()} years", str(df.sex.value_counts().to_dict()),
                       f"{df.bmi.min():.1f}-{df.bmi.max():.1f} kg/m²", str(df.smoker.value_counts().to_dict()),
                       f"${df.charges.min():,.0f}-${df.charges.max():,.0f}"],
    "plot vs charges": ["scatter", "box plot", "scatter", "box plot", "histogram"],
    "expectation": ["older -> higher (more health risk)", "little effect", "higher BMI -> higher",
                    "smokers much higher", "-"]}, index=["age", "sex", "bmi", "smoker", "charges"])
print(vtab.to_string())
decision("target = charges; scatter plots for age/BMI, box plots for sex/smoker; "
         "groups are unequal (274 smokers vs 1064) -> use densities / means, not counts")

# =============================================================================================
header("STEP 2 - Look at the target alone")
# =============================================================================================
sk = stats.skew(df.charges)
print(f"mean ${df.charges.mean():,.0f}, median ${df.charges.median():,.0f}, skewness {sk:.2f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
bins = np.linspace(0, df["k$"].max() * 1.02, 45)
a1.hist(df["k$"], bins=bins, color="0.6", edgecolor="k", lw=0.4)
a1.axvline(df["k$"].mean(), color="k", lw=2, label=f"mean = ${df.charges.mean():,.0f}")
a1.axvline(df["k$"].median(), color="k", ls="--", lw=1.5, label=f"median = ${df.charges.median():,.0f}")
a1.set(xlabel="Charges (thousand $)", ylabel="Number of people", title=f"(a) Charges: skewness = {sk:.2f}")
a1.legend()
a2.hist(np.log10(df.charges), bins=40, color="0.6", edgecolor="k", lw=0.4)
a2.set(xlabel="log$_{10}$(charges in $)", ylabel="Number of people",
       title="(b) Log scale: still not one single peak")
save(fig, "step2_target_distribution")
decision("charges are strongly right-skewed with a long upper tail -> probably hidden groups. "
         "Find which factor separates them (step 3); report medians alongside means.")

# =============================================================================================
header("STEP 3 - Screen every factor against charges, one at a time")
# =============================================================================================
screen = {}
for v, lab in [("age", "Age"), ("bmi", "BMI"), ("smk", "Smoker"), ("fem", "Female")]:
    r, p = stats.pearsonr(df[v], df.charges)
    rho, _ = stats.spearmanr(df[v], df.charges)
    screen[lab] = {"Pearson r": r, "p": p, "Spearman rho": rho, "variance explained alone (r^2)": r ** 2}
screen = pd.DataFrame(screen).T.sort_values("variance explained alone (r^2)", ascending=False)
print(screen.round(4).to_string())

fig, ax = plt.subplots(1, 5, figsize=(21, 4.3), gridspec_kw={"width_ratios": [1.1, 1, 1, 0.6, 0.6]})
cm = df[["age", "bmi", "smk", "fem", "charges"]].corr()
im = ax[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
labs = ["Age", "BMI", "Smoker", "Female", "Charges"]
ax[0].set_xticks(range(5), labs, rotation=35)
ax[0].set_yticks(range(5), labs)
ax[0].grid(False)
for i in range(5):
    for j in range(5):
        ax[0].text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if abs(cm.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[0].set_title("(a) Correlation matrix")
for a, v, lab in [(ax[1], "age", "Age (years)"), (ax[2], "bmi", "BMI (kg/m²)")]:
    a.scatter(df[v], df["k$"], s=8, alpha=0.4, color="0.3")
    a.set(xlabel=lab, ylabel="Charges (thousand $)", title=f"({'bc'[v == 'bmi']}) Charges vs {lab.split(' ')[0]}")
for a, v, lev, t in [(ax[3], "smoker", ["no", "yes"], "(d) Smoker"), (ax[4], "sex", ["female", "male"], "(e) Sex")]:
    a.boxplot([df[df[v] == l]["k$"] for l in lev], widths=0.6, showmeans=True, meanprops=BOX_MEAN)
    a.set_xticks([1, 2], lev)
    a.set(title=t, ylabel="Charges (thousand $)")
save(fig, "step3_screening")
decision("smoking dominates (r = 0.79, 62 % of variance alone). Age shows BANDS in (b), BMI a fan in (c): "
         "structure that one variable cannot explain -> split by smoking and re-plot (step 4).")

# =============================================================================================
header("STEP 4 - Split by the strongest factor (smoking) and look again; repeat")
# =============================================================================================
# 4a: colour by smoking
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for s, (lab, c) in SMK.items():
    g = df[df.smk == s]
    ax[0].scatter(g.age, g["k$"], s=9, alpha=0.5, color=c, label=lab)
    ax[1].scatter(g.bmi, g["k$"], s=9, alpha=0.5, color=c, label=lab)
ax[0].set(xlabel="Age (years)", ylabel="Charges (thousand $)", title="(a) Age, coloured by smoking: smokers still split in 2 bands")
ax[1].set(xlabel="BMI (kg/m²)", ylabel="Charges (thousand $)", title="(b) BMI, coloured by smoking: smokers jump near BMI 30")
for a in ax:
    a.legend()
save(fig, "step4a_split_by_smoking")
print("4a: non-smokers form one band, smokers form TWO bands; in the BMI plot the smoker cloud jumps upward near BMI 30.")

# find where the jump is: test every BMI threshold for smokers
sm = df[df.smk == 1]
thr = np.arange(25, 35.01, 0.5)
gaps = [sm[sm.bmi >= t].charges.mean() - sm[sm.bmi < t].charges.mean() for t in thr]
best = thr[int(np.argmax(gaps))]
print(f"Threshold scan (smokers): biggest jump in mean charges at BMI = {best:.1f} "
      f"(gap ${max(gaps):,.0f}); WHO obesity cut-off is 30.")
df["obese"] = (df.bmi >= 30).astype(int)
decision("the jump is at the WHO obesity limit (BMI 30) -> split smokers by obese / not obese and re-plot")

# 4b: four groups
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
slopes = {}
for k, (lab, c) in GROUPS.items():
    g = df[(df.smk == k[0]) & (df.obese == k[1])]
    l = stats.linregress(g.age, g.charges)
    slopes[lab] = {"n": len(g), "slope ($/year)": l.slope, "SE": l.stderr, "intercept ($)": l.intercept,
                   "r": l.rvalue}
    ax[0].scatter(g.age, g["k$"], s=9, alpha=0.5, color=c, label=lab)
    xx = np.array([18, 64])
    ax[0].plot(xx, (l.intercept + l.slope * xx) / 1000, color=c, lw=2.2)
ax[0].set(xlabel="Age (years)", ylabel="Charges (thousand $)", title="(a) Three parallel bands explained")
ax[0].legend(fontsize=8)
edges = np.arange(15, 55, 2.5)
cen = (edges[:-1] + edges[1:]) / 2
for s, (lab, c) in SMK.items():
    g = df[df.smk == s]
    ax[1].scatter(g.bmi, g["k$"], s=8, alpha=0.35, color=c)
    m = [g[(g.bmi >= lo) & (g.bmi < hi)]["k$"].mean() if ((g.bmi >= lo) & (g.bmi < hi)).sum() >= 3 else np.nan
         for lo, hi in zip(edges[:-1], edges[1:])]
    ax[1].plot(cen, m, color=c, lw=2.2, marker="o", ms=4, mec="k", label=f"{lab}: mean per BMI bin")
ax[1].axvline(30, color="k", ls="--", lw=1)
ax[1].set(xlabel="BMI (kg/m²)", ylabel="Charges (thousand $)", title="(b) BMI matters only for smokers")
ax[1].legend(fontsize=8)
data = [df[(df.smk == k[0]) & (df.obese == k[1])]["k$"] for k in GROUPS]
bp = ax[2].boxplot(data, patch_artist=True, widths=0.6, showmeans=True, meanprops=BOX_MEAN)
for patch, (_, c) in zip(bp["boxes"], GROUPS.values()):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax[2].set_xticks(range(1, 5), ["NS\nBMI<30", "NS\nBMI≥30", "S\nBMI<30", "S\nBMI≥30"])
ax[2].set(ylabel="Charges (thousand $)", title="(c) The four groups")
save(fig, "step4b_split_by_smoking_and_obesity")
slopes = pd.DataFrame(slopes).T
print(slopes.round(2).to_string())
t_ob_ns = stats.ttest_ind(df[(df.smk == 0) & (df.obese == 1)].charges, df[(df.smk == 0) & (df.obese == 0)].charges,
                          equal_var=False)
t_ob_s = stats.ttest_ind(df[(df.smk == 1) & (df.obese == 1)].charges, df[(df.smk == 1) & (df.obese == 0)].charges,
                         equal_var=False)
print(f"Obese vs not: non-smokers +${df[(df.smk==0)&(df.obese==1)].charges.mean()-df[(df.smk==0)&(df.obese==0)].charges.mean():,.0f} "
      f"(p = {t_ob_ns.pvalue:.3f});  smokers +${df[(df.smk==1)&(df.obese==1)].charges.mean()-df[(df.smk==1)&(df.obese==0)].charges.mean():,.0f} "
      f"(p = {t_ob_s.pvalue:.1e})")
decision("age slopes are equal in all groups (~$260-280/yr) -> one age term; smoking shifts the level -> smoker term; "
         "obesity matters only with smoking -> smoker x obese INTERACTION term")

# 4c: check for a confounder - sex
print("\n4c: is the sex difference real or due to another variable?")
raw = df[df.fem == 0].charges.mean() - df[df.fem == 1].charges.mean()
print(f"raw male - female difference: ${raw:,.0f} (Welch p = "
      f"{stats.ttest_ind(df[df.fem==0].charges, df[df.fem==1].charges, equal_var=False).pvalue:.3f})")
print("smoking rate:", (df.groupby("sex").smk.mean() * 100).round(1).to_dict(), "%")
within = {lab: df[(df.smk == s) & (df.fem == 0)].charges.mean() - df[(df.smk == s) & (df.fem == 1)].charges.mean()
          for s, (lab, _) in SMK.items()}
print("male - female difference WITHIN smoking groups:", {k: f"${v:,.0f}" for k, v in within.items()})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].bar(["female", "male"], df.groupby("sex").smk.mean().reindex(["female", "male"]) * 100,
          color=["tab:purple", "tab:blue"], edgecolor="k")
ax[0].set(ylabel="Smokers (%)", title="(a) More men smoke")
pos = 0
for s, (lab, c) in SMK.items():
    for j, sx in enumerate(["female", "male"]):
        ax[1].boxplot(df[(df.smk == s) & (df.sex == sx)]["k$"], positions=[pos], widths=0.6, showmeans=True,
                      meanprops=BOX_MEAN, patch_artist=True,
                      boxprops=dict(facecolor=["tab:purple", "tab:blue"][j], alpha=0.5))
        pos += 1
    pos += 0.6
ax[1].set_xticks([0, 1, 2.6, 3.6], ["F\nnon-smoker", "M\nnon-smoker", "F\nsmoker", "M\nsmoker"])
ax[1].set(ylabel="Charges (thousand $)", title="(b) Within each smoking group, sexes are similar")
save(fig, "step4c_sex_confounding_check")
decision("the raw sex difference is largely confounded with smoking (more men smoke) -> keep sex in the model to test it after adjusting")

# =============================================================================================
header("STEP 5 - Build the model to match what the plots showed")
# =============================================================================================
df["smk_x_obese"] = df.smk * df.obese
models = {"M1: smoker only": ["smk"],
          "M2: + age": ["smk", "age"],
          "M3: + BMI + sex (naive additive)": ["smk", "age", "bmi", "fem"],
          "M4: + smoker x obese (from plots)": ["smk", "age", "bmi", "fem", "smk_x_obese"]}
fits = {m: ols(df[c].to_numpy(float), y, c) for m, c in models.items()}
comp = pd.DataFrame({m: {"terms": len(c), "R^2": f["R2"], "adjusted R^2": f["adjR2"], "RMSE ($)": f["RMSE"]}
                     for (m, c), f in zip(models.items(), fits.values())}).T
print(comp.to_string())
f3, f4 = fits["M3: + BMI + sex (naive additive)"], fits["M4: + smoker x obese (from plots)"]
F = (f3["ss_res"] - f4["ss_res"]) / (f4["ss_res"] / (len(y) - f4["k"]))
print(f"F-test M3 -> M4 (interaction): F = {F:.0f}, p = {stats.f.sf(F, 1, len(y) - f4['k']):.2g}")
print("\nFinal model M4 coefficients:")
print(f4["table"].round(4).to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
steps = ["smoker", "+ age", "+ BMI\n+ sex", "+ smoker\n× obese"]
for a, col, scale, c, lab in [(ax[0], "adjusted R^2", 1, "tab:blue", "Adjusted R²"),
                              (ax[1], "RMSE ($)", 1000, "tab:red", "RMSE (thousand $)")]:
    vals = comp[col].to_numpy(float) / scale
    a.plot(range(len(comp)), vals, "o-", color=c, lw=2, ms=8)
    for i, v in enumerate(vals):
        a.annotate(f"{v:.3f}" if scale == 1 else f"{v:.2f}", (i, v), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=9)
    a.set_xticks(range(len(comp)), steps)
    a.set(ylabel=lab, xlabel="Terms added to the model")
ax[0].set_title("(a) Variance explained (higher is better)")
ax[1].set_title("(b) Typical prediction error (lower is better)")
save(fig, "step5_model_building")
decision("adding the interaction found in step 4 gives the biggest single improvement after smoking itself")

# =============================================================================================
header("STEP 6 - Check the models with residuals")
# =============================================================================================
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
lim = [0, df["k$"].max() * 1.05]
for j, (m, f) in enumerate([("M3 naive additive", f3), ("M4 with interaction", f4)]):
    for s, (lab, c) in SMK.items():
        msk = (df.smk == s).to_numpy()
        ax[0, j].scatter(f["yhat"][msk] / 1000, y[msk] / 1000, s=8, alpha=0.5, color=c, label=lab)
        ax[1, j].scatter(f["yhat"][msk] / 1000, f["res"][msk] / 1000, s=8, alpha=0.5, color=c)
    ax[0, j].plot(lim, lim, "k--", lw=1, label="perfect prediction")
    ax[0, j].set(xlim=lim, ylim=lim, xlabel="Predicted (thousand $)", ylabel="Actual (thousand $)",
                 title=f"{m}: R² = {f['R2']:.3f}, RMSE = \\${f['RMSE']:,.0f}")
    ax[0, j].legend(fontsize=8)
    ax[1, j].axhline(0, color="k", lw=1)
    ax[1, j].set(xlabel="Predicted (thousand $)", ylabel="Residual (thousand $)", title="Residuals")
save(fig, "step6_residual_check")
r4 = f4["res"]
big = r4 > 5000
print(f"M3: smoker residuals form two bands at about ±$10 000 -> structure left -> model incomplete")
print(f"M4: {np.mean(np.abs(r4) < 5000) * 100:.0f} % of residuals within ±$5 000; "
      f"{big.sum()} people ({big.mean() * 100:.0f} %) have residual > +$5 000, "
      f"{(big & (df.smk == 0).to_numpy()).sum()} of them non-smokers")
print("Those high-residual people do not differ much in age/BMI/sex:")
print(df.assign(high=big).groupby("high")[["age", "bmi", "fem", "smk"]].mean().round(2).to_string())
decision("M4 residuals are flat for most people; the remaining cloud of high charges cannot be explained "
         "by age, sex, BMI or smoking -> a factor NOT in the data (e.g. medical history) -> state as limitation")

# =============================================================================================
header("STEP 7 - Summarise the answer")
# =============================================================================================
tb = f4["table"]
eff = [("Smoker (BMI < 30)", tb.loc["smk", "coefficient"], tb.loc["smk", "SE"]),
       ("Extra if smoker AND obese", tb.loc["smk_x_obese", "coefficient"], tb.loc["smk_x_obese", "SE"]),
       ("Age +10 years", 10 * tb.loc["age", "coefficient"], 10 * tb.loc["age", "SE"]),
       ("BMI +5 units", 5 * tb.loc["bmi", "coefficient"], 5 * tb.loc["bmi", "SE"]),
       ("Female vs male", tb.loc["fem", "coefficient"], tb.loc["fem", "SE"])]
for name, v, se in eff:
    print(f"{name:28s} {v:+10,.0f} $   (95 % CI {v - 1.96 * se:+,.0f} to {v + 1.96 * se:+,.0f})")
fig, ax = plt.subplots(figsize=(8.5, 4.2))
names = [e[0] for e in eff][::-1]
vals = [e[1] / 1000 for e in eff][::-1]
ci = [1.96 * e[2] / 1000 for e in eff][::-1]
ax.barh(names, vals, xerr=ci, color="tab:blue", edgecolor="k", capsize=4)
for i, (v, c) in enumerate(zip(vals, ci)):
    ax.text(v + c + 0.3, i, f"+${v * 1000:,.0f}", va="center", fontsize=9)
ax.axvline(0, color="k", lw=1)
ax.set_xlim(right=max(v + c for v, c in zip(vals, ci)) + 3)
ax.set(xlabel="Change in yearly charges (thousand $), 95 % CI", title="What drives insurance charges (final model)")
save(fig, "step7_effect_sizes")

with pd.ExcelWriter(f"{OUT_DIR}/ex4_step_results.xlsx") as w:
    vtab.to_excel(w, sheet_name="1 Variables")
    screen.to_excel(w, sheet_name="3 Screening")
    slopes.to_excel(w, sheet_name="4 Age slopes per group")
    comp.to_excel(w, sheet_name="5 Model comparison")
    f4["table"].to_excel(w, sheet_name="5 Final model")
print(f"\nSaved figures and tables to ./{OUT_DIR}")
plt.show()
