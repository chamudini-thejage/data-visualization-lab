"""
Exercise 7 - Star classification and the Hertzsprung-Russell (HR) diagram, step by step
  Step 1  Understand the question and the variables
  Step 2  Data quality: clean colour labels, class balance, physical consistency checks
  Step 3  Look at each feature alone, per star type
  Step 4  Reproduce the HR diagram (question 1: do the stars follow it?)
  Step 5  Look at feature pairs (pair plot)
  Step 6  Categorical features (spectral class, colour) vs star type
  Step 7  Build classifiers: single features, pairs, all; interpretable decision tree rules
  Step 8  Validate honestly: cross-validation, confusion matrix, decision regions
  Step 9  Summarise
Run:  python ex7_stars_stepwise.py   (needs Exercise7_star_data.xlsx in the same folder)
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

DATA_FILE = "Exercise7_star_data.xlsx"
OUT_DIR = "ex7_step_outputs"
DEPTH = 4                       # maximum depth of the decision trees (kept small = readable rules)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})
pd.set_option("display.width", 200)
NAMES = ["Brown dwarf", "Red dwarf", "White dwarf", "Main sequence", "Supergiant", "Hypergiant"]
SHORT = ["BD", "RD", "WD", "MS", "SG", "HG"]
PAL = ["#8c564b", "#d62728", "#7f7f7f", "#ff7f0e", "#1f77b4", "#9467bd"]
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def header(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)


def decision(text):
    print(f"--> DECISION: {text}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")


def hr_axes(ax):
    """Apply HR-diagram conventions: log temperature decreasing to the right, bright stars at the top."""
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks([40000, 20000, 10000, 5000, 3000], ["40 000", "20 000", "10 000", "5 000", "3 000"])
    ax.minorticks_off()


# =============================================================================================
header("STEP 1 - Understand the question and the variables")
# =============================================================================================
df = pd.read_excel(DATA_FILE, sheet_name="Data")
df.columns = ["T", "L", "R", "Mv", "type", "color", "spec"]
df["Type"] = df.type.map(dict(enumerate(NAMES)))
for v in ["T", "L", "R"]:
    df[f"log{v}"] = np.log10(df[v])
print(f"{len(df)} stars; missing values: {int(df.isna().sum().sum())}")
print(df[["T", "L", "R", "Mv"]].describe().T[["min", "max"]].assign(
    **{"max/min": lambda t: t["max"] / t["min"].where(t["min"] > 0)}).to_string())
decision("target = star type (6 classes) -> CLASSIFICATION problem. T, L, R span many orders of magnitude "
         "-> use log10 T, log10 L, log10 R (Mv is already logarithmic)")

# =============================================================================================
header("STEP 2 - Data quality")
# =============================================================================================
print("Raw colour labels:", df.color.nunique(), "different spellings")
key = df.color.str.lower().str.replace(r"[\s\-]+", " ", regex=True).str.strip()
COLOR_MAP = {"red": "Red", "blue": "Blue", "blue white": "Blue-white", "white": "White", "whitish": "White",
             "yellow white": "Yellow-white", "yellowish white": "Yellow-white", "white yellow": "Yellow-white",
             "yellowish": "Yellow", "pale yellow orange": "Orange", "orange": "Orange", "orange red": "Orange"}
df["Color"] = key.map(COLOR_MAP)
print("Cleaned colours:", df.Color.value_counts().to_dict(), "| unmapped:", int(df.Color.isna().sum()))
print("Class balance:", df.Type.value_counts().reindex(NAMES).to_dict())
# physics check 1: Stefan-Boltzmann  L = R^2 (T/5778)^4
df["log_L_over_SB"] = df.logL - (2 * df.logR + 4 * np.log10(df["T"] / 5778))
print(f"Stefan-Boltzmann check log10(L / R²T⁴): median {df.log_L_over_SB.median():+.2f}, "
      f"IQR {df.log_L_over_SB.quantile(.25):+.2f} to {df.log_L_over_SB.quantile(.75):+.2f} (0 = perfect)")
print(df.groupby("Type").log_L_over_SB.median().reindex(NAMES).round(2).to_string())
# physics check 2: Mv = 4.83 - 2.5 log10 L
mv_fit = stats.linregress(df.logL, df.Mv)
print(f"Mv vs log10 L: slope {mv_fit.slope:.2f} (theory -2.5), intercept {mv_fit.intercept:.2f} (theory 4.83), r = {mv_fit.rvalue:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i, n in enumerate(NAMES):
    s = df[df.type == i]
    ax[0].scatter(s.logL, s.Mv, s=16, color=PAL[i], alpha=0.75, label=n)
    ax[1].scatter(2 * s.logR + 4 * np.log10(s["T"] / 5778), s.logL, s=16, color=PAL[i], alpha=0.75)
xx = np.linspace(df.logL.min(), df.logL.max(), 50)
ax[0].plot(xx, 4.83 - 2.5 * xx, "k--", lw=1.5, label="theory: Mv = 4.83 − 2.5 log L")
ax[0].invert_yaxis()
ax[0].set(xlabel="log$_{10}$ L (L$_\\odot$)", ylabel="Absolute magnitude Mv",
          title=f"(a) Magnitude–luminosity (fit slope {mv_fit.slope:.2f})")
ax[0].legend(fontsize=7)
lim = [df.logL.min() - 1, df.logL.max() + 1]
ax[1].plot(lim, lim, "k--", lw=1.5, label="Stefan–Boltzmann (perfect)")
ax[1].set(xlabel="log$_{10}$[R² (T/5778)⁴]  (predicted log L)", ylabel="log$_{10}$ L (given)",
          title="(b) Stefan–Boltzmann consistency")
ax[1].legend()
save(fig, "step2_physics_checks")
decision("classes are perfectly balanced (40 each) -> plain accuracy is a fair score. The data broadly follows "
         "known physics, with scatter from manually calculated values -> trustworthy enough to classify")

# =============================================================================================
header("STEP 3 - Each feature alone, per star type")
# =============================================================================================
FEATS = {"logT": "log$_{10}$ T (K)", "logL": "log$_{10}$ L (L$_\\odot$)", "logR": "log$_{10}$ R (R$_\\odot$)",
         "Mv": "Absolute magnitude Mv"}
rng_tab = df.groupby("Type")[["T", "L", "R", "Mv"]].agg(["min", "max"]).reindex(NAMES)
print(rng_tab.to_string(float_format=lambda v: f"{v:.3g}"))
fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
for a, (f, lab) in zip(ax, FEATS.items()):
    data = [df[df.type == i][f] for i in range(6)]
    bp = a.boxplot(data, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], PAL):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, d in enumerate(data):
        a.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(d)), d, s=6, color="k", alpha=0.4, zorder=3)
    a.set_xticks(range(1, 7), SHORT)
    if f == "Mv":
        a.invert_yaxis()
    a.set(title=lab, xlabel="Star type")
fig.suptitle("Feature distributions by type (BD brown dwarf, RD red dwarf, WD white dwarf, MS main sequence, "
             "SG supergiant, HG hypergiant)", fontsize=10)
save(fig, "step3_features_by_type")
decision("no single feature separates all six: temperature overlaps everywhere (SG/HG span 3 000-40 000 K), "
         "Mv mixes white and red dwarfs, radius separates most -> combine features (steps 4-5)")

# =============================================================================================
header("STEP 4 - Reproduce the HR diagram (question 1)")
# =============================================================================================
fig, ax = plt.subplots(figsize=(10, 7.5))
for i, n in enumerate(NAMES):
    s = df[df.type == i]
    ax.scatter(s["T"], s.Mv, s=np.clip(s.logR * 35 + 80, 12, 300), color=PAL[i], alpha=0.75,
               edgecolor="k", lw=0.4, label=n)
hr_axes(ax)
for txt, (x, y) in {"MAIN SEQUENCE": (9000, 0.5), "SUPERGIANTS / HYPERGIANTS": (12000, -9.5),
                    "WHITE DWARFS": (14000, 15.3), "RED & BROWN DWARFS": (3300, 9)}.items():
    ax.text(x, y, txt, fontsize=9, fontweight="bold", color="0.25", ha="center")
ax.set(xlabel="Surface temperature (K)  — hot ← → cool", ylabel="Absolute magnitude Mv  (bright ↑)",
       title="Hertzsprung–Russell diagram of the 240 stars (marker size ∝ log radius)")
ax.legend(loc="lower left")
save(fig, "step4_HR_diagram")
ms = df[df.type == 3]
r_ms = stats.pearsonr(ms.logT, ms.Mv)
print(f"Main sequence: Mv vs log T correlation r = {r_ms[0]:.3f} (a tight diagonal band)")
decision("the stars occupy separate, well-defined regions exactly as in the textbook HR diagram -> "
         "the data follows the HR diagram; the HR plane (T, Mv) should classify very well")

# =============================================================================================
header("STEP 5 - Feature pairs")
# =============================================================================================
fl = list(FEATS)
fig, axes = plt.subplots(4, 4, figsize=(13, 12))
for i, fy in enumerate(fl):
    for j, fx in enumerate(fl):
        a = axes[i, j]
        if i == j:
            for k in range(6):
                a.hist(df[df.type == k][fx], bins=12, color=PAL[k], alpha=0.6)
        else:
            for k in range(6):
                s = df[df.type == k]
                a.scatter(s[fx], s[fy], s=6, color=PAL[k], alpha=0.7)
        if i == 3:
            a.set_xlabel(FEATS[fx], fontsize=9)
        if j == 0:
            a.set_ylabel(FEATS[fy], fontsize=9)
        a.tick_params(labelsize=7)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for c, n in zip(PAL, NAMES)]
fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
save(fig, "step5_pair_plot")
print("Spearman correlation between features:")
print(df[fl].corr(method="spearman").round(2).to_string())
decision("log L and Mv are strongly linked (Pearson r = -0.98 in step 2: the same information on two scales); "
         "radius + Mv or T + Mv give the cleanest separation -> test these pairs in the classifier")

# =============================================================================================
header("STEP 6 - Categorical features vs star type")
# =============================================================================================
spec_order = ["O", "B", "A", "F", "G", "K", "M"]
col_order = ["Blue", "Blue-white", "White", "Yellow-white", "Yellow", "Orange", "Red"]
ct_spec = pd.crosstab(df.Type, df.spec).reindex(index=NAMES, columns=spec_order, fill_value=0)
ct_col = pd.crosstab(df.Type, df.Color).reindex(index=NAMES, columns=col_order, fill_value=0)
print(ct_spec.to_string())
print(ct_col.to_string())
cramer = {}
for nm, ct in [("spectral class", ct_spec), ("colour", ct_col)]:
    chi2 = stats.chi2_contingency(ct.loc[:, ct.sum() > 0])[0]
    cramer[nm] = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
spec_T = df.groupby("spec")["T"].median().reindex(spec_order)
print("Cramér's V with star type:", {k: round(float(v), 2) for k, v in cramer.items()})
print("Median temperature per spectral class (K):", {k: int(v) for k, v in spec_T.items()})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for a, ct, t in [(ax[0], ct_spec, f"(a) Spectral class vs type (Cramér's V = {cramer['spectral class']:.2f})"),
                 (ax[1], ct_col, f"(b) Colour vs type (Cramér's V = {cramer['colour']:.2f})")]:
    im = a.imshow(ct.values, cmap="Blues")
    a.set_xticks(range(ct.shape[1]), ct.columns, rotation=30, ha="right")
    a.set_yticks(range(6), NAMES)
    a.grid(False)
    for (r, c), v in np.ndenumerate(ct.values):
        if v:
            a.text(c, r, v, ha="center", va="center", fontsize=9, color="white" if v > 25 else "black")
    a.set_title(t, fontsize=11)
d = df.copy()
d["spec_i"] = d.spec.map({s: i for i, s in enumerate(spec_order)})
ax[2].scatter(d.spec_i + np.random.default_rng(1).normal(0, 0.08, len(d)), d["T"], s=10,
              c=[PAL[t] for t in d.type], alpha=0.7)
ax[2].set_xticks(range(7), spec_order)
ax[2].set_yscale("log")
ax[2].set(xlabel="Spectral class", ylabel="Temperature (K, log)", title="(c) Spectral class tracks temperature")
save(fig, "step6_categorical_features")
decision("spectral class and colour follow TEMPERATURE (O hot … M cool), not type: M/red stars are brown dwarfs, "
         "red dwarfs, supergiants AND hypergiants -> they cannot classify on their own; use the numeric features")

# =============================================================================================
header("STEP 7 - Build classifiers")
# =============================================================================================
PLAIN = {"logT": "log T", "logL": "log L", "logR": "log R", "Mv": "Mv"}
sets = {PLAIN[f]: [f] for f in fl}
for a_, b_ in itertools.combinations(fl, 2):
    sets[f"{PLAIN[a_]} + {PLAIN[b_]}"] = [a_, b_]
sets["all four"] = fl
y = df.type.to_numpy()
res = {}
for nm, f in sets.items():
    acc = cross_val_score(DecisionTreeClassifier(max_depth=DEPTH, random_state=0), df[f].values, y, cv=CV)
    res[nm] = {"features": len(f), "CV accuracy mean": acc.mean(), "CV accuracy SD": acc.std()}
res = pd.DataFrame(res).T.sort_values("CV accuracy mean", ascending=False)
print(f"Decision tree (max depth {DEPTH}), 5-fold cross-validated accuracy:")
print(res.round(3).to_string())
fig, ax = plt.subplots(figsize=(9, 6))
o = res.sort_values("CV accuracy mean")
ax.barh(o.index, o["CV accuracy mean"] * 100,
        xerr=o["CV accuracy SD"] * 100, color=["tab:blue" if n > 1 else "0.6" for n in o.features],
        edgecolor="k", capsize=3)
ax.axvline(100 / 6, color="tab:red", ls=":", label="random guessing (16.7 %)")
ax.set(xlabel="Cross-validated accuracy (%)", title="Which features classify stars? (grey = single feature)",
       xlim=(0, 105))
ax.legend()
save(fig, "step7_feature_set_comparison")
# simple interpretable rule set: radius + magnitude
rule_tree = DecisionTreeClassifier(max_depth=DEPTH, random_state=0).fit(df[["logR", "Mv"]].values, y)
print("\nReadable classification rules (tree on log10 R and Mv; class numbers = star type):")
print(export_text(rule_tree, feature_names=["log10_R", "Mv"]))
decision("one feature is not enough (best single: radius); any good PAIR reaches ~99-100 %; "
         "the HR-plane pair (T, Mv) is enough -> the HR diagram itself is a classifier")

# =============================================================================================
header("STEP 8 - Validate: confusion matrices and decision regions")
# =============================================================================================
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.2]})
for a, (nm, f) in zip(ax[:2], [("Mv only", ["Mv"]), ("log T + Mv (HR plane)", ["logT", "Mv"])]):
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=DEPTH, random_state=0), df[f].values, y, cv=CV)
    cm = confusion_matrix(y, pred)
    a.imshow(cm, cmap="Blues", vmin=0, vmax=40)
    for (r, c), v in np.ndenumerate(cm):
        if v:
            a.text(c, r, v, ha="center", va="center", color="white" if v > 25 else "black", fontsize=10)
    a.set_xticks(range(6), SHORT)
    a.set_yticks(range(6), SHORT)
    a.grid(False)
    a.set(xlabel="Predicted type", ylabel="True type", title=f"Confusion matrix: {nm}\naccuracy {np.mean(pred == y) * 100:.1f} %")
    off = [(NAMES[i], NAMES[j], cm[i, j]) for i in range(6) for j in range(6) if i != j and cm[i, j]]
    print(f"{nm}: accuracy {np.mean(pred == y) * 100:.1f} %; confusions (true -> predicted, count): "
          f"{[(t, p_, int(c)) for t, p_, c in off] if off else 'none'}")
# decision regions of the HR-plane tree
hr_tree = DecisionTreeClassifier(max_depth=DEPTH, random_state=0).fit(df[["logT", "Mv"]].values, y)
gx, gy = np.meshgrid(np.linspace(df.logT.min() - 0.05, df.logT.max() + 0.05, 400),
                     np.linspace(df.Mv.min() - 1, df.Mv.max() + 1, 400))
Zc = hr_tree.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
from matplotlib.colors import ListedColormap
a = ax[2]
a.contourf(10 ** gx, gy, Zc, levels=np.arange(-0.5, 6.5), cmap=ListedColormap(PAL), alpha=0.25)
for i, n in enumerate(NAMES):
    s = df[df.type == i]
    a.scatter(s["T"], s.Mv, s=14, color=PAL[i], edgecolor="k", lw=0.3, label=n)
hr_axes(a)
a.set(xlabel="Temperature (K)", ylabel="Mv", title="Decision regions in the HR plane")
a.legend(fontsize=7, loc="lower left")
save(fig, "step8_validation")
decision("Mv alone confuses white dwarfs with red dwarfs (same brightness); adding temperature removes it "
         "because white dwarfs are hot -> classification is essentially perfect in the HR plane")

# =============================================================================================
header("STEP 9 - Summary")
# =============================================================================================
best_single = res[res.features == 1].iloc[0]
print(f"1. The 240 stars fall into distinct regions of the HR diagram: main sequence diagonal, "
      f"giants on top, white dwarfs bottom-left, red/brown dwarfs bottom-right.")
print(f"2. Best single feature: {best_single.name} ({best_single['CV accuracy mean'] * 100:.1f} %); "
      f"HR plane (T + Mv): {res.loc['log T + Mv', 'CV accuracy mean'] * 100:.1f} %; "
      f"all four: {res.loc['all four', 'CV accuracy mean'] * 100:.1f} %.")
print("3. Spectral class and colour describe temperature, not type, so they cannot classify alone.")
print("4. Luminosity and magnitude carry the same information (Mv = 4.83 - 2.5 log L).")
print("Limitations: small dataset (40 per type), partly calculated values (Stefan-Boltzmann scatter), "
      "hand-assigned classes; very clean separation may not hold for real survey data.")
with pd.ExcelWriter(f"{OUT_DIR}/ex7_step_results.xlsx") as w:
    df.to_excel(w, sheet_name="Cleaned data", index=False)
    rng_tab.to_excel(w, sheet_name="Ranges per type")
    ct_spec.to_excel(w, sheet_name="Spectral class x type")
    ct_col.to_excel(w, sheet_name="Colour x type")
    res.to_excel(w, sheet_name="Classifier comparison")
print(f"\nSaved figures and tables to ./{OUT_DIR}")
plt.show()
