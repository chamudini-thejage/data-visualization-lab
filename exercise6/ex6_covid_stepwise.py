"""
Exercise 6 - COVID-19 in Asia: which countries are doing well and which need immediate attention?
A multi-criteria ranking problem, analysed step by step:
  Step 1  Understand the question; define what "doing well" means
  Step 2  Check data quality (missing values, consistency, very small numbers)
  Step 3  Derive comparison metrics (rates, not totals)
  Step 4  Look at each metric alone
  Step 5  Look at how the metrics relate (correlations, testing vs deaths quadrant plot)
  Step 6  Combine the metrics into one ranking (composite score)
  Step 7  Test whether the ranking is robust (sensitivity analysis)
  Step 8  Classify the countries into groups (k-means clustering + heatmap)
  Step 9  Summarise the answer and the limitations
Run:  python ex6_covid_stepwise.py   (needs Exercise6_covid_asia_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_FILE = "Exercise6_covid_asia_data.xlsx"
OUT_DIR = "ex6_step_outputs"
MIN_CASES = 1000          # countries with fewer cases give unstable rates
N_EXTREME = 10            # size of the "needs attention" / "doing well" lists
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})
pd.set_option("display.width", 220)


def header(text):
    print("\n" + "=" * 95 + f"\n{text}\n" + "=" * 95)


def decision(text):
    print(f"--> DECISION: {text}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")


# =============================================================================================
header("STEP 1 - Understand the question and define 'doing well'")
# =============================================================================================
df = pd.read_excel(DATA_FILE, sheet_name="Data")
print(f"{len(df)} countries/territories, {df.shape[1]} columns")
# the four criteria and their direction (+1: higher value = WORSE, -1: higher value = BETTER)
CRITERIA = {"TotalDeathsPerMillion": ("Deaths per million", +1, "outcome: how many people died"),
            "CFR": ("Case fatality rate (%)", +1, "outcome + care: share of detected cases who died"),
            "TotalTestsPerMillion": ("Tests per million", -1, "detection: how hard the country looks for cases"),
            "ActivePct": ("Active cases (%)", +1, "current situation: share of cases still ongoing")}
for k, (lab, s, why) in CRITERIA.items():
    print(f"  {lab:25s} {'higher = worse' if s > 0 else 'higher = better':16s} {why}")
decision("there is no target column, so 'doing well' = low deaths, low fatality, strong testing, few active cases. "
         "Raw totals mostly reflect population size -> compare countries only with rates and ratios.")

# =============================================================================================
header("STEP 2 - Check data quality")
# =============================================================================================
miss = df[df.isna().any(axis=1)]
for _, r in miss.iterrows():
    print(f"Missing: {r.Country}: {', '.join(c for c in df.columns if pd.isna(r[c]))}")
chk = df.TotalDeaths.fillna(0) + df.TotalRecovered + df.ActiveCases - df.TotalCases
print(f"Cases = deaths + recovered + active: {np.sum(chk.abs() <= 1)} of {len(df)} rows consistent")
pm = df.TotalCases / df.TotalPopulation * 1e6
print(f"Cases per million recomputed from totals: max deviation {np.max(np.abs(pm - df.TotalCasesPerMillion) / df.TotalCasesPerMillion) * 100:.1f} %")
small = df[df.TotalCases < MIN_CASES]
print(f"Fewer than {MIN_CASES} cases (rates unstable): {small.Country.tolist()}")
df["ranked"] = df[["TotalDeaths", "TotalTests"]].notna().all(axis=1) & (df.TotalCases >= MIN_CASES)
decision(f"data is internally consistent; exclude {df.loc[~df.ranked, 'Country'].tolist()} from the ranking "
         "(missing criteria or too few cases) but keep them in plots where possible")

# =============================================================================================
header("STEP 3 - Derive comparison metrics")
# =============================================================================================
df["CFR"] = 100 * df.TotalDeaths / df.TotalCases                 # case fatality rate (%)
df["ActivePct"] = 100 * df.ActiveCases / df.TotalCases            # share still active (%)
df["Positivity"] = 100 * df.TotalCases / df.TotalTests            # cases per test (%), high = under-testing
df["RecoveredPct"] = 100 * df.TotalRecovered / df.TotalCases
df["TestsPerCase"] = df.TotalTests / df.TotalCases
METRICS = {"TotalCasesPerMillion": "Cases per million", "TotalDeathsPerMillion": "Deaths per million",
           "CFR": "Case fatality rate (%)", "TotalTestsPerMillion": "Tests per million",
           "Positivity": "Positivity (%)", "ActivePct": "Active cases (%)"}
print(df[["Country"] + list(METRICS)].describe().round(2).to_string())
decision("metrics span several orders of magnitude -> analyse them on LOG scales")

# =============================================================================================
header("STEP 4 - Look at each metric alone")
# =============================================================================================
fig, axes = plt.subplots(1, 6, figsize=(24, 11), sharey=False)
for a, (col, lab) in zip(axes, METRICS.items()):
    d = df.dropna(subset=[col]).sort_values(col)
    vals = d[col].clip(lower=0.01)
    a.barh(d.Country, vals, color=plt.cm.viridis(np.linspace(0.15, 0.9, len(d))), edgecolor="k", lw=0.3)
    a.set_xscale("log")
    a.axvline(np.median(vals), color="tab:red", ls="--", lw=1.2, label=f"median {np.median(vals):.3g}")
    a.tick_params(axis="y", labelsize=7)
    a.set(title=lab, xlabel=f"{lab} (log scale)")
    a.set_ylim(-0.6, len(d) - 0.4)
    a.legend(fontsize=8, loc="lower right")
    print(f"{lab:25s} lowest: {d.Country.iloc[0]} ({d[col].iloc[0]:.3g}), highest: {d.Country.iloc[-1]} ({d[col].iloc[-1]:.3g}), "
          f"max/min = {d[col].max() / max(d[col].min(), 0.01):.0f}x")
save(fig, "step4_each_metric_sorted")
decision("every metric has a few extreme countries (e.g. Yemen CFR, Syria testing); no single metric tells the "
         "whole story -> look at how they relate (step 5)")

# =============================================================================================
header("STEP 5 - How the metrics relate")
# =============================================================================================
logm = np.log10(df[list(METRICS)].clip(lower=0.01))
sc = logm.corr(method="spearman")
print("Spearman correlations (log metrics):")
print(sc.round(2).to_string())
fig, ax = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1, 1.6]})
labs = list(METRICS.values())
im = ax[0].imshow(sc, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(labs)), labs, rotation=40, ha="right")
ax[0].set_yticks(range(len(labs)), labs)
ax[0].grid(False)
for i in range(len(labs)):
    for j in range(len(labs)):
        ax[0].text(j, i, f"{sc.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if abs(sc.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax[0], fraction=0.046, label="Spearman ρ")
ax[0].set_title("(a) Correlation between metrics")
d = df.dropna(subset=["TotalTestsPerMillion", "TotalDeathsPerMillion"])
s = ax[1].scatter(d.TotalTestsPerMillion, d.TotalDeathsPerMillion, s=np.sqrt(d.TotalPopulation) / 50,
                  c=d.CFR, cmap="RdYlGn_r", vmin=0, vmax=6, edgecolor="k", lw=0.5, alpha=0.85)
for _, r in d.iterrows():
    ax[1].annotate(r.Country, (r.TotalTestsPerMillion, r.TotalDeathsPerMillion), fontsize=7,
                   xytext=(4, 3), textcoords="offset points")
mx, my = d.TotalTestsPerMillion.median(), d.TotalDeathsPerMillion.median()
ax[1].axvline(mx, color="0.4", ls=":")
ax[1].axhline(my, color="0.4", ls=":")
ax[1].set(xscale="log", yscale="log", xlabel="Tests per million (log)", ylabel="Deaths per million (log)",
          title="(b) Testing vs deaths (bubble area ∝ population, colour = CFR; dotted = medians)")
kw = dict(transform=ax[1].transAxes, fontsize=9, fontweight="bold")
ax[1].text(0.01, 0.98, "LOW testing, HIGH deaths", va="top", color="darkred", **kw)
ax[1].text(0.99, 0.98, "HIGH testing, HIGH deaths\n(spread problem)", va="top", ha="right", color="darkorange", **kw)
ax[1].text(0.01, 0.02, "LOW testing, LOW deaths\n(possibly under-counted)", color="purple", **kw)
ax[1].text(0.99, 0.02, "HIGH testing, LOW deaths\n(doing well)", ha="right", color="darkgreen", **kw)
fig.colorbar(s, ax=ax[1], label="Case fatality rate (%)")
save(fig, "step5_relations_and_quadrants")
df["Quadrant"] = np.where(df.TotalTestsPerMillion >= mx,
                          np.where(df.TotalDeathsPerMillion >= my, "high testing / high deaths", "high testing / low deaths"),
                          np.where(df.TotalDeathsPerMillion >= my, "low testing / high deaths", "low testing / low deaths"))
df.loc[df[["TotalTestsPerMillion", "TotalDeathsPerMillion"]].isna().any(axis=1), "Quadrant"] = "not classified"
for q, g in df.groupby("Quadrant"):
    print(f"{q:28s}: {', '.join(g.Country)}")
r_t_cfr = stats.spearmanr(logm.TotalTestsPerMillion, logm.CFR, nan_policy="omit")
decision(f"more testing goes with LOWER CFR (ρ = {r_t_cfr.statistic:.2f}): countries that test little only find severe "
         "cases. Metrics capture different aspects -> combine them into one score (step 6)")

# =============================================================================================
header("STEP 6 - Combine the metrics into one ranking")
# =============================================================================================
R = df[df.ranked].copy().reset_index(drop=True)
Z = pd.DataFrame({c: s * stats.zscore(np.log10(R[c].clip(lower=0.01))) for c, (_, s, _) in CRITERIA.items()})
Z.index = R.Country
R["Score_z"] = Z.mean(axis=1).values                                   # method A: mean z-score (higher = worse)
RK = pd.DataFrame({c: (s * R[c]).rank() for c, (_, s, _) in CRITERIA.items()})
R["Score_rank"] = RK.mean(axis=1).values                               # method B: mean rank (higher = worse)
R["Rank_z"] = R.Score_z.rank(ascending=False).astype(int)              # 1 = needs most attention
R["Rank_r"] = R.Score_rank.rank(ascending=False, method="min").astype(int)
rho_methods = stats.spearmanr(R.Score_z, R.Score_rank).statistic
print(f"Agreement between z-score and rank methods: Spearman ρ = {rho_methods:.3f}")
print(R.sort_values("Rank_z")[["Country", "Rank_z", "Rank_r", "Score_z"]].round(2).to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 12))
o = R.sort_values("Score_z")
ax.barh(o.Country, o.Score_z, color=plt.cm.RdYlGn_r((o.Score_z - o.Score_z.min()) / (o.Score_z.max() - o.Score_z.min())),
        edgecolor="k", lw=0.3)
ax.axvline(0, color="k", lw=1)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylim(-0.6, len(o) - 0.4)
ax.set(xlabel="Composite score = mean z of log metrics (positive = worse than Asian average)",
       title="Composite ranking: top = needs most attention")
save(fig, "step6_composite_ranking")
decision("the two methods agree closely; next check whether the result depends on the choices made (step 7)")

# =============================================================================================
header("STEP 7 - Sensitivity analysis: is the ranking robust?")
# =============================================================================================
scen = {"z-score, equal weights": Z.mean(axis=1), "rank average": pd.Series(RK.mean(axis=1).values, index=Z.index)}
for c, (lab, _, _) in CRITERIA.items():
    scen[f"without {lab}"] = Z.drop(columns=c).mean(axis=1)
w_out = {"TotalDeathsPerMillion": 2, "CFR": 2, "TotalTestsPerMillion": 1, "ActivePct": 1}
scen["outcomes x2 weight"] = (Z * pd.Series(w_out)).sum(axis=1) / sum(w_out.values())
w_det = {"TotalDeathsPerMillion": 1, "CFR": 1, "TotalTestsPerMillion": 2, "ActivePct": 1}
scen["testing x2 weight"] = (Z * pd.Series(w_det)).sum(axis=1) / sum(w_det.values())
ranks = pd.DataFrame({k: v.rank(ascending=False) for k, v in scen.items()})   # 1 = worst
summ = pd.DataFrame({"median rank": ranks.median(axis=1), "best rank": ranks.min(axis=1), "worst rank": ranks.max(axis=1),
                     f"% scenarios in worst {N_EXTREME}": 100 * (ranks <= N_EXTREME).mean(axis=1),
                     f"% scenarios in best {N_EXTREME}": 100 * (ranks > len(ranks) - N_EXTREME).mean(axis=1)})
summ = summ.sort_values("median rank")
print(f"{len(scen)} scenarios: {list(scen)}")
print(summ.round(0).to_string())
robust_bad = summ[summ[f"% scenarios in worst {N_EXTREME}"] >= 75].index.tolist()
robust_good = summ[summ[f"% scenarios in best {N_EXTREME}"] >= 75].index.tolist()
print(f"\nROBUST 'needs attention' (worst {N_EXTREME} in >= 75 % of scenarios): {robust_bad}")
print(f"ROBUST 'doing well'      (best {N_EXTREME} in >= 75 % of scenarios): {robust_good}")
fig, ax = plt.subplots(figsize=(9, 12))
yy = np.arange(len(summ))[::-1]
col = ["tab:red" if c in robust_bad else "tab:green" if c in robust_good else "0.5" for c in summ.index]
ax.hlines(yy, summ["best rank"], summ["worst rank"], color=col, lw=3, alpha=0.6)
ax.scatter(summ["median rank"], yy, color=col, edgecolor="k", zorder=3, s=30)
ax.set_yticks(yy, summ.index, fontsize=8)
ax.axvline(N_EXTREME + 0.5, color="tab:red", ls=":", lw=1)
ax.axvline(len(summ) - N_EXTREME + 0.5, color="tab:green", ls=":", lw=1)
ax.set_ylim(-0.8, len(summ) - 0.2)
ax.set(xlabel="Rank across scenarios (1 = needs most attention); dot = median, bar = best–worst",
       title=f"Robustness of the ranking over {len(scen)} scoring scenarios\n(red/green = robust; grey = depends on method)")
save(fig, "step7_sensitivity")
decision("countries that keep their position under every method are firm conclusions; the rest are reported as borderline")

# =============================================================================================
header("STEP 8 - Classify the countries into groups (k-means on standardised log metrics)")
# =============================================================================================
X = Z.values
sil = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=0).fit_predict(X)) for k in range(2, 7)}
print("Silhouette score per number of clusters (higher = better separated):", {k: round(v, 3) for k, v in sil.items()})
K = 4
km = KMeans(K, n_init=50, random_state=0).fit(X)
cl = pd.Series(km.labels_, index=Z.index)
prof = Z.groupby(cl).mean()
prof["n"] = cl.value_counts().sort_index()
prof["mean score"] = Z.mean(axis=1).groupby(cl).mean()
prof = prof.sort_values("mean score", ascending=False)


PHRASES = {"TotalDeathsPerMillion": ("many deaths", "few deaths"), "CFR": ("high fatality", "low fatality"),
           "TotalTestsPerMillion": ("little testing", "heavy testing"), "ActivePct": ("many active cases", "few active cases")}


def describe_cluster(row, cut=0.4):
    """Name a cluster from its average z profile: list the criteria where it is clearly worse/better than average."""
    parts = [PHRASES[c][0] if row[c] > cut else PHRASES[c][1] for c in PHRASES if abs(row[c]) > cut]
    return ", ".join(parts) if parts else "close to average"


names = {c: f"G{i + 1}: {describe_cluster(r)}" for i, (c, r) in enumerate(prof.iterrows())}
R["Cluster"] = cl.map(names).values
print(f"k = {K} (chosen for interpretability; silhouette {sil[K]:.2f}). Cluster profiles (mean z, positive = worse):")
print(prof.rename(index=names).round(2).to_string())
for n, g in R.groupby("Cluster"):
    print(f"{n}: {', '.join(g.sort_values('Score_z', ascending=False).Country)}")
H = Z.copy()
H["cluster"] = cl.map(names)
H["score"] = Z.mean(axis=1)
H = H.sort_values(["cluster", "score"], ascending=[True, False])
fig, ax = plt.subplots(figsize=(10, 14))
im = ax.imshow(H[list(CRITERIA)].values, cmap="RdYlGn_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(CRITERIA)), [v[0] + ("\n(reversed)" if v[1] < 0 else "") for v in CRITERIA.values()],
              rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(H)), H.index, fontsize=7)
ax.grid(False)
bounds = np.where(H.cluster.values[1:] != H.cluster.values[:-1])[0]
for b in bounds:
    ax.axhline(b + 0.5, color="k", lw=2)
starts = np.r_[0, bounds + 1]
for st, name in zip(starts, H.cluster.values[starts]):
    ax.text(len(CRITERIA) - 0.4, st - 0.1, name, fontsize=8, fontweight="bold", va="top", ha="left")
fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.09,
             label="z-score (positive = worse than average)")
ax.set_title("Countries × criteria, grouped by cluster\n(red = problem, green = good)")
save(fig, "step8_cluster_heatmap")
decision("clusters group countries by the PATTERN of their problems (not by severity): the heatmap shows WHY each "
         "country ranks where it does; the composite score (step 6-7) gives HOW serious it is")

# =============================================================================================
header("STEP 9 - Summary")
# =============================================================================================
worst_metric = Z.idxmax(axis=1).map(lambda c: CRITERIA[c][0])
best_metric = Z.idxmin(axis=1).map(lambda c: CRITERIA[c][0])
show = ["Country", "TotalDeathsPerMillion", "CFR", "TotalTestsPerMillion", "Positivity", "ActivePct", "Rank_z", "Cluster"]
att = R[R.Country.isin(robust_bad)].sort_values("Rank_z")[show].copy()
att["main problem"] = att.Country.map(worst_metric)
good = R[R.Country.isin(robust_good)].sort_values("Rank_z", ascending=False)[show].copy()
good["main strength"] = good.Country.map(best_metric)
print("NEEDS IMMEDIATE ATTENTION (robust):")
print(att.round(2).to_string(index=False))
print("\nDOING RELATIVELY WELL (robust):")
print(good.round(2).to_string(index=False))
print("\nLIMITATIONS: single cumulative snapshot with no date; reporting quality and testing policy differ between "
      "countries; CFR depends on population age structure; low testing can hide both cases and deaths; "
      "the ranking depends on which criteria define 'doing well' (tested in step 7); correlation is not causation.")
with pd.ExcelWriter(f"{OUT_DIR}/ex6_step_results.xlsx") as w:
    df.to_excel(w, sheet_name="All metrics", index=False)
    R.sort_values("Rank_z").to_excel(w, sheet_name="Ranking", index=False)
    summ.to_excel(w, sheet_name="Sensitivity")
    prof.rename(index=names).to_excel(w, sheet_name="Cluster profiles")
    att.to_excel(w, sheet_name="Needs attention", index=False)
    good.to_excel(w, sheet_name="Doing well", index=False)
print(f"\nSaved figures and tables to ./{OUT_DIR}")
plt.show()
