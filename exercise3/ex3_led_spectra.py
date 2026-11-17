"""
Exercise 3 - Emission spectra of LEDs (Ocean Optics compact spectrometer)
  1. Plot all emission spectra on the same axis system (x = wavelength)
  2. Normalize the data set and plot it again
For every LED: background (dark) level, peak wavelength, FWHM, centroid, peak photon energy,
linewidth in energy (meV and in units of kT), signal-to-noise ratio.
LEDs with no real emission (only room-light lines) are detected and flagged automatically.
Run:  python ex3_led_spectra.py   (needs Exercise3_LED_data.xlsx in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- settings ----------------
DATA_FILE = "Exercise3_LED_data.xlsx"
OUT_DIR = "ex3_outputs"
MIN_SIGNAL = 50          # counts above background needed to call it real LED emission
HG_LINES = (435.8, 546.1)  # mercury lines from fluorescent room lights (nm)
KT_MEV = 25.7            # thermal energy kT at ~25 °C (meV)
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 12, "legend.fontsize": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})

# ---------------- load data ----------------
df = pd.read_excel(DATA_FILE, sheet_name="Data")
wl = df.iloc[:, 0].to_numpy(float)                 # wavelength (nm)
spectra = df.iloc[:, 1:]                           # one column per LED (counts)
print(f"{len(wl)} wavelengths ({wl.min():.0f}-{wl.max():.0f} nm), {spectra.shape[1]} LEDs")


def wavelength_to_colour(lam):
    """Approximate display colour of a wavelength (violet...red); IR shown dark red."""
    lam = np.clip(lam, 380, 780)
    if lam < 440:   r, g, b = -(lam - 440) / 60, 0, 1
    elif lam < 490: r, g, b = 0, (lam - 440) / 50, 1
    elif lam < 510: r, g, b = 0, 1, -(lam - 510) / 20
    elif lam < 580: r, g, b = (lam - 510) / 70, 1, 0
    elif lam < 645: r, g, b = 1, -(lam - 645) / 65, 0
    else:           r, g, b = 1, 0, 0
    f = 0.3 + 0.7 * (lam - 380) / 40 if lam < 420 else 0.3 + 0.7 * (780 - lam) / 80 if lam > 700 else 1
    return (r * f * 0.9, g * f * 0.8, b * f * 0.9)


IR_STYLES = [("#c0392b", "--"), ("#7d3c98", "--"), ("#3d3d3d", "--"), ("#a04000", "-."), ("#1a5276", "-.")]


def led_style(peak_nm, ir_index):
    """Colour and line style for an LED: true colour if visible, distinct dashed styles if infrared."""
    if peak_nm <= 700:
        return wavelength_to_colour(peak_nm), "-"
    return IR_STYLES[ir_index % len(IR_STYLES)]


def half_max_crossing(x, y, i_from, step):
    """Walk from the peak until y < 0.5 and linearly interpolate the 0.5 crossing."""
    i = i_from
    while 0 < i < len(y) - 1 and y[i] >= 0.5:
        i += step
    return np.interp(0.5, [y[i], y[i - step]], [x[i], x[i - step]])


# ---------------- analyse each LED ----------------
rows, norm = [], {}
for led in spectra.columns:
    I = spectra[led].to_numpy(float)
    bg = np.median(np.sort(I)[: len(I) // 5])           # background = median of lowest 20 % of points
    noise = np.std(np.sort(I)[: len(I) // 5], ddof=1)   # noise of the background region
    peak_counts = I.max() - bg
    ip = int(np.argmax(I))
    if peak_counts < MIN_SIGNAL:                         # no real LED emission
        rows.append({"LED": led, "Background (counts)": bg, "Peak above background (counts)": peak_counts,
                     "SNR": peak_counts / noise, "Status": "no LED signal (room-light lines only)"})
        continue
    In = (I - bg) / peak_counts                          # normalised: background 0, peak 1
    norm[led] = In
    left = half_max_crossing(wl, In, ip, -1)
    right = half_max_crossing(wl, In, ip, +1)
    fwhm = right - left
    m = In > 0.1                                          # centroid over the region above 10 % of peak
    centroid = np.sum(wl[m] * In[m]) / np.sum(In[m])
    E_peak = 1239.84 / wl[ip]                             # photon energy (eV)
    dE = 1239.84 * (1 / left - 1 / right) * 1000          # FWHM in energy (meV)
    rows.append({"LED": led, "Background (counts)": bg, "Peak above background (counts)": peak_counts,
                 "SNR": peak_counts / noise, "Peak wavelength (nm)": wl[ip], "Centroid (nm)": centroid,
                 "FWHM (nm)": fwhm, "Half-max left (nm)": left, "Half-max right (nm)": right,
                 "Peak photon energy (eV)": E_peak, "FWHM (meV)": dE, "FWHM / kT": dE / KT_MEV,
                 "Asymmetry: centroid - peak (nm)": centroid - wl[ip], "Status": "OK"})

res = pd.DataFrame(rows).sort_values("Peak wavelength (nm)", na_position="last")
pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False))
ok = res[res["Status"] == "OK"]
order = list(ok["LED"])                                  # LEDs sorted by peak wavelength
bad = list(res.loc[res["Status"] != "OK", "LED"])
STYLE, n_ir = {}, 0
for led in order:
    p = float(ok.loc[ok.LED == led, "Peak wavelength (nm)"].iloc[0])
    STYLE[led] = led_style(p, n_ir)
    n_ir += p > 700
LEGEND = dict(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

# ======================= FIGURE 1: raw spectra, linear scale =======================
fig, ax = plt.subplots(figsize=(10, 5.5))
for led in order:
    p = float(ok.loc[ok.LED == led, "Peak wavelength (nm)"].iloc[0])
    ax.plot(wl, spectra[led], color=STYLE[led][0], ls=STYLE[led][1], lw=1.4, label=f"{led} ({p:.0f} nm)")
for led in bad:
    ax.plot(wl, spectra[led], color="0.6", lw=1, ls="--", label=f"{led} (no LED signal)")
ax.set(xlabel="Wavelength (nm)", ylabel="Intensity (counts)", title="Emission spectra of the LEDs (raw data)",
       xlim=(wl.min(), wl.max()))
ax.legend(**LEGEND)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex3_fig1_raw_spectra.png", bbox_inches="tight")

# ======================= FIGURE 2: raw spectra, log scale =======================
fig, ax = plt.subplots(figsize=(10, 5.5))
for led in order:
    p = float(ok.loc[ok.LED == led, "Peak wavelength (nm)"].iloc[0])
    ax.plot(wl, spectra[led], color=STYLE[led][0], ls=STYLE[led][1], lw=1.4, label=f"{led} ({p:.0f} nm)")
for led in bad:
    ax.plot(wl, spectra[led], color="0.6", lw=1, ls="--", label=f"{led} (no LED signal)")
for h in HG_LINES:
    ax.axvline(h, color="0.4", ls=":", lw=0.8)
ax.text(HG_LINES[1] + 3, spectra.values.max() * 0.7, "Hg lines 436 & 546 nm\n(room fluorescent light)", fontsize=8)
ax.set_yscale("log")
ax.set(xlabel="Wavelength (nm)", ylabel="Intensity (counts, log scale)",
       title="Raw spectra on a log scale (weak IR LED and background visible)", xlim=(wl.min(), wl.max()))
ax.legend(**LEGEND)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex3_fig2_raw_spectra_log.png", bbox_inches="tight")

# ======================= FIGURE 3: normalised spectra =======================
fig, ax = plt.subplots(figsize=(10, 5.5))
for led in order:
    r = ok[ok.LED == led].iloc[0]
    col, ls = STYLE[led]
    ax.plot(wl, norm[led], color=col, ls=ls, lw=1.4,
            label=f"{led}: {r['Peak wavelength (nm)']:.0f} nm, FWHM {r['FWHM (nm)']:.1f} nm")
    ax.hlines(0.5, r["Half-max left (nm)"], r["Half-max right (nm)"], color=col, lw=3, alpha=0.5)
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.text(355, 0.52, "half maximum", fontsize=8)
ax.set(xlabel="Wavelength (nm)", ylabel="Normalised intensity",
       title="Normalised emission spectra: $(I - I_{bg}) / (I_{max} - I_{bg})$",
       xlim=(wl.min(), wl.max()), ylim=(-0.05, 1.08))
ax.legend(**LEGEND)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex3_fig3_normalised_spectra.png", bbox_inches="tight")

# ======================= FIGURE 4 (extra): linewidth vs wavelength =======================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
for _, r in ok.iterrows():
    col = STYLE[r["LED"]][0]
    a1.scatter(r["Peak wavelength (nm)"], r["FWHM (nm)"], color=col, s=60, edgecolor="k", zorder=3)
    a2.scatter(r["Peak photon energy (eV)"], r["FWHM / kT"], color=col, s=60, edgecolor="k", zorder=3)
    for a, x, y in [(a1, r["Peak wavelength (nm)"], r["FWHM (nm)"]),
                    (a2, r["Peak photon energy (eV)"], r["FWHM / kT"])]:
        a.annotate(r["LED"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
a1.set(xlabel="Peak wavelength (nm)", ylabel="FWHM (nm)", title="(a) Linewidth in wavelength")
a2.axhline(1.8, color="k", ls="--", lw=1, label="≈1.8 kT (ideal band-to-band emission)")
a2.set(xlabel="Peak photon energy (eV)", ylabel="FWHM / kT", title="(b) Linewidth in energy units")
a2.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ex3_fig4_linewidth.png", bbox_inches="tight")

# ---------------- save results ----------------
with pd.ExcelWriter(f"{OUT_DIR}/ex3_results.xlsx") as w:
    res.to_excel(w, sheet_name="LED parameters", index=False)
    pd.DataFrame({"Wavelength_nm": wl, **{k: norm[k] for k in order}}).to_excel(
        w, sheet_name="Normalised spectra", index=False)
print(f"\nSaved figures and results to ./{OUT_DIR}")
plt.show()
