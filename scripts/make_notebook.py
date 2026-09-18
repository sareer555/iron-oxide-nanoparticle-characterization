"""
make_notebook.py
================
Builds (and executes) notebooks/iron_oxide_characterization_analysis.ipynb.

The notebook is generated programmatically so that it stays in step with the
analysis module rather than drifting from it. It is then executed, so the
delivered .ipynb contains real outputs and inline figures.

Run:  python scripts/make_notebook.py
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

NB_PATH = os.path.join(cfg.PROJECT_ROOT, "notebooks",
                       "iron_oxide_characterization_analysis.ipynb")

md = new_markdown_cell
code = new_code_cell
cells: list = []

# ==========================================================================
# 1. Project overview
# ==========================================================================
cells.append(md(r"""
# Comprehensive Characterization and Python-Based Data Analysis of Iron Oxide Nanoparticles

### A portfolio demonstration of a five-technique materials-characterization workflow

---

> ## ⚠️ SYNTHETIC DATA — PLEASE READ FIRST
>
> **Every dataset in this notebook is synthetic.** It was generated
> computationally by `scripts/generate_synthetic_datasets.py` from explicit
> physical models plus realistic noise.
>
> - No material was synthesized.
> - No instrument was operated.
> - No measurement was performed.
>
> The purpose is to demonstrate **data-analysis and interpretation
> capability** — processing, quantification, statistics and critical reading
> of characterization data. Nothing here is evidence about any real sample,
> and nothing here should be cited as an experimental result.

---

## What this notebook demonstrates

| # | Section | Techniques / methods |
|---|---------|----------------------|
| 1 | Project overview | — |
| 2 | Imports | NumPy, pandas, SciPy, Matplotlib |
| 3 | Dataset loading | provenance-aware CSV loading |
| 4 | XRD | ALS background, pseudo-Voigt fitting, Scherrer, Williamson–Hall, Nelson–Riley |
| 5 | FTIR | baseline correction, Savitzky–Golay, band assignment with confidence grading |
| 6 | UV–Vis | turbidity correction, Tauc analysis *as a cautionary demonstration* |
| 7 | SEM | descriptive statistics, lognormal fitting, bootstrap CI, moment means |
| 8 | TGA | DTG derivative, region integration, logistic step deconvolution |
| 9 | Cross-technique | reconciling XRD vs SEM size metrics |
| 10 | Summary | consolidated results |
| 11 | Limitations | what these data can and cannot support |

## The hypothetical scenario

The five datasets were designed around one internally consistent
*hypothetical* scenario, **which was never carried out**. It exists only to
constrain the data generation so the five datasets tell a coherent story:

- **Material:** Fe₃O₄ (magnetite), cubic inverse spinel, space group *Fd*3̄*m*
- **Assumed route:** surfactant-free aqueous co-precipitation of Fe(II)/Fe(III)
  chlorides (1:2) in excess NH₄OH at 80 °C under N₂ — a Massart-type
  preparation
- **No capping agent** — a binding constraint: it is *why* the TGA data contain
  no organic-decomposition step and the FTIR data contain no C–H or C=O bands
- **Inert TGA atmosphere (N₂)** — in air, Fe₃O₄ oxidation would cause a mass
  *gain* of up to +3.45 %
"""))

# ==========================================================================
# 2. Imports
# ==========================================================================
cells.append(md("## 2. Imports and configuration\n\n"
                "The project modules live in `scripts/`. `config.py` holds the "
                "generative ground truth and the shared plotting style; "
                "`analysis.py` holds the processing functions used below, so "
                "the notebook and the batch pipeline cannot diverge."))
cells.append(code("""import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join("..", "scripts")))
import config as cfg
import analysis as an

cfg.apply_plot_style()
%matplotlib inline

print("numpy      ", np.__version__)
print("pandas     ", pd.__version__)
import scipy, matplotlib
print("scipy      ", scipy.__version__)
print("matplotlib ", matplotlib.__version__)
print()
print("Random seed:", cfg.RANDOM_SEED, "-> every dataset is reproducible")
print(cfg.DISCLAIMER_LONG)"""))

# ==========================================================================
# 3. Dataset loading
# ==========================================================================
cells.append(md("## 3. Dataset loading\n\n"
                "Every raw CSV carries a `#`-prefixed header stating that the "
                "data are synthetic and documenting the generative model. "
                "Reading that header first is a habit worth keeping: the "
                "provenance of a file is part of its data."))
cells.append(code("""RAW = cfg.RAW_DIR

with open(os.path.join(RAW, "xrd_data.csv"), encoding="utf-8") as fh:
    for line in fh:
        if not line.startswith("#"):
            break
        print(line.rstrip())"""))

cells.append(code("""datasets = {
    "XRD":    "xrd_data.csv",
    "FTIR":   "ftir_data.csv",
    "UV-Vis": "uvvis_data.csv",
    "SEM":    "sem_particle_measurements.csv",
    "TGA":    "tga_data.csv",
}

frames = {}
summary = []
for name, fname in datasets.items():
    df = pd.read_csv(os.path.join(RAW, fname), comment="#")
    frames[name] = df
    summary.append({
        "technique": name,
        "file": fname,
        "rows": len(df),
        "columns": ", ".join(df.columns),
    })

pd.DataFrame(summary)"""))

cells.append(code("""# Basic integrity checks before any analysis: shape, missing values, ranges.
for name, df in frames.items():
    n_missing = int(df.isna().sum().sum())
    num = df.select_dtypes("number")
    print(f"{name:7s} rows={len(df):5d}  missing={n_missing:3d}  "
          f"numeric columns={list(num.columns)}")"""))

# ==========================================================================
# 4. XRD
# ==========================================================================
cells.append(md(r"""## 4. XRD analysis

### 4.1 Background estimation

The raw pattern sits on a decaying background — with Cu Kα radiation an
iron-bearing sample fluoresces strongly, so the background is genuinely high.
An **asymmetric least squares (ALS)** baseline is used: points above the
current baseline (peaks) get a small weight, points below get a large one, so
the baseline is pulled *under* the peaks rather than through them.

**Stiffness matters.** A baseline flexible enough to follow the Lorentzian
peak wings artificially narrows the reflections and inflates the apparent
crystallite size."""))
cells.append(code("""xrd = frames["XRD"]
two_theta = xrd["two_theta_deg"].to_numpy(float)
counts = xrd["intensity_counts"].to_numpy(float)

background = an.baseline_als(counts, lam=1e7, p=0.001, n_iter=30)
net = counts - background

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(two_theta, counts, lw=0.7, color=cfg.COLORS["raw"], label="Raw")
ax.plot(two_theta, background, lw=1.4, ls="--", color=cfg.COLORS["accent"],
        label="ALS background")
ax.set_xlabel(r"2$\\theta$ (degrees)"); ax.set_ylabel("Intensity (counts)")
ax.set_xlim(20, 80); ax.legend()
ax.set_title("Synthetic XRD pattern and estimated background")
plt.show()

print(f"Background spans {background.min():.0f}-{background.max():.0f} counts")
print(f"Strongest net peak: {net.max():.0f} counts")"""))

cells.append(md(r"""### 4.2 Peak detection and indexing

For a cubic system, $d_{hkl} = a/\sqrt{h^2+k^2+l^2}$, and Bragg's law
$\lambda = 2d\sin\theta$ gives the expected positions. The reflection sequence
(220), (311), (400), (422), (511), (440) is the fingerprint of the inverse
spinel structure."""))
cells.append(code("""smooth = savgol_filter(np.clip(net, 0, None), 25, 3)

# Noise estimated on the SMOOTHED trace in a peak-free window. Using raw
# point-to-point scatter would over-estimate the threshold and lose the
# weak reflections.
quiet = (two_theta > 45) & (two_theta < 51)
sigma = float(np.std(smooth[quiet]))
found, _ = find_peaks(smooth, height=3*sigma, prominence=4.5*sigma, distance=17)

print(f"smoothed noise sigma = {sigma:.1f} counts")
print(f"{len(found)} candidate reflections detected\\n")

rows = []
for h, k, l, i_rel in cfg.MAGNETITE_REFLECTIONS:
    pos = cfg.two_theta_from_hkl(h, k, l)
    d = cfg.A_MAGNETITE_ANG / np.sqrt(h*h + k*k + l*l)
    pos_mag = cfg.two_theta_from_hkl(h, k, l, a_ang=cfg.A_MAGHEMITE_ANG)
    dist = np.abs(two_theta[found] - pos)
    matched = bool(dist.min() <= 0.5) if len(found) else False
    rows.append({"hkl": f"({h}{k}{l})", "d_ang": round(d, 4),
                 "2theta_Fe3O4": round(pos, 3),
                 "2theta_gamma_Fe2O3": round(pos_mag, 3),
                 "shift_deg": round(pos_mag - pos, 3),
                 "I_rel_ref": i_rel, "detected": matched})

idx = pd.DataFrame(rows)
idx"""))
cells.append(md("""Note the `shift_deg` column: switching from magnetite to
maghemite moves the reflections by only **0.2–0.5°**. That small a shift is
comparable to the errors introduced by specimen displacement or a zero offset
on a real diffractometer — which is exactly why XRD peak positions alone do
not settle the magnetite/maghemite question."""))

cells.append(md(r"""### 4.3 Profile fitting and the Scherrer equation

$$D = \frac{K\lambda}{\beta\cos\theta}$$

| Symbol | Meaning | Unit handling |
|--------|---------|---------------|
| $D$ | volume-weighted crystallite size | returned in nm |
| $K$ | shape factor, 0.9 for spheres | dimensionless |
| $\lambda$ | Cu K$\alpha_1$ wavelength | **0.154060 nm** (not Å) so $D$ is in nm |
| $\beta$ | sample broadening (FWHM) | **degrees → radians**; omitting this inflates $D$ by 57.3× |
| $\theta$ | Bragg angle | **$\theta = 2\theta/2$** |

Instrumental broadening is removed in quadrature:
$\beta_{sample} = \sqrt{\beta_{obs}^2 - \beta_{inst}^2}$."""))
cells.append(code("""# The full fitting routine lives in analysis.py; run it and load the results.
res = an.analyse_xrd()
fits = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "xrd_peak_analysis.csv"),
                   comment="#")
fits[["hkl", "two_theta_fit_deg", "d_spacing_ang", "fwhm_obs_deg",
      "fwhm_corrected_deg", "scherrer_size_nm", "i_rel_measured",
      "used_for_size"]]"""))

cells.append(code("""# Verify the Scherrer arithmetic by hand for the strongest reflection,
# showing every unit conversion explicitly.
row = fits.loc[fits.hkl == "(311)"].iloc[0]

K      = 0.9
lam_nm = 0.154060
beta_deg = float(row["fwhm_corrected_deg"])
beta_rad = beta_deg * np.pi / 180.0          # degrees -> RADIANS
theta_deg = float(row["two_theta_fit_deg"]) / 2.0   # 2-theta -> theta
theta_rad = theta_deg * np.pi / 180.0

D = K * lam_nm / (beta_rad * np.cos(theta_rad))

print(f"beta  = {beta_deg:.4f} deg = {beta_rad:.6f} rad")
print(f"theta = {theta_deg:.4f} deg = {theta_rad:.6f} rad")
print(f"D     = {K} x {lam_nm} / ({beta_rad:.6f} x {np.cos(theta_rad):.6f})"
      f" = {D:.2f} nm")
print(f"pipeline value: {row['scherrer_size_nm']:.2f} nm")
print()
D_wrong = K * lam_nm / (beta_deg * np.cos(theta_rad))
print(f"If beta were left in DEGREES: {D_wrong:.4f} nm "
      f"-- wrong by a factor of {D/D_wrong:.1f}")"""))

cells.append(md(r"""### 4.4 Williamson–Hall: separating size from strain

Scherrer attributes *all* broadening to size. If microstrain is present, that
biases the size low. Size and strain broadening scale differently with angle:

$$\beta\cos\theta = \frac{K\lambda}{D} + 4\varepsilon\sin\theta$$

so a plot of $\beta\cos\theta$ against $4\sin\theta$ gives the size from the
**intercept** and the strain from the **slope**."""))
cells.append(code("""strong = fits[fits.used_for_size]
th = np.radians(strong.two_theta_fit_deg / 2)
x = 4 * np.sin(th)
y = np.radians(strong.fwhm_corrected_deg) * np.cos(th)
w = 1.0 / np.radians(strong.fwhm_obs_err_deg.clip(lower=1e-5))

coef, cov = np.polyfit(x, y, 1, w=w, cov=True)
D_wh = cfg.SCHERRER_K * cfg.WAVELENGTH_NM / coef[1]

print(f"Scherrer, mean of strong peaks : {strong.scherrer_size_nm.mean():.2f} nm")
print(f"Williamson-Hall intercept      : D = {D_wh:.2f} nm")
print(f"Williamson-Hall slope          : eps = {coef[0]:.2e}")
print()
print(f"Injected ground truth          : D = {cfg.GROUND_TRUTH['xrd_crystallite_size_nm']} nm, "
      f"eps = {cfg.GROUND_TRUTH['xrd_microstrain']:.1e}")
print()
print("Scherrer UNDER-estimates because it absorbs the strain contribution")
print("into the size term. That is the physics, not a bug.")"""))

cells.append(md(f"""### 4.5 XRD outcome

The analysis recovers the injected crystallite size to within a few percent
and the lattice parameter to within one standard error. The **microstrain is
the weakest recovery** — with only a handful of reflections over a limited
angular range, the Williamson–Hall *slope* is far less well constrained than
the *intercept*. It should be read as an order of magnitude (~10⁻⁴), not as a
precise value."""))

# ==========================================================================
# 5. FTIR
# ==========================================================================
cells.append(md(r"""## 5. FTIR analysis

Band positions alone rarely justify a confident assignment. The workflow below
grades every detected band by confidence and explicitly declines to assign one
of them.

**Smoothing is chosen, not defaulted.** The narrowest genuine feature is about
9 cm⁻¹ wide; at 1 cm⁻¹ sampling an 11-point Savitzky–Golay filter stays well
below that, so band shapes and areas survive. Smoothing wide enough to look
attractive is usually wide enough to distort what you are measuring."""))
cells.append(code("""ftir_res = an.analyse_ftir()
peaks_tbl = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                     "ftir_peak_assignments.csv"), comment="#")
peaks_tbl[["peak_wavenumber_cm_1", "peak_absorbance_au", "assignment",
           "confidence"]]"""))
cells.append(code("""for _, r in peaks_tbl.iterrows():
    print(f"{r['peak_wavenumber_cm_1']:7.1f} cm-1  [{r['confidence']:>6s}]  "
          f"{r['assignment']}")
    print(f"{'':16s}{r['interpretation_note']}")
    print()"""))
cells.append(md("""### 5.1 Evidence from absence

Bands that are *not* present carry information — but absence has to be tested
properly. Asking whether absorbance is merely low in a window fails, because
the tail of a strong neighbouring band leaks into it (the 1630 cm⁻¹ water bend
bleeds into the carbonyl region). The defensible test is whether a **resolved
local maximum** exists there."""))
cells.append(code("""for name, absent in ftir_res["absence_checks"].items():
    print(f"{'ABSENT ' if absent else 'PRESENT'}  {name}")

print()
print("No C-H or C=O bands -> consistent with the uncoated-surface assumption")
print("and with the absence of an organic step in the TGA data.")
print("These are consistency checks, not proof: FTIR could miss a few percent")
print("of a trace phase without showing a resolved band.")"""))

# ==========================================================================
# 6. UV-Vis
# ==========================================================================
cells.append(md(r"""## 6. UV–Vis analysis

This section reaches a **negative result**, deliberately.

Two problems arise before any modelling:

1. **Photometric range.** Absorbance above ~2.5 is dominated by stray light on
   a conventional spectrophotometer. Those points are flagged and excluded.
2. **Scattering.** For a *dispersion*, the instrument records attenuation =
   absorption + turbidity. An empirical $\lambda^{-n}$ term is fitted at long
   wavelengths and subtracted."""))
cells.append(code("""uv_res = an.analyse_uvvis()

print(f"Points above the photometric limit : {uv_res['n_points_above_limit']}")
print(f"Reliable only above               : {uv_res['reliable_above_nm']:.0f} nm")
print(f"Effective turbidity exponent n    : {uv_res['scattering_exponent_n_effective']:.2f}")
print(f"Scattering share of A at 600 nm   : "
      f"{100*uv_res['scattering_fraction_at_600nm']:.0f} %")
print()
print(uv_res["absorption_character"])"""))

cells.append(md(r"""### 6.1 Tauc analysis — demonstrated, then *not* reported

$$(\alpha h\nu)^{1/r} = B\,(h\nu - E_g)$$

with $r = \tfrac12$ (direct allowed) or $r = 2$ (indirect allowed).

Three reasons this cannot yield a band gap here:

1. **Wrong material.** Fe₃O₄ is a mixed-valence, near-metallic conductor above
   the Verwey transition — not a wide-gap semiconductor. The Tauc relation was
   derived for amorphous semiconductors.
2. **Wrong quantity.** Tauc needs the absorption coefficient α. We have the
   absorbance of a dispersion, which conflates absorption with scattering.
3. **Unstable answer.** Quantified below."""))
cells.append(code("""rows = []
for key, v in uv_res["tauc_results"].items():
    base, trans = key.split("__")
    rows.append({"baseline": base.replace("_", " "),
                 "transition": trans.replace("_", " "),
                 "apparent_Eg_eV": v["Eg_eV"],
                 "fit_R2": v["r2"],
                 "window_eV": f"{v['window_eV'][0]:.2f}-{v['window_eV'][1]:.2f}"})
tauc_df = pd.DataFrame(rows)
display(tauc_df)

lo, hi = uv_res["tauc_full_defensible_range_eV"]
print(f"\\nEvery fit has R2 >= {tauc_df.fit_R2.min():.3f} -- they ALL look excellent.")
print(f"Yet the values span {tauc_df.apparent_Eg_eV.min():.2f}-"
      f"{tauc_df.apparent_Eg_eV.max():.2f} eV.")
print(f"Including the choice of linear window: {lo:.2f}-{hi:.2f} eV "
      f"({hi-lo:.2f} eV wide).")
print()
print("CONCLUSION: no band gap is reported from these data.")"""))
cells.append(md("""A spread of that size is larger than any difference such an
analysis would be trying to resolve. Reporting a single value and quoting its
excellent R² would be indefensible — yet it is common in the literature.
**Knowing which numbers not to report is part of analytical competence.**"""))

# ==========================================================================
# 7. SEM
# ==========================================================================
cells.append(md("""## 7. SEM-based particle-size analysis

**No micrograph is fabricated.** What is analysed is the measurement table an
analyst obtains *after* segmenting a real micrograph (e.g. in ImageJ/Fiji) —
the numeric input to the statistical workflow being demonstrated."""))
cells.append(code("""sem = frames["SEM"]
d = sem["equivalent_diameter_nm"].to_numpy(float)
sem.head(10)"""))
cells.append(code("""desc = sem["equivalent_diameter_nm"].describe()
q1, med, q3 = np.percentile(d, [25, 50, 75])

print(f"n                  {len(d)}")
print(f"mean               {d.mean():.2f} nm")
print(f"median             {med:.2f} nm")
print(f"std dev            {d.std(ddof=1):.2f} nm")
print(f"std error          {d.std(ddof=1)/np.sqrt(len(d)):.3f} nm")
print(f"min / max          {d.min():.2f} / {d.max():.2f} nm")
print(f"Q1 / Q3            {q1:.2f} / {q3:.2f} nm")
print(f"IQR                {q3-q1:.2f} nm")
print(f"CV                 {100*d.std(ddof=1)/d.mean():.1f} %")
print(f"skewness           {stats.skew(d):.3f}  (right-skewed)")"""))

cells.append(md("""### 7.1 Distribution shape, and the limits of n = 84"""))
cells.append(code("""shape, loc, scale = stats.lognorm.fit(d, floc=0)
ks_ln = stats.kstest(d, "lognorm", args=(shape, loc, scale))
mu_n, sd_n = stats.norm.fit(d)
ks_no = stats.kstest(d, "norm", args=(mu_n, sd_n))

print(f"lognormal: median = {scale:.2f} nm, GSD = {np.exp(shape):.3f}, "
      f"KS p = {ks_ln.pvalue:.3f}")
print(f"normal   : mean   = {mu_n:.2f} nm, SD  = {sd_n:.3f}, "
      f"KS p = {ks_no.pvalue:.3f}")
print()
print("Neither model is rejected. With n = 84 these data CANNOT distinguish")
print("lognormal from normal. The lognormal fit is reported because particle")
print("formation physics motivates it -- NOT because these data demonstrate it.")
print("Treating a non-significant KS test as proof of a distribution is a")
print("misuse of the test.")"""))

cells.append(md(r"""### 7.2 Number vs volume weighting

A plain mean of diameters is *number-weighted*. XRD, scattering and
sedimentation all weight toward larger particles. The moment means are

$$D_{[4,3]} = \frac{\sum d^4}{\sum d^3} \quad\text{(volume-weighted)}
\qquad
D_{[3,2]} = \frac{\sum d^3}{\sum d^2} \quad\text{(surface-weighted)}$$

Comparing a number-weighted SEM mean directly against a volume-weighted XRD
size compares two different statistics of two different quantities."""))
cells.append(code("""d43 = (d**4).sum() / (d**3).sum()
d32 = (d**3).sum() / (d**2).sum()

rng = np.random.default_rng(cfg.RANDOM_SEED)
boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
ci = np.percentile(boot, [2.5, 97.5])

print(f"number mean        D      = {d.mean():.2f} nm")
print(f"surface-weighted   D[3,2] = {d32:.2f} nm")
print(f"volume-weighted    D[4,3] = {d43:.2f} nm")
print(f"\\n95% bootstrap CI on the mean: {ci[0]:.2f} - {ci[1]:.2f} nm")"""))
cells.append(code("""sem_res = an.analyse_sem()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(d, bins="fd", density=True, color="#B8C4D4",
             edgecolor=cfg.COLORS["processed"], lw=0.8)
xs = np.linspace(d.min()*0.85, d.max()*1.12, 400)
axes[0].plot(xs, stats.lognorm.pdf(xs, shape, loc, scale),
             color=cfg.COLORS["accent"], lw=2, label="lognormal fit")
axes[0].axvline(d.mean(), ls="--", color=cfg.COLORS["fit"], label="mean")
axes[0].axvline(med, ls=":", color=cfg.COLORS["highlight"], label="median")
axes[0].set_xlabel("Equivalent diameter (nm)")
axes[0].set_ylabel("Probability density (nm$^{-1}$)")
axes[0].legend(); axes[0].set_title(f"Size distribution (n = {len(d)})")

bp = axes[1].boxplot(d, widths=0.4, patch_artist=True, showmeans=True)
bp["boxes"][0].set_facecolor("#B8C4D4")
axes[1].plot(np.random.default_rng(7).normal(1, 0.035, len(d)), d, "o",
             ms=3, color="#5A6B80", alpha=0.45)
axes[1].set_ylabel("Equivalent diameter (nm)"); axes[1].set_xticks([])
axes[1].set_title("Box plot with raw measurements overlaid")
plt.tight_layout(); plt.show()"""))

# ==========================================================================
# 8. TGA
# ==========================================================================
cells.append(md(r"""## 8. TGA analysis

DTG $= -\mathrm{d}m/\mathrm{d}T$ is computed analytically from the local
Savitzky–Golay polynomial rather than by finite differences — differentiation
amplifies noise, and an under-smoothed DTG trace invents maxima that are not
there. A wider window is used for the derivative than for the mass curve."""))
cells.append(code("""tga_res = an.analyse_tga()
regions = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                   "tga_mass_loss_regions.csv"), comment="#")
regions[["region", "T_start_C", "T_end_C", "mass_loss_percent",
         "fraction_of_total_loss_percent", "dtg_max_temperature_C",
         "assignment_confidence"]]"""))

cells.append(md("""### 8.1 Region integration vs step deconvolution

Region-based integration depends on where the analyst draws the boundaries,
and adjacent events overlap across them. Fitting the whole curve as a sum of
logistic steps plus a linear drift removes that arbitrariness and returns the
amplitude of each **event** rather than of each temperature **interval**."""))
cells.append(code("""dec = tga_res["step_deconvolution"]
print(f"converged: {dec['converged']},  R2 = {dec['r_squared']:.5f}\\n")

gt = [cfg.GROUND_TRUTH[k] for k in
      ("tga_step1_pct", "tga_step2_pct", "tga_step3_pct")]
for i, (s, truth) in enumerate(zip(dec["steps"], gt), start=1):
    print(f"step {i}: {s['amplitude_percent']:.2f} +/- "
          f"{s['amplitude_err_percent']:.2f} %  at {s['midpoint_C']:.1f} C   "
          f"(injected {truth:.2f} %)")

print(f"\\nSummed amplitude      : {dec['summed_amplitude_percent']:.2f} %")
print(f"Measured total loss   : {tga_res['total_mass_loss_percent']:.2f} %")
print("\\nThe sum exceeds the measured loss because part of the lowest-")
print("temperature event happens BEFORE the run reaches 25 C -- a real effect")
print("that makes TGA under-report total volatile content.")"""))

cells.append(md("""### 8.2 Combining TGA with XRD: surface water coverage

Treating the crystallites as spheres of the Williamson–Hall diameter gives a
specific surface area; the Region II mass loss then converts to an areal water
density. This is a genuine cross-technique result — neither technique could
produce it alone."""))
cells.append(code("""D_nm = res["williamson_hall_size_nm"]
rho = 5.17                      # g/cm3, magnetite
ssa = 6.0 / (rho * D_nm * 1e-7) / 1e4      # m2/g
region_ii = regions.loc[regions.region == "Region II",
                        "mass_loss_percent"].iloc[0]

molecules = (region_ii / 100.0) / 18.015 * 6.02214076e23   # per gram
coverage = molecules / (ssa * 1e18)

print(f"XRD crystallite size      : {D_nm:.2f} nm")
print(f"Implied specific surface  : {ssa:.0f} m2/g  (spheres)")
print(f"Region II mass loss       : {region_ii:.2f} %")
print(f"-> surface water coverage : {coverage:.2f} molecules/nm2")
print()
print("Typical hydroxylated iron-oxide surfaces: ~2-12 /nm2.")
print("The datasets are mutually consistent. Note this inherits the")
print("sphericity assumption and ignores porosity and agglomeration.")"""))

cells.append(md("""### 8.3 Why atmosphere is not a footnote

The inert atmosphere here is an **assumption**. In air, Fe₃O₄ oxidises to
γ-Fe₂O₃ and then α-Fe₂O₃ above ~200 °C, a mass **gain** of up to **+3.45 %**
that would partly cancel the losses — a genuinely hydrated sample could look
nearly anhydrous. A TGA result reported without its atmosphere, heating rate
and sample mass is uninterpretable."""))

# ==========================================================================
# 9. Cross-technique
# ==========================================================================
cells.append(md("""## 9. Cross-technique comparison"""))
cells.append(code("""cross = an.cross_technique_synthesis()
matrix = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                  "technique_comparison_matrix.csv"),
                     comment="#")
pd.set_option("display.max_colwidth", 70)
matrix[["technique", "main_information", "quantitative_output",
        "key_limitation"]]"""))

cells.append(md("""### 9.1 Why XRD and SEM sizes disagree

This is the most informative single result in the project. The two numbers
differ — and that is the expected outcome of measuring different quantities
with different statistical weightings, not a discrepancy to explain away."""))
cells.append(code("""cmp = cross["size_comparison"]
print(f"XRD Scherrer (mean)        : {cmp['xrd_scherrer_mean_nm']:.2f} nm")
print(f"XRD Williamson-Hall        : {cmp['xrd_williamson_hall_nm']:.2f} nm")
print(f"SEM number mean            : {cmp['sem_number_mean_nm']:.2f} nm")
print(f"SEM volume-weighted D[4,3] : {cmp['sem_volume_weighted_D43_nm']:.2f} nm")
print()
print(f"ratio (number-weighted)    : {cmp['ratio_sem_number_to_xrd']:.2f}")
print(f"ratio (volume-weighted)    : {cmp['ratio_sem_volume_to_xrd']:.2f}")
print()
for i, reason in enumerate(cmp["reasons_for_divergence"], start=1):
    print(f"{i}. {reason}\\n")"""))

# ==========================================================================
# 10. Summary
# ==========================================================================
cells.append(md("""## 10. Summary of demonstration outcomes"""))
cells.append(code("""with open(os.path.join(cfg.REPORT_DIR, "analysis_results.json"),
          encoding="utf-8") as fh:
    R = json.load(fh)

X, S_, T_, U_ = R["xrd"], R["sem"], R["tga"], R["uvvis"]
rows = [
    ("XRD", "Phase assignment", X["phase_assignment"]),
    ("XRD", "Lattice parameter",
     f"{X['lattice_parameter_ang']:.4f} +/- {X['lattice_parameter_err_ang']:.4f} A"),
    ("XRD", "Crystallite size (Scherrer)",
     f"{X['scherrer_mean_nm']:.2f} +/- {X['scherrer_sd_nm']:.2f} nm"),
    ("XRD", "Crystallite size (W-H)",
     f"{X['williamson_hall_size_nm']:.2f} +/- {X['williamson_hall_size_err_nm']:.2f} nm"),
    ("XRD", "Microstrain (order of magnitude)",
     f"~{X['williamson_hall_microstrain']:.0e}"),
    ("FTIR", "Bands resolved", f"{R['ftir']['n_bands_detected']}"),
    ("FTIR", "Organic bands", "none detected (uncoated surface)"),
    ("UV-Vis", "Optical band gap", "NOT REPORTED - see section 6"),
    ("UV-Vis", "Tauc spread across defensible choices",
     f"{U_['tauc_full_defensible_range_eV'][0]:.2f}-"
     f"{U_['tauc_full_defensible_range_eV'][1]:.2f} eV"),
    ("SEM", "Number mean diameter",
     f"{S_['mean_nm']:.2f} nm (n = {S_['n']})"),
    ("SEM", "Volume-weighted D[4,3]",
     f"{S_['volume_weighted_D43_nm']:.2f} nm"),
    ("TGA", "Total mass loss", f"{T_['total_mass_loss_percent']:.2f} %"),
    ("TGA", "Events resolved",
     ", ".join(f"{p['temperature_C']:.0f} C" for p in T_["dtg_peaks"])),
    ("Cross", "SEM / XRD size ratio",
     f"{cmp['ratio_sem_number_to_xrd']:.2f} (number), "
     f"{cmp['ratio_sem_volume_to_xrd']:.2f} (volume)"),
]
pd.DataFrame(rows, columns=["Technique", "Quantity", "Demonstration outcome"])"""))

cells.append(md("""### 10.1 Pipeline validation

Because the generative parameters are known, the analysis can be checked
against them — impossible with real data. `scripts/validate.py` runs 55
automated checks across parameter recovery, unit/arithmetic consistency and
synthetic-data integrity."""))
cells.append(code("""gtv = cfg.GROUND_TRUTH
checks = [
    ("Crystallite size (W-H)", f"{gtv['xrd_crystallite_size_nm']:.2f} nm",
     f"{X['williamson_hall_size_nm']:.2f} nm"),
    ("Lattice parameter", f"{gtv['xrd_lattice_parameter_ang']:.4f} A",
     f"{X['lattice_parameter_ang']:.4f} A"),
    ("Microstrain", f"{gtv['xrd_microstrain']:.1e}",
     f"{X['williamson_hall_microstrain']:.1e}"),
    ("SEM lognormal median", f"{gtv['sem_lognormal_median_nm']:.2f} nm",
     f"{S_['lognormal_median_nm']:.2f} nm"),
    ("SEM geometric SD", f"{gtv['sem_lognormal_gsd']:.3f}",
     f"{S_['lognormal_gsd']:.3f}"),
    ("TGA step 1", f"{gtv['tga_step1_pct']:.2f} %",
     f"{T_['step_deconvolution']['steps'][0]['amplitude_percent']:.2f} %"),
    ("TGA step 2", f"{gtv['tga_step2_pct']:.2f} %",
     f"{T_['step_deconvolution']['steps'][1]['amplitude_percent']:.2f} %"),
    ("TGA step 3", f"{gtv['tga_step3_pct']:.2f} %",
     f"{T_['step_deconvolution']['steps'][2]['amplitude_percent']:.2f} %"),
]
pd.DataFrame(checks, columns=["Quantity", "Injected", "Recovered"])"""))

# ==========================================================================
# 11. Limitations
# ==========================================================================
cells.append(md("""## 11. Limitations

### 11.1 The overriding limitation

**These data are synthetic.** They contain exactly the physics the generative
models encode and no more. Real specimens bring preferred orientation,
anisotropic crystallite shapes, stacking faults, amorphous fractions invisible
to diffraction, inhomogeneity between aliquots, instrument drift and
preparation artefacts. **Nothing here is evidence about any real material.**

### 11.2 Method-specific limitations

| Method | Limitation |
|--------|-----------|
| Scherrer | Gives a **volume-weighted coherent-domain size**, not a particle size. Assumes strain-free crystallites unless corrected; *K* is uncertain at the ±10 % level; unreliable above ~100 nm. |
| Single-peak fitting | Rietveld refinement of the full pattern would constrain parameters better; single-peak methods were used here to show them explicitly. |
| Kα₂ | The pattern was generated for a single wavelength. Real lab data contain the doublet, which must be stripped before width analysis. |
| FTIR | Band intensities are not concentrations without absorptivities. ATR and KBr give different relative intensities and positions. |
| Band overlap | The Fe–O region is congested and shifts with size, oxidation state and sampling geometry — FTIR cannot settle magnetite vs maghemite. |
| UV–Vis | Dispersion optics conflate absorption with scattering; no band gap is reportable for this material. |
| SEM statistics | n = 84 constrains the mean but not the distribution shape or tails; agglomeration, thresholding, sampling bias and coating all bias sizes upward. |
| TGA | Measures mass only, never identity. Depends on atmosphere, heating rate, sample mass and packing — all assumed here. |

### 11.3 What this notebook *does* establish

That the author can build a reproducible analytical pipeline, apply
established physical relations with correct unit handling, quantify
uncertainty, validate results against known inputs, distinguish signal from
artefact, grade confidence in interpretations — and decline to report a number
when the data do not support one.

---

*End of notebook. Generated from `scripts/make_notebook.py`; all analysis
functions live in `scripts/analysis.py`.*"""))


def build(execute: bool = True) -> str:
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Notebook written ({len(cells)} cells): "
          f"notebooks/iron_oxide_characterization_analysis.ipynb")

    if execute:
        try:
            from nbclient import NotebookClient
            nb = nbf.read(NB_PATH, as_version=4)
            client = NotebookClient(
                nb, timeout=900, kernel_name="python3",
                resources={"metadata": {"path": os.path.dirname(NB_PATH)}})
            client.execute()
            with open(NB_PATH, "w", encoding="utf-8") as fh:
                nbf.write(nb, fh)
            print("Notebook executed successfully; outputs embedded.")
        except Exception as exc:
            print(f"NOTE: execution failed ({type(exc).__name__}: {exc})")
            print("The notebook file itself is still valid and runnable.")
            raise
    return NB_PATH


if __name__ == "__main__":
    build(execute="--no-exec" not in sys.argv)
