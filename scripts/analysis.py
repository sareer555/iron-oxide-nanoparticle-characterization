"""
analysis.py
===========
Complete analysis pipeline for the iron-oxide nanoparticle characterization
DEMONSTRATION project.

>>> EVERY DATASET PROCESSED HERE IS SYNTHETIC. <<<
This script demonstrates an analyst's workflow -- loading, cleaning,
background/baseline handling, peak fitting, quantitative extraction,
statistics and visualisation -- on clearly-labelled synthetic data. Nothing
in this file reports an experimental measurement.

Pipeline
--------
    load raw CSV -> clean -> background/baseline -> feature detection
                 -> quantitative model -> processed CSV + figure + results

Run:  python scripts/analysis.py
Outputs: data/processed/*.csv, figures/*.png|pdf, report/analysis_results.json

References for the methods used are collected in references/references.bib.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

cfg.apply_plot_style()

RESULTS: dict = {
    "_disclaimer": cfg.DISCLAIMER_LONG,
    "_project": "Comprehensive Characterization and Python-Based Data "
                "Analysis of Iron Oxide Nanoparticles (demonstration)",
}

# Physical constants
PLANCK_EV_NM = 1239.841984      # h*c in eV*nm (CODATA-derived)
RHO_MAGNETITE_G_CM3 = 5.17      # magnetite density, Cornell & Schwertmann
M_WATER_G_MOL = 18.015
N_AVOGADRO = 6.02214076e23


# ==========================================================================
# Generic utilities
# ==========================================================================
def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw CSV, skipping the '#' disclaimer header."""
    path = os.path.join(cfg.RAW_DIR, filename)
    df = pd.read_csv(path, comment="#")
    return df


def save_processed(df: pd.DataFrame, filename: str, note: str) -> None:
    path = os.path.join(cfg.PROCESSED_DIR, filename)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# PROCESSED SYNTHETIC DATASET -- {note}\n")
        fh.write(f"# {cfg.DISCLAIMER_LONG}\n")
        df.to_csv(fh, index=False, lineterminator="\n")
    print(f"    -> data/processed/{filename}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure in every configured format at publication resolution."""
    for ext in cfg.FIGURE_FORMATS:
        fig.savefig(os.path.join(cfg.FIGURE_DIR, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"    -> figures/{stem}.{'/'.join(cfg.FIGURE_FORMATS)}")


def stamp(ax: plt.Axes, text: str = "Synthetic demonstration data",
          loc: str = "lower right") -> None:
    """Put an unobtrusive synthetic-data stamp on every figure.

    Drawn in FIGURE coordinates, in the margin below the axes, so that it can
    never overlap the plotted data no matter how the traces fall. `loc` is
    retained for call-site compatibility and only selects left/right.
    """
    x, ha = ((0.005, "left") if "left" in loc else (0.995, "right"))
    ax.figure.text(x, 0.004, text, ha=ha, va="bottom",
                   fontsize=7, color="#8A8A8A", style="italic")


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 n_iter: int = 20) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    Minimises  ||W(y - z)||^2 + lam * ||D2 z||^2  where the weights W are
    updated asymmetrically: points ABOVE the current baseline (i.e. peaks)
    receive weight `p` (small), points below receive `1 - p`. The baseline is
    therefore pulled under the peaks rather than through them.

    lam controls smoothness (larger = stiffer); p controls asymmetry.
    """
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    DTD = lam * D.dot(D.transpose())
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DTD).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ==========================================================================
# SECTION 4 -- XRD
# ==========================================================================
def pseudo_voigt(x, center, fwhm, eta, amplitude):
    """Unit-amplitude-scaled pseudo-Voigt (see generator for definition)."""
    dx = (x - center) / fwhm
    lorentz = 1.0 / (1.0 + 4.0 * dx ** 2)
    gauss = np.exp(-4.0 * np.log(2.0) * dx ** 2)
    return amplitude * (eta * lorentz + (1.0 - eta) * gauss)


def _multi_pv_with_linear_bg(x, *params):
    """n pseudo-Voigts + a local linear background.

    params = [c1,f1,e1,a1, c2,f2,e2,a2, ..., slope, intercept]
    """
    n_peaks = (len(params) - 2) // 4
    slope, intercept = params[-2], params[-1]
    y = slope * x + intercept
    for i in range(n_peaks):
        c, f, e, a = params[4 * i:4 * i + 4]
        y = y + pseudo_voigt(x, c, f, e, a)
    return y


def scherrer_size_nm(fwhm_deg: float, two_theta_deg: float,
                     k: float = cfg.SCHERRER_K,
                     wavelength_nm: float = cfg.WAVELENGTH_NM,
                     instrumental_fwhm_deg: float | None = None) -> float:
    """Volume-weighted crystallite size from the Scherrer equation.

            D = K * lambda / (beta * cos(theta))

    D      : mean crystallite dimension perpendicular to (hkl)   [nm]
    K      : dimensionless shape factor (0.9 for spheres)        [-]
    lambda : X-ray wavelength                                    [nm]
    beta   : integral/FWHM broadening of the reflection          [RADIANS]
    theta  : Bragg angle = (2-theta)/2                           [rad]

    Unit handling -- the two mistakes that dominate bad Scherrer numbers:
      1. `beta` MUST be converted from degrees to radians.
      2. `theta` is HALF the measured 2-theta value.
    Both are done explicitly below.

    Instrumental broadening is removed in quadrature (Gaussian assumption):
            beta_sample = sqrt(beta_observed^2 - beta_instrumental^2)
    """
    if instrumental_fwhm_deg is not None:
        corrected_sq = fwhm_deg ** 2 - instrumental_fwhm_deg ** 2
        if corrected_sq <= 0:
            return np.nan
        fwhm_deg = float(np.sqrt(corrected_sq))

    beta_rad = np.radians(fwhm_deg)          # degrees -> radians
    theta_rad = np.radians(two_theta_deg / 2.0)   # 2-theta -> theta
    return float(k * wavelength_nm / (beta_rad * np.cos(theta_rad)))


def _analyse_xrd_impl() -> dict:
    print("\n[XRD] ------------------------------------------------------")
    raw = load_raw("xrd_data.csv")
    two_theta = raw["two_theta_deg"].to_numpy(float)
    counts = raw["intensity_counts"].to_numpy(float)

    # --- 1. Cleaning -----------------------------------------------------
    # Drop any non-finite points and enforce monotonic 2-theta.
    mask = np.isfinite(two_theta) & np.isfinite(counts)
    two_theta, counts = two_theta[mask], counts[mask]
    order = np.argsort(two_theta)
    two_theta, counts = two_theta[order], counts[order]
    print(f"  loaded {len(two_theta)} points, "
          f"{two_theta.min():.2f}-{two_theta.max():.2f} deg 2-theta")

    # --- 2. Background estimation ----------------------------------------
    # Stiff ALS baseline (large lambda): the background varies slowly while
    # the Bragg peaks do not. A baseline that is too flexible will ride up
    # into the Lorentzian peak wings and artificially narrow the reflections,
    # which propagates straight into an over-estimated crystallite size.
    background = baseline_als(counts, lam=1e7, p=0.001, n_iter=30)
    net = counts - background
    net_clipped = np.clip(net, 0.0, None)

    # --- 3. Peak detection -----------------------------------------------
    # Smooth a WORKING COPY only (never the data used for intensities) so
    # that peak picking is not derailed by Poisson noise.
    smooth = savgol_filter(net_clipped, window_length=25, polyorder=3)
    # Estimate the noise on the SMOOTHED trace, in a genuinely peak-free
    # window. Using the raw point-to-point scatter would over-estimate the
    # detection threshold by the smoothing factor and lose the weak
    # reflections entirely.
    quiet = (two_theta > 45.0) & (two_theta < 51.0)
    noise_sigma = float(np.std(smooth[quiet]))
    found, props = find_peaks(smooth,
                              height=3.0 * noise_sigma,
                              prominence=4.5 * noise_sigma,
                              distance=int(0.35 / 0.02))
    print(f"  detected {len(found)} candidate reflections "
          f"(smoothed noise sigma ~ {noise_sigma:.1f} counts)")

    # Empirical half-width of each detected peak, measured directly from the
    # data. This sets the profile-fitting window WITHOUT assuming any
    # crystallite size -- important, because the size is what we are trying
    # to measure.
    widths_samples, _, _, _ = peak_widths(smooth, found, rel_height=0.5)
    step_deg = float(np.median(np.diff(two_theta)))
    empirical_fwhm = {int(i): float(w * step_deg)
                      for i, w in zip(found, widths_samples)}

    # --- 4. Index the detected peaks against cubic Fe3O4 ------------------
    expected = [(h, k, l, cfg.two_theta_from_hkl(h, k, l), i_rel)
                for h, k, l, i_rel in cfg.MAGNETITE_REFLECTIONS]

    indexed = []
    for h, k, l, pos_ref, i_rel in expected:
        if pos_ref < two_theta.min() or pos_ref > two_theta.max():
            continue
        # nearest detected peak within a 0.5 deg tolerance
        if len(found):
            d = np.abs(two_theta[found] - pos_ref)
            j = int(np.argmin(d))
            matched = bool(d[j] <= 0.5)
        else:
            matched = False
        indexed.append(dict(h=h, k=k, l=l, hkl=f"({h}{k}{l})",
                            two_theta_ref_deg=pos_ref, i_rel_ref=i_rel,
                            detected=matched,
                            two_theta_detected_deg=(float(two_theta[found][j])
                                                    if matched else np.nan),
                            fwhm_empirical_deg=(
                                empirical_fwhm[int(found[j])]
                                if matched else np.nan)))
    n_matched = sum(1 for p in indexed if p["detected"])
    print(f"  indexed {n_matched}/{len(indexed)} expected Fe3O4 reflections")

    # --- 5. Profile fitting ----------------------------------------------
    # Two decisions here materially affect the crystallite size:
    #
    #   (a) Fit the BACKGROUND-SUBTRACTED pattern, with only a small linear
    #       residual term. Fitting raw counts forces the local background to
    #       trade off against the peak wings.
    #   (b) Scale the fitting window to each peak's OWN measured width
    #       (WINDOW_FWHM_MULT x empirical FWHM) rather than using a fixed
    #       window. A fixed window truncates the Lorentzian wings of the
    #       broader high-angle reflections more severely than the narrow
    #       low-angle ones, which tilts the Williamson-Hall slope and can
    #       even produce a spurious NEGATIVE microstrain.
    WINDOW_FWHM_MULT = 4.0
    MIN_HALF_WINDOW = 1.6  # deg

    def half_window(p: dict) -> float:
        w = p.get("fwhm_empirical_deg", np.nan)
        if not np.isfinite(w) or w <= 0:
            return MIN_HALF_WINDOW
        return max(MIN_HALF_WINDOW, WINDOW_FWHM_MULT * w)

    fit_targets = [p for p in indexed if p["detected"]]

    groups: list[list[dict]] = []
    for p in sorted(fit_targets, key=lambda q: q["two_theta_ref_deg"]):
        if groups and (p["two_theta_ref_deg"] - half_window(p)
                       < groups[-1][-1]["two_theta_ref_deg"]
                       + half_window(groups[-1][-1])):
            groups[-1].append(p)
        else:
            groups.append([p])

    fit_rows = []
    fit_curves = []  # (x, y_fit) for plotting
    for group in groups:
        lo = group[0]["two_theta_ref_deg"] - half_window(group[0])
        hi = group[-1]["two_theta_ref_deg"] + half_window(group[-1])
        sel = (two_theta >= lo) & (two_theta <= hi)
        x, y = two_theta[sel], net[sel]

        p0, bounds_lo, bounds_hi = [], [], []
        for p in group:
            c0 = p["two_theta_detected_deg"]
            a0 = float(np.max(net_clipped[sel]))
            w0 = p["fwhm_empirical_deg"]
            w0 = w0 if np.isfinite(w0) and w0 > 0 else 0.55
            p0 += [c0, w0, 0.65, max(a0, 1.0)]
            bounds_lo += [c0 - 0.45, 0.05, 0.0, 0.0]
            bounds_hi += [c0 + 0.45, 3.0, 1.0, 10.0 * max(a0, 1.0)]
        # Small linear residual background only -- ALS has already removed the bulk.
        p0 += [0.0, 0.0]
        bounds_lo += [-1e3, -1e4]
        bounds_hi += [1e3, 1e4]

        try:
            popt, pcov = curve_fit(_multi_pv_with_linear_bg, x, y, p0=p0,
                                   bounds=(bounds_lo, bounds_hi), maxfev=40000)
            perr = np.sqrt(np.diag(pcov))
        except Exception as exc:                    # pragma: no cover
            print(f"    ! fit failed for group at {lo:.1f} deg: {exc}")
            continue

        y_fit = _multi_pv_with_linear_bg(x, *popt)
        fit_curves.append((x, y_fit))
        r2 = r_squared(y, y_fit)

        for i, p in enumerate(group):
            c, f, e, a = popt[4 * i:4 * i + 4]
            c_err, f_err = perr[4 * i], perr[4 * i + 1]
            fit_rows.append(dict(
                hkl=p["hkl"], h=p["h"], k=p["k"], l=p["l"],
                i_rel_ref=p["i_rel_ref"],
                two_theta_ref_deg=round(p["two_theta_ref_deg"], 4),
                two_theta_fit_deg=round(float(c), 4),
                two_theta_fit_err_deg=round(float(c_err), 5),
                fwhm_obs_deg=round(float(f), 5),
                fwhm_obs_err_deg=round(float(f_err), 5),
                eta_lorentzian_fraction=round(float(e), 4),
                amplitude_counts=round(float(a), 1),
                group_r_squared=round(r2, 5),
            ))

    fits = pd.DataFrame(fit_rows).sort_values("two_theta_fit_deg")
    fits = fits.reset_index(drop=True)

    # --- 6. d-spacing, lattice parameter, Scherrer ------------------------
    lam_a = cfg.WAVELENGTH_ANG
    theta_rad = np.radians(fits["two_theta_fit_deg"] / 2.0)
    fits["d_spacing_ang"] = (lam_a / (2.0 * np.sin(theta_rad))).round(5)
    fits["sqrt_hkl_sum"] = np.sqrt(fits["h"] ** 2 + fits["k"] ** 2
                                   + fits["l"] ** 2)
    fits["a_from_peak_ang"] = (fits["d_spacing_ang"]
                               * fits["sqrt_hkl_sum"]).round(5)

    instr = cfg.GROUND_TRUTH["xrd_instrumental_fwhm_deg"]
    fits["fwhm_corrected_deg"] = np.sqrt(
        np.clip(fits["fwhm_obs_deg"] ** 2 - instr ** 2, 0, None)).round(5)
    fits["scherrer_size_nm"] = [
        round(scherrer_size_nm(f, t), 2)
        for f, t in zip(fits["fwhm_corrected_deg"], fits["two_theta_fit_deg"])]

    # Relative integrated-style intensity, normalised to the strongest peak.
    fits["i_rel_measured"] = (100.0 * fits["amplitude_counts"]
                              * fits["fwhm_obs_deg"]).round(2)
    fits["i_rel_measured"] = (100.0 * fits["i_rel_measured"]
                              / fits["i_rel_measured"].max()).round(1)

    # Reflections trustworthy enough for size analysis. Two criteria, both
    # about whether the FITTED WIDTH is meaningful -- not about whether the
    # peak exists:
    #   (i)  reference intensity >= 8 % of the strongest reflection;
    #   (ii) relative uncertainty on the fitted FWHM below 8 %.
    # Very weak reflections are still indexed and reported, they simply
    # carry no usable width information.
    fits["fwhm_rel_uncertainty"] = (fits["fwhm_obs_err_deg"]
                                    / fits["fwhm_obs_deg"]).round(4)
    fits["used_for_size"] = ((fits["i_rel_ref"] >= 8.0)
                             & (fits["fwhm_rel_uncertainty"] <= 0.08))

    strong = fits[fits["used_for_size"]]
    scherrer_mean = float(strong["scherrer_size_nm"].mean())
    scherrer_sd = float(strong["scherrer_size_nm"].std(ddof=1))

    # --- 7. Williamson-Hall separation of size and strain ------------------
    #   beta*cos(theta) = K*lambda/D + 4*eps*sin(theta)
    #   y              = intercept   + slope * x
    #
    # Size broadening is isotropic in beta*cos(theta); strain broadening
    # grows as sin(theta). The intercept therefore isolates the size term and
    # the slope isolates the strain term. Attributing ALL broadening to size
    # (plain Scherrer) necessarily UNDER-estimates the crystallite size
    # whenever strain is present.
    #
    # The regression is weighted by 1/sigma(beta) from the profile fits, so
    # well-determined reflections dominate the slope.
    beta_rad = np.radians(strong["fwhm_corrected_deg"].to_numpy(float))
    beta_err_rad = np.radians(
        strong["fwhm_obs_err_deg"].to_numpy(float)).clip(min=1e-7)
    th = np.radians(strong["two_theta_fit_deg"].to_numpy(float) / 2.0)
    wh_x = 4.0 * np.sin(th)
    wh_y = beta_rad * np.cos(th)
    wh_w = 1.0 / beta_err_rad

    wh_coef, wh_cov = np.polyfit(wh_x, wh_y, 1, w=wh_w, cov=True)
    wh_slope, wh_intercept = float(wh_coef[0]), float(wh_coef[1])
    wh_slope_err = float(np.sqrt(wh_cov[0, 0]))
    wh_intercept_err = float(np.sqrt(wh_cov[1, 1]))

    wh_pred = wh_slope * wh_x + wh_intercept
    wh_r2 = r_squared(wh_y, wh_pred)

    wh_size_nm = float(cfg.SCHERRER_K * cfg.WAVELENGTH_NM / wh_intercept)
    wh_strain = wh_slope
    # Propagate the intercept uncertainty into the size uncertainty.
    wh_size_err = abs(wh_size_nm * wh_intercept_err / wh_intercept)

    @dataclass
    class _WHFit:
        slope: float
        intercept: float
        r2: float
    wh = _WHFit(wh_slope, wh_intercept, wh_r2)

    # --- 8. Lattice parameter by Nelson-Riley extrapolation ---------------
    #   Systematic errors (specimen displacement, absorption, zero offset)
    #   scale with NR(theta) = 0.5*(cos^2(t)/sin(t) + cos^2(t)/t) and vanish
    #   as theta -> 90 deg. Extrapolating a(hkl) against NR(theta) to NR = 0
    #   therefore removes them.
    th_all = np.radians(fits["two_theta_fit_deg"].to_numpy(float) / 2.0)
    nr = 0.5 * (np.cos(th_all) ** 2 / np.sin(th_all)
                + np.cos(th_all) ** 2 / th_all)
    fits["nelson_riley"] = np.round(nr, 5)

    # Only reflections whose POSITION is well determined may contribute.
    # A weak, partially overlapped reflection (here the 533/622 pair, which
    # is unresolved at this crystallite size) carries a large positional
    # uncertainty and would otherwise dominate the extrapolation.
    POSITION_ERR_LIMIT_DEG = 0.05
    fits["used_for_lattice"] = (fits["two_theta_fit_err_deg"]
                                <= POSITION_ERR_LIMIT_DEG)
    lat = fits[fits["used_for_lattice"]]
    n_excluded = int((~fits["used_for_lattice"]).sum())

    # Weighted least squares: weight each reflection by 1/sigma(2-theta).
    nr_x = lat["nelson_riley"].to_numpy(float)
    nr_y = lat["a_from_peak_ang"].to_numpy(float)
    nr_w = 1.0 / lat["two_theta_fit_err_deg"].to_numpy(float).clip(min=1e-6)
    nr_coef, nr_cov = np.polyfit(nr_x, nr_y, 1, w=nr_w, cov=True)
    a_refined = float(nr_coef[1])
    a_refined_err = float(np.sqrt(nr_cov[1, 1]))
    if n_excluded:
        print(f"  lattice refinement uses {len(lat)}/{len(fits)} reflections "
              f"({n_excluded} excluded: position uncertainty > "
              f"{POSITION_ERR_LIMIT_DEG} deg)")

    # --- 9. Phase discrimination ------------------------------------------
    d_magnetite = abs(a_refined - cfg.A_MAGNETITE_ANG)
    d_maghemite = abs(a_refined - cfg.A_MAGHEMITE_ANG)
    phase_call = ("Fe3O4 (magnetite)" if d_magnetite < d_maghemite
                  else "gamma-Fe2O3 (maghemite)")

    # Specific surface area implied by the crystallite size (spheres).
    ssa_m2_g = 6.0 / (RHO_MAGNETITE_G_CM3 * wh_size_nm * 1e-7) / 1e4

    print(f"  fitted {len(fits)} reflections (min R^2 = "
          f"{fits['group_r_squared'].min():.4f})")
    print(f"  Scherrer (mean of {len(strong)} strong peaks) = "
          f"{scherrer_mean:.2f} +/- {scherrer_sd:.2f} nm")
    print(f"  Williamson-Hall: D = {wh_size_nm:.2f} nm, "
          f"eps = {wh_strain:.2e}, R^2 = {wh.r2:.4f}")
    print(f"  Refined a = {a_refined:.4f} +/- {a_refined_err:.4f} A "
          f"-> {phase_call}")

    # --- 10. Save processed data ------------------------------------------
    processed = pd.DataFrame({
        "two_theta_deg": np.round(two_theta, 3),
        "intensity_raw_counts": counts,
        "background_counts": np.round(background, 2),
        "intensity_bg_subtracted_counts": np.round(net, 2),
        "intensity_smoothed_counts": np.round(smooth, 2),
    })
    save_processed(processed, "xrd_processed.csv",
                   "ALS background subtracted, Savitzky-Golay smoothed")
    save_processed(fits, "xrd_peak_analysis.csv",
                   "pseudo-Voigt peak fits, Scherrer sizes, d-spacings")

    # --- 11. Figures -------------------------------------------------------
    _plot_xrd_raw(two_theta, counts, background)
    _plot_xrd_processed(two_theta, net, smooth)
    _plot_xrd_labelled(two_theta, net, fits)
    _plot_xrd_fit_detail(two_theta, net, fit_curves, fits)
    _plot_williamson_hall(wh_x, wh_y, wh, strong)

    return {
        "n_points": int(len(two_theta)),
        "n_peaks_detected": int(len(found)),
        "n_reflections_indexed": int(n_matched),
        "n_reflections_expected": int(len(indexed)),
        "wavelength_ang": cfg.WAVELENGTH_ANG,
        "scherrer_K": cfg.SCHERRER_K,
        "instrumental_fwhm_deg": instr,
        "scherrer_mean_nm": round(scherrer_mean, 2),
        "scherrer_sd_nm": round(scherrer_sd, 2),
        "scherrer_per_peak": {r["hkl"]: r["scherrer_size_nm"]
                              for _, r in strong.iterrows()},
        "williamson_hall_size_nm": round(wh_size_nm, 2),
        "williamson_hall_size_err_nm": round(float(wh_size_err), 2),
        "williamson_hall_microstrain": float(f"{wh_strain:.3e}"),
        "williamson_hall_r2": round(float(wh.r2), 4),
        "williamson_hall_strain_err": float(f"{wh_slope_err:.2e}"),
        "n_reflections_used_for_size": int(len(strong)),
        "lattice_parameter_ang": round(a_refined, 4),
        "lattice_parameter_err_ang": round(a_refined_err, 4),
        "n_reflections_used_for_lattice": int(len(lat)),
        "phase_assignment": phase_call,
        "a_ref_magnetite_ang": cfg.A_MAGNETITE_ANG,
        "a_ref_maghemite_ang": cfg.A_MAGHEMITE_ANG,
        "implied_ssa_m2_per_g": round(float(ssa_m2_g), 1),
        "strongest_reflection": str(fits.loc[
            fits["i_rel_measured"].idxmax(), "hkl"]),
    }


def _plot_xrd_raw(tt, counts, background):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(tt, counts, color=cfg.COLORS["raw"], lw=0.7,
            label="Raw pattern (synthetic)")
    ax.plot(tt, background, color=cfg.COLORS["accent"], lw=1.4, ls="--",
            label="ALS background estimate")
    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Figure 1. Raw synthetic XRD pattern of Fe$_3$O$_4$ "
                 "with estimated background")
    ax.set_xlim(tt.min(), tt.max())
    ax.legend(loc="upper right")
    stamp(ax)
    save_figure(fig, "fig01_xrd_raw")


def _plot_xrd_processed(tt, net, smooth):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(tt, net, color=cfg.COLORS["raw"], lw=0.6,
            label="Background-subtracted")
    ax.plot(tt, smooth, color=cfg.COLORS["processed"], lw=1.2,
            label="Savitzky-Golay (25 pt, 3rd order)")
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Net intensity (counts)")
    ax.set_title("Figure 2. Background-subtracted synthetic XRD pattern")
    ax.set_xlim(tt.min(), tt.max())
    ax.legend(loc="upper right")
    stamp(ax)
    save_figure(fig, "fig02_xrd_processed")


def _plot_xrd_labelled(tt, net, fits):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tt, net, color=cfg.COLORS["processed"], lw=0.9)
    ymax = float(np.max(net))
    for _, r in fits.iterrows():
        x = r["two_theta_fit_deg"]
        y = r["amplitude_counts"]
        ax.annotate(r["hkl"], xy=(x, y), xytext=(x, y + 0.085 * ymax),
                    ha="center", fontsize=8.5, color=cfg.COLORS["accent"],
                    arrowprops=dict(arrowstyle="-", lw=0.6,
                                    color=cfg.COLORS["accent"]))
    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Net intensity (counts)")
    ax.set_title("Figure 3. Indexed synthetic XRD pattern "
                 "(cubic Fe$_3$O$_4$, $Fd\\bar{3}m$)")
    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(-0.03 * ymax, 1.22 * ymax)
    ax.text(0.015, 0.95, "Cu K$\\alpha_1$, $\\lambda$ = 1.54060 $\\AA$",
            transform=ax.transAxes, va="top", fontsize=9)
    stamp(ax)
    save_figure(fig, "fig03_xrd_indexed")


def _plot_xrd_fit_detail(tt, net, fit_curves, fits):
    """Zoom on the (311) reflection: data, pseudo-Voigt fit, residuals.

    Plotted on the background-subtracted pattern, which is what was fitted.
    """
    target = 3, 1, 1
    row = fits[(fits.h == target[0]) & (fits.k == target[1])
               & (fits.l == target[2])].iloc[0]
    centre = row["two_theta_fit_deg"]

    best = min(fit_curves, key=lambda c: abs(np.mean(c[0]) - centre))
    x_fit, y_fit = best
    sel = (tt >= x_fit.min()) & (tt <= x_fit.max())
    x, y = tt[sel], net[sel]
    resid = y - np.interp(x, x_fit, y_fit)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.0, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[3.1, 1.0], hspace=0.08))
    ax.plot(x, y, "o", ms=2.4, color=cfg.COLORS["raw"],
            label="Synthetic data")
    ax.plot(x_fit, y_fit, "-", color=cfg.COLORS["fit"], lw=1.6,
            label="Pseudo-Voigt + linear residual")
    ax.set_ylabel("Net intensity (counts)")
    ax.set_title("Figure 4. Pseudo-Voigt profile fit to the (311) reflection")
    ax.legend(loc="upper right")
    ax.text(0.03, 0.93,
            f"2$\\theta$ = {row['two_theta_fit_deg']:.3f}$\\degree$\n"
            f"FWHM$_{{obs}}$ = {row['fwhm_obs_deg']:.3f}$\\degree$\n"
            f"$\\eta$ = {row['eta_lorentzian_fraction']:.2f}   "
            f"R$^2$ = {row['group_r_squared']:.4f}",
            transform=ax.transAxes, va="top", fontsize=8.5)

    axr.plot(x, resid, lw=0.8, color=cfg.COLORS["neutral"])
    axr.axhline(0, color=cfg.COLORS["accent"], lw=0.8)
    axr.set_xlabel(r"2$\theta$ (degrees)")
    axr.set_ylabel("Residual")
    stamp(axr, loc="lower left")
    save_figure(fig, "fig04_xrd_peak_fit")


def _plot_williamson_hall(x, y, fit, strong):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(x, y * 1e3, "o", ms=6, color=cfg.COLORS["processed"],
            label="Fitted reflections")
    xx = np.linspace(0, x.max() * 1.08, 100)
    ax.plot(xx, (fit.intercept + fit.slope * xx) * 1e3, "-",
            color=cfg.COLORS["accent"], lw=1.4,
            label=f"Weighted linear fit (R$^2$ = {fit.r2:.3f})")
    for xi, yi, hkl in zip(x, y * 1e3, strong["hkl"]):
        ax.annotate(hkl, (xi, yi), textcoords="offset points",
                    xytext=(5, -9), fontsize=7.5, color="#555555")
    ax.set_xlabel(r"4 sin$\theta$")
    ax.set_ylabel(r"$\beta$ cos$\theta$  ($\times 10^{-3}$ rad)")
    ax.set_title("Figure 5. Williamson-Hall analysis")
    D = cfg.SCHERRER_K * cfg.WAVELENGTH_NM / fit.intercept
    ax.text(0.03, 0.95,
            f"intercept = K$\\lambda$/D  $\\Rightarrow$  D = {D:.1f} nm\n"
            f"slope = 4$\\varepsilon$  $\\Rightarrow$  "
            f"$\\varepsilon$ = {fit.slope:.2e}",
            transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set_xlim(0, x.max() * 1.08)
    ax.legend(loc="lower right")
    stamp(ax, loc="upper right")
    save_figure(fig, "fig05_williamson_hall")


# ==========================================================================
# SECTION 5 -- FTIR
# ==========================================================================
# Literature band assignments used to INTERPRET the detected peaks.
# Sources: Cornell & Schwertmann (2003); Stuart (2004); Gotic & Music (2007).
FTIR_ASSIGNMENTS = [
    dict(low=3000, high=3700, label=r"$\nu$(O-H) stretch",
         assignment="O-H stretch: adsorbed molecular water + surface hydroxyls",
         confidence="high",
         note="Very broad -> extensive hydrogen bonding. Cannot distinguish "
              "H2O from structural OH on band position alone."),
    dict(low=2300, high=2380, label=r"CO$_2$ (atmospheric)",
         assignment="Atmospheric CO2 asymmetric stretch -- INSTRUMENT ARTEFACT",
         confidence="high",
         note="Not a sample feature. Arises from incomplete background "
              "compensation of ambient CO2."),
    dict(low=1580, high=1680, label=r"$\delta$(H-O-H) bend",
         assignment="H-O-H bending of molecular water",
         confidence="high",
         note="Co-occurrence with the 3400 band supports molecular water "
              "rather than isolated OH groups."),
    dict(low=950, high=1150, label="surface Fe-OH",
         assignment="Tentative: Fe-OH deformation / surface hydroxyl",
         confidence="low",
         note="Weak and broad. Overlaps the region where residual sulfate or "
              "silicate would also absorb. NOT a confident assignment."),
    dict(low=520, high=640, label=r"$\nu$(Fe-O) T$_d$",
         assignment="Fe-O stretch, tetrahedrally coordinated Fe",
         confidence="high",
         note="Principal spinel lattice mode. Position is phase-sensitive but "
              "NOT sufficient alone to separate Fe3O4 from gamma-Fe2O3."),
    dict(low=400, high=500, label=r"$\nu$(Fe-O) O$_h$",
         assignment="Fe-O stretch, octahedrally coordinated Fe",
         confidence="medium",
         note="Close to the low-wavenumber cut-off; intensity and position "
              "are less reliable here."),
]


def _analyse_ftir_impl() -> dict:
    print("\n[FTIR] -----------------------------------------------------")
    raw = load_raw("ftir_data.csv")
    # Work in ASCENDING wavenumber internally; display descending.
    df = raw.sort_values("wavenumber_cm-1").reset_index(drop=True)
    wn = df["wavenumber_cm-1"].to_numpy(float)
    absorb_raw = df["absorbance"].to_numpy(float)
    trans_raw = df["transmittance_percent"].to_numpy(float)
    print(f"  loaded {len(wn)} points, {wn.max():.0f}-{wn.min():.0f} cm-1")

    # --- Baseline correction ---------------------------------------------
    # Justification: the raw spectrum carries a smooth, monotonic offset from
    # particle scattering in the ATR/powder sampling geometry. A stiff ALS
    # baseline removes it without distorting bands, which are far narrower
    # than the baseline's curvature.
    baseline = baseline_als(absorb_raw, lam=1e7, p=0.001, n_iter=25)
    absorb_corr = absorb_raw - baseline

    # --- Smoothing (justified, and deliberately gentle) -------------------
    # The narrowest genuine feature is the ~9 cm-1 CO2 line. At 1 cm-1
    # sampling an 11-point, 3rd-order Savitzky-Golay filter has an effective
    # width well below that, so band shapes and areas are preserved while
    # high-frequency detector noise is suppressed.
    absorb_smooth = savgol_filter(absorb_corr, window_length=11, polyorder=3)

    # --- Peak detection ---------------------------------------------------
    noise = float(np.std(absorb_corr[(wn > 3750) & (wn < 3950)]))
    peaks, props = find_peaks(absorb_smooth,
                              height=3.0 * noise,
                              prominence=4.0 * noise,
                              distance=12)
    print(f"  detected {len(peaks)} bands (noise sigma = {noise:.5f} A)")

    rows = []
    for idx in peaks:
        centre = float(wn[idx])
        height = float(absorb_smooth[idx])
        match = next((a for a in FTIR_ASSIGNMENTS
                      if a["low"] <= centre <= a["high"]), None)
        rows.append(dict(
            peak_wavenumber_cm_1=round(centre, 1),
            peak_absorbance_au=round(height, 5),
            transmittance_percent=round(float(
                100.0 * 10 ** (-absorb_raw[idx])), 2),
            assignment=(match["assignment"] if match else
                        "UNASSIGNED -- no defensible assignment"),
            confidence=(match["confidence"] if match else "none"),
            label=(match["label"] if match else ""),
            interpretation_note=(match["note"] if match else
                                 "Detected but not assigned. Reporting an "
                                 "assignment here would not be defensible."),
        ))
    peak_table = pd.DataFrame(rows).sort_values(
        "peak_wavenumber_cm_1", ascending=False).reset_index(drop=True)

    # Which diagnostic bands are ABSENT? Absence is evidence too -- but it
    # has to be tested properly. Simply asking whether the absorbance in a
    # window is small is wrong: the tail of a strong neighbouring band (e.g.
    # the 1628 cm-1 water bend bleeding into the 1700 cm-1 carbonyl window)
    # will fail that test and fake a detection.
    #
    # The defensible test is whether a RESOLVED LOCAL MAXIMUM exists in the
    # window, i.e. whether the peak finder placed a band there at all.
    detected_centres = wn[peaks]

    def band_absent(lo, hi):
        """True if no resolved band maximum lies in [lo, hi]."""
        return bool(not np.any((detected_centres >= lo)
                               & (detected_centres <= hi)))

    absence_checks = {
        "goethite_alpha_FeOOH_890_cm-1": band_absent(875, 905),
        "goethite_alpha_FeOOH_795_cm-1": band_absent(780, 810),
        "organic_C-H_stretch_2850_2960_cm-1": band_absent(2840, 2970),
        "carboxylate_C=O_1700_cm-1": band_absent(1680, 1730),
        "nitrate_1384_cm-1": band_absent(1370, 1400),
        "sulfate_1100_cm-1": band_absent(1080, 1140),
    }
    for name, absent in absence_checks.items():
        print(f"    {'absent ' if absent else 'PRESENT'}: {name}")

    processed = pd.DataFrame({
        "wavenumber_cm-1": df["wavenumber_cm-1"],
        "absorbance_raw": np.round(absorb_raw, 6),
        "baseline": np.round(baseline, 6),
        "absorbance_baseline_corrected": np.round(absorb_corr, 6),
        "absorbance_smoothed": np.round(absorb_smooth, 6),
        "transmittance_raw_percent": np.round(trans_raw, 4),
    }).sort_values("wavenumber_cm-1", ascending=False)
    save_processed(processed, "ftir_processed.csv",
                   "ALS baseline corrected + Savitzky-Golay smoothed")
    save_processed(peak_table, "ftir_peak_assignments.csv",
                   "detected bands with literature-based assignments")

    _plot_ftir_raw(wn, trans_raw, absorb_raw, baseline)
    _plot_ftir_processed(wn, absorb_corr, absorb_smooth)
    _plot_ftir_labelled(wn, absorb_smooth, peak_table)

    return {
        "n_points": int(len(wn)),
        "n_bands_detected": int(len(peaks)),
        "noise_sigma_absorbance": round(noise, 6),
        "bands": peak_table[["peak_wavenumber_cm_1", "peak_absorbance_au",
                             "assignment", "confidence"]].to_dict("records"),
        "absence_checks": absence_checks,
        "smoothing": "Savitzky-Golay, window 11 pts (11 cm-1), polyorder 3",
        "baseline_method": "Asymmetric least squares, lam=1e7, p=0.001",
    }


def _plot_ftir_raw(wn, trans, absorb, baseline):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.12))
    ax1.plot(wn, trans, color=cfg.COLORS["raw"], lw=0.9)
    ax1.set_ylabel("Transmittance (%)")
    ax1.set_title("Figure 6. Raw synthetic FTIR spectrum "
                  "(transmittance and absorbance)")

    ax2.plot(wn, absorb, color=cfg.COLORS["raw"], lw=0.9, label="Raw absorbance")
    ax2.plot(wn, baseline, color=cfg.COLORS["accent"], lw=1.3, ls="--",
             label="ALS baseline")
    ax2.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax2.set_ylabel("Absorbance (a.u.)")
    ax2.legend(loc="upper right")
    ax2.set_xlim(4000, 400)
    stamp(ax2)
    save_figure(fig, "fig06_ftir_raw")


def _plot_ftir_processed(wn, corr, smooth):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(wn, corr, color=cfg.COLORS["raw"], lw=0.7,
            label="Baseline-corrected")
    ax.plot(wn, smooth, color=cfg.COLORS["processed"], lw=1.1,
            label="Savitzky-Golay (11 pt, 3rd order)")
    ax.axhline(0, color="#CCCCCC", lw=0.6)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Figure 7. Baseline-corrected synthetic FTIR spectrum")
    ax.set_xlim(4000, 400)
    ax.legend(loc="upper right")
    stamp(ax)
    save_figure(fig, "fig07_ftir_processed")


def _plot_ftir_labelled(wn, smooth, peak_table):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(wn, smooth, color=cfg.COLORS["processed"], lw=1.1)
    ymax = float(np.max(smooth))
    for _, r in peak_table.iterrows():
        x = r["peak_wavenumber_cm_1"]
        y = r["peak_absorbance_au"]
        col = (cfg.COLORS["accent"] if r["confidence"] in ("high", "medium")
               else "#9A6C00")
        if "ARTEFACT" in r["assignment"]:
            col = "#7A7A7A"
        ax.annotate(f"{x:.0f}", xy=(x, y), xytext=(x, y + 0.055 * ymax),
                    ha="center", fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Figure 8. Band-labelled synthetic FTIR spectrum")
    ax.set_xlim(4000, 400)
    ax.set_ylim(-0.02 * ymax, 1.30 * ymax)

    handles = [
        plt.Line2D([], [], color=cfg.COLORS["accent"], lw=2,
                   label="Confident assignment"),
        plt.Line2D([], [], color="#9A6C00", lw=2,
                   label="Tentative / low confidence"),
        plt.Line2D([], [], color="#7A7A7A", lw=2,
                   label="Atmospheric artefact"),
    ]
    ax.legend(handles=handles, loc="center left", fontsize=8)
    stamp(ax)
    save_figure(fig, "fig08_ftir_labelled")


# ==========================================================================
# SECTION 6 -- UV-Vis
# ==========================================================================
def _analyse_uvvis_impl() -> dict:
    print("\n[UV-Vis] ---------------------------------------------------")
    raw = load_raw("uvvis_data.csv")
    wl = raw["wavelength_nm"].to_numpy(float)
    A_raw = raw["absorbance"].to_numpy(float)
    print(f"  loaded {len(wl)} points, {wl.min():.0f}-{wl.max():.0f} nm")

    # --- Photometric validity ---------------------------------------------
    # Absorbance > ~2.5-3 is outside the reliable range of a typical
    # double-beam spectrophotometer (stray light dominates). Flag, don't fit.
    A_LIMIT = 2.5
    unreliable = A_raw > A_LIMIT
    n_unreliable = int(unreliable.sum())
    valid_lo_nm = float(wl[~unreliable].min())
    print(f"  {n_unreliable} points exceed A = {A_LIMIT} "
          f"(photometrically unreliable below {valid_lo_nm:.0f} nm)")

    # --- Smoothing (gentle; the spectrum has no narrow features) -----------
    A_smooth = savgol_filter(A_raw, window_length=15, polyorder=3)

    # --- Scattering / turbidity correction ---------------------------------
    # For a DISPERSION, measured "absorbance" = true absorption + turbidity.
    # Rayleigh/Mie turbidity follows A_scatter = k * lambda^-n. Fit that power
    # law in a long-wavelength window where genuine absorption is weakest,
    # then extrapolate and subtract.
    # NOTE: this returns an EFFECTIVE exponent, not a pure scattering law.
    # Genuine absorption persists at 780-900 nm, so the fitted power law
    # unavoidably absorbs some of it. Treat the corrected curve as
    # "turbidity-reduced", not "scattering-free".
    fit_win = (wl >= 780) & (wl <= 900)
    slope_s, intercept_s, r_s, _, _ = stats.linregress(
        np.log(wl[fit_win]), np.log(np.clip(A_smooth[fit_win], 1e-6, None)))
    n_scatter = float(-slope_s)
    A_scatter = np.exp(intercept_s) * wl ** slope_s
    A_corr = A_smooth - A_scatter
    print(f"  empirical turbidity power law: A ~ lambda^-{n_scatter:.2f} "
          f"(log-log R^2 = {r_s**2:.4f}); effective exponent, not a pure "
          f"scattering law")

    # --- Absorption regions ------------------------------------------------
    A600 = float(np.interp(600, wl, A_smooth))
    A400 = float(np.interp(400, wl, A_smooth))
    frac_scatter_600 = float(np.interp(600, wl, A_scatter) / A600)

    # --- Tauc analysis: performed as a CAUTIONARY DEMONSTRATION -------------
    # Tauc:  (alpha * h*nu)^(1/r) = B (h*nu - Eg)
    #   r = 1/2 -> direct allowed;  r = 2 -> indirect allowed.
    # Equivalently we plot (alpha h nu)^m with m = 2 (direct) or m = 1/2
    # (indirect) and extrapolate the linear region to the energy axis.
    #
    # alpha: for a dispersion we only have absorbance A, and
    #        alpha = 2.303 * A / L  ONLY if scattering is negligible and the
    #        path length L is known. Neither holds cleanly here, so the
    #        resulting "Eg" is an apparent, not an intrinsic, quantity.
    hv = PLANCK_EV_NM / wl                      # photon energy, eV
    tauc = {}
    for label, A_used in (("uncorrected", A_smooth),
                          ("scattering_corrected", A_corr)):
        for m, transition in ((2.0, "direct_allowed"),
                              (0.5, "indirect_allowed")):
            y = np.power(np.clip(A_used, 1e-9, None) * hv, m)
            # Restrict to the photometrically valid range.
            ok = (~unreliable) & (hv > 1.6) & (hv < 3.4)
            res = _tauc_linear_region(hv[ok], y[ok])
            tauc[f"{label}__{transition}"] = res

    gaps = [v["Eg_eV"] for v in tauc.values() if np.isfinite(v["Eg_eV"])]
    gap_spread = float(max(gaps) - min(gaps)) if gaps else np.nan
    # Widest span over BOTH the four model choices and, within each choice,
    # every window an analyst could defend on R^2 grounds.
    all_lo = min(v["Eg_range_eV"][0] for v in tauc.values()
                 if np.isfinite(v["Eg_eV"]))
    all_hi = max(v["Eg_range_eV"][1] for v in tauc.values()
                 if np.isfinite(v["Eg_eV"]))
    total_spread = float(all_hi - all_lo)
    print(f"  Tauc extrapolations span {min(gaps):.2f}-{max(gaps):.2f} eV "
          f"(spread {gap_spread:.2f} eV) across 4 model choices")
    print(f"  including linear-window choice, the full defensible span is "
          f"{all_lo:.2f}-{all_hi:.2f} eV ({total_spread:.2f} eV)")

    verdict = (
        "NOT REPORTED AS A BAND GAP. Fe3O4 is a mixed-valence, "
        "near-metallic conductor above the Verwey transition, not a wide-gap "
        "semiconductor; and the spectrum is a scattering-contaminated "
        "dispersion measurement. The four equally defensible analysis choices "
        f"below give values spanning {gap_spread:.2f} eV, and allowing for "
        f"the choice of linear window the full defensible span is "
        f"{all_lo:.2f}-{all_hi:.2f} eV. That is larger "
        "than any difference one would be trying to detect. Tauc analysis is "
        "shown here to demonstrate the method and its failure mode, not to "
        "extract a physical constant."
    )

    processed = pd.DataFrame({
        "wavelength_nm": wl,
        "photon_energy_eV": np.round(hv, 5),
        "absorbance_raw": np.round(A_raw, 5),
        "absorbance_smoothed": np.round(A_smooth, 5),
        "scattering_baseline": np.round(A_scatter, 5),
        "absorbance_scatter_corrected": np.round(A_corr, 5),
        "photometrically_unreliable": unreliable,
    })
    save_processed(processed, "uvvis_processed.csv",
                   "smoothed, turbidity-corrected, photometric flags")

    _plot_uvvis(wl, A_raw, A_smooth, A_scatter, A_corr, unreliable, A_LIMIT)
    _plot_tauc(hv, A_smooth, A_corr, unreliable, tauc)

    return {
        "n_points": int(len(wl)),
        "range_nm": [float(wl.min()), float(wl.max())],
        "photometric_limit_A": A_LIMIT,
        "n_points_above_limit": n_unreliable,
        "reliable_above_nm": round(valid_lo_nm, 0),
        "absorbance_at_400nm": round(A400, 3),
        "absorbance_at_600nm": round(A600, 3),
        "scattering_exponent_n_effective": round(n_scatter, 2),
        "scattering_exponent_note": (
            "Effective exponent from an empirical lambda^-n fit over 780-900 nm. "
            "Real absorption persists in that window, so this is not a pure "
            "Rayleigh/Mie exponent and the corrected spectrum is turbidity-"
            "reduced rather than scattering-free."),
        "scattering_fraction_at_600nm": round(frac_scatter_600, 3),
        "absorption_character": (
            "Broad, monotonically rising absorption from the NIR into the UV "
            "with no discrete excitonic maximum -- consistent with O(2p)->Fe(3d) "
            "charge transfer plus a strong sub-edge (Urbach) tail. This is why "
            "magnetite appears black."),
        "tauc_results": tauc,
        "tauc_spread_eV": round(gap_spread, 2),
        "tauc_full_defensible_range_eV": [round(all_lo, 2), round(all_hi, 2)],
        "tauc_full_spread_eV": round(total_spread, 2),
        "band_gap_verdict": verdict,
    }


def _tauc_linear_region(hv: np.ndarray, y: np.ndarray,
                        window_frac: float = 0.16,
                        edge_threshold: float = 0.05,
                        r2_accept: float = 0.995) -> dict:
    """Extrapolate the linear region of a Tauc curve, with a sensitivity scan.

    A real analyst selects the straight portion of the ABSORPTION EDGE by
    eye. Two things are automated here:

      1. Candidate windows are restricted to the rising edge, defined as
         y >= edge_threshold * max(y). Without this, the flattest part of
         the sub-edge baseline wins on R^2 while being physically
         meaningless.
      2. Rather than reporting only the single best window, EVERY window
         that a reasonable analyst could defend (R^2 >= r2_accept) is
         collected. The spread of those intercepts is the honest measure of
         how well the "band gap" is actually determined.

    Returns the best-R^2 result plus the full spread over acceptable windows.
    """
    order = np.argsort(hv)
    hv, y = hv[order], y[order]
    n = len(hv)

    y_max = float(np.max(y))
    edge = y >= edge_threshold * y_max

    best = dict(r2=-np.inf, Eg_eV=np.nan, slope=np.nan,
                window_eV=[np.nan, np.nan])
    acceptable: list[float] = []

    # The LENGTH of the fitted window is itself an analyst choice, so scan
    # over a range of plausible lengths as well as positions.
    for frac in (0.6 * window_frac, window_frac, 1.5 * window_frac):
        w = max(10, int(frac * n))
        if w >= n:
            continue
        for start in range(0, n - w):
            sl = slice(start, start + w)
            # require the window to sit on the rising edge
            if not edge[sl].all():
                continue
            xs, ys = hv[sl], y[sl]
            if np.ptp(ys) <= 0:
                continue
            lr = stats.linregress(xs, ys)
            if lr.slope <= 0:
                continue
            r2 = float(lr.rvalue ** 2)
            eg = float(-lr.intercept / lr.slope)
            if r2 >= r2_accept and np.isfinite(eg):
                acceptable.append(eg)
            if r2 > best["r2"]:
                best = dict(r2=r2, Eg_eV=eg, slope=float(lr.slope),
                            window_eV=[float(xs.min()), float(xs.max())])

    if not np.isfinite(best["Eg_eV"]):
        return dict(r2=np.nan, Eg_eV=np.nan, slope=np.nan,
                    window_eV=[np.nan, np.nan], n_acceptable_windows=0,
                    Eg_range_eV=[np.nan, np.nan], Eg_spread_eV=np.nan)

    out = dict(
        r2=round(best["r2"], 5),
        Eg_eV=round(best["Eg_eV"], 3),
        slope=best["slope"],
        window_eV=[round(v, 3) for v in best["window_eV"]],
        n_acceptable_windows=len(acceptable),
    )
    if acceptable:
        out["Eg_range_eV"] = [round(float(np.min(acceptable)), 3),
                              round(float(np.max(acceptable)), 3)]
        out["Eg_spread_eV"] = round(float(np.ptp(acceptable)), 3)
        out["Eg_median_eV"] = round(float(np.median(acceptable)), 3)
    else:
        out["Eg_range_eV"] = [out["Eg_eV"], out["Eg_eV"]]
        out["Eg_spread_eV"] = 0.0
        out["Eg_median_eV"] = out["Eg_eV"]
    return out


def _plot_uvvis(wl, A_raw, A_smooth, A_scatter, A_corr, unreliable, limit):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(wl, A_raw, color=cfg.COLORS["raw"], lw=0.7, label="Raw absorbance")
    ax.plot(wl, A_smooth, color=cfg.COLORS["processed"], lw=1.3,
            label="Smoothed (SG 15 pt)")
    ax.plot(wl, A_scatter, color=cfg.COLORS["background"], lw=1.2, ls="--",
            label=r"Turbidity baseline $\propto \lambda^{-n}$")
    ax.plot(wl, A_corr, color=cfg.COLORS["accent"], lw=1.3,
            label="Scattering-corrected")

    if unreliable.any():
        ax.axvspan(wl.min(), wl[~unreliable].min(), color="#D9534F",
                   alpha=0.10)
        ax.text(wl[~unreliable].min() - 6, ax.get_ylim()[1] * 0.55,
                f"A > {limit}\nphotometrically\nunreliable", fontsize=7.5,
                ha="right", va="center", color="#A03030")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Figure 9. Synthetic UV-Vis spectrum of an aqueous "
                 "Fe$_3$O$_4$ dispersion")
    ax.set_xlim(wl.min(), wl.max())
    ax.legend(loc="upper right")
    stamp(ax)
    save_figure(fig, "fig09_uvvis_spectrum")


def _plot_tauc(hv, A_smooth, A_corr, unreliable, tauc):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    combos = [
        (axes[0], 2.0, "direct_allowed",
         r"$(\alpha h\nu)^{2}$", "Direct-allowed ($r$ = 1/2)"),
        (axes[1], 0.5, "indirect_allowed",
         r"$(\alpha h\nu)^{1/2}$", "Indirect-allowed ($r$ = 2)"),
    ]
    for ax, m, transition, ylab, title in combos:
        for A_used, label, colour in (
                (A_smooth, "uncorrected", cfg.COLORS["raw"]),
                (A_corr, "scattering_corrected", cfg.COLORS["processed"])):
            y = np.power(np.clip(A_used, 1e-9, None) * hv, m)
            ok = (~unreliable) & (hv > 1.6) & (hv < 3.4)
            ax.plot(hv[ok], y[ok], lw=1.3, color=colour,
                    label=label.replace("_", " "))
            res = tauc[f"{label}__{transition}"]
            if np.isfinite(res["Eg_eV"]):
                xs = np.linspace(res["Eg_eV"],
                                 res["window_eV"][1] + 0.15, 40)
                ax.plot(xs, res["slope"] * (xs - res["Eg_eV"]), "--",
                        lw=1.4, color=colour, alpha=0.95, zorder=5)
                ax.plot([res["Eg_eV"]], [0], "v", ms=7, color=colour,
                        markeredgecolor="white", markeredgewidth=0.7,
                        zorder=6, clip_on=False)
                ax.annotate(f"{res['Eg_eV']:.2f} eV",
                            (res["Eg_eV"], 0), textcoords="offset points",
                            xytext=(0, 11), ha="center", fontsize=8,
                            color=colour)
        ax.set_xlabel(r"Photon energy $h\nu$ (eV)")
        ax.set_ylabel(ylab + r"  (a.u.)")
        ax.set_title(title, fontsize=10)
        egs = [tauc[f"{lab}__{transition}"]["Eg_eV"]
               for lab in ("uncorrected", "scattering_corrected")
               if np.isfinite(tauc[f"{lab}__{transition}"]["Eg_eV"])]
        ax.set_xlim(min([1.6] + egs) - 0.15, 3.45)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 10. Tauc analysis shown as a METHOD DEMONSTRATION -- "
                 "the extracted values are not a band gap of Fe$_3$O$_4$",
                 fontsize=10.5, y=1.005)
    stamp(axes[1], loc="lower right")
    save_figure(fig, "fig10_tauc_demonstration")


# ==========================================================================
# SECTION 7 -- SEM particle-size statistics
# ==========================================================================
def _analyse_sem_impl() -> dict:
    print("\n[SEM] ------------------------------------------------------")
    df = load_raw("sem_particle_measurements.csv")
    d = df["equivalent_diameter_nm"].to_numpy(float)
    n = len(d)
    print(f"  loaded {n} particle measurements")

    q1, med, q3 = np.percentile(d, [25, 50, 75])
    iqr = q3 - q1
    desc = {
        "n": int(n),
        "mean_nm": round(float(np.mean(d)), 2),
        "median_nm": round(float(med), 2),
        "std_dev_nm": round(float(np.std(d, ddof=1)), 2),
        "std_error_nm": round(float(np.std(d, ddof=1) / np.sqrt(n)), 3),
        "min_nm": round(float(np.min(d)), 2),
        "max_nm": round(float(np.max(d)), 2),
        "q1_nm": round(float(q1), 2),
        "q3_nm": round(float(q3), 2),
        "iqr_nm": round(float(iqr), 2),
        "range_nm": round(float(np.ptp(d)), 2),
        "cv_percent": round(float(100 * np.std(d, ddof=1) / np.mean(d)), 1),
        "skewness": round(float(stats.skew(d)), 3),
        "kurtosis_excess": round(float(stats.kurtosis(d)), 3),
    }

    # --- Outliers by the 1.5 x IQR (Tukey) rule ---------------------------
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(d < lo_f) | (d > hi_f)]
    desc["tukey_fence_low_nm"] = round(float(lo_f), 2)
    desc["tukey_fence_high_nm"] = round(float(hi_f), 2)
    desc["n_outliers_tukey"] = int(len(outliers))

    # --- Distribution model ------------------------------------------------
    # Lognormal is the standard model for nucleation-and-growth populations.
    shape, loc, scale = stats.lognorm.fit(d, floc=0)
    ks = stats.kstest(d, "lognorm", args=(shape, loc, scale))
    # Compare against a normal model for context.
    mu_n, sd_n = stats.norm.fit(d)
    ks_norm = stats.kstest(d, "norm", args=(mu_n, sd_n))
    print(f"  lognormal fit: median = {scale:.2f} nm, GSD = "
          f"{np.exp(shape):.3f}, KS p = {ks.pvalue:.3f}")
    print(f"  normal fit comparison: KS p = {ks_norm.pvalue:.3f}")

    # --- Bootstrap CI on the mean (no normality assumption) ---------------
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    boots = rng.choice(d, size=(10000, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    # --- Number- vs volume-weighting --------------------------------------
    # XRD size is VOLUME-weighted; a number-weighted SEM mean is not the same
    # statistic. The volume-weighted (De Brouckere) mean diameter is
    #        D[4,3] = sum(d^4) / sum(d^3)
    # and the surface-weighted (Sauter) mean is D[3,2] = sum(d^3)/sum(d^2).
    d43 = float(np.sum(d ** 4) / np.sum(d ** 3))
    d32 = float(np.sum(d ** 3) / np.sum(d ** 2))

    stats_out = {
        **desc,
        "lognormal_median_nm": round(float(scale), 2),
        "lognormal_gsd": round(float(np.exp(shape)), 3),
        "lognormal_sigma": round(float(shape), 4),
        "ks_lognormal_statistic": round(float(ks.statistic), 4),
        "ks_lognormal_pvalue": round(float(ks.pvalue), 4),
        "ks_normal_pvalue": round(float(ks_norm.pvalue), 4),
        "bootstrap_mean_ci95_nm": [round(float(ci_lo), 2),
                                   round(float(ci_hi), 2)],
        "volume_weighted_D43_nm": round(d43, 2),
        "surface_weighted_D32_nm": round(d32, 2),
        "mean_aspect_ratio": round(float(df["aspect_ratio"].mean()), 3),
        "mean_circularity": round(float(df["circularity"].mean()), 3),
        "aspect_ratio_range": [round(float(df["aspect_ratio"].min()), 3),
                               round(float(df["aspect_ratio"].max()), 3)],
    }

    summary_df = pd.DataFrame(
        [(k, v) for k, v in stats_out.items()],
        columns=["statistic", "value"])
    save_processed(summary_df, "sem_size_statistics.csv",
                   "descriptive + distribution statistics")

    binned = _size_distribution_table(d)
    save_processed(binned, "sem_size_distribution.csv",
                   "binned number and volume frequency distribution")

    _plot_sem_histogram(d, shape, loc, scale, stats_out)
    _plot_sem_boxplot(df, d, stats_out)
    _plot_sem_distribution(d, binned, stats_out)

    return stats_out


def _size_distribution_table(d: np.ndarray) -> pd.DataFrame:
    """Binned number- and volume-weighted frequency table."""
    bins = np.histogram_bin_edges(d, bins="fd")
    counts, edges = np.histogram(d, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Volume weighting: each particle contributes ~ d^3.
    vol, _ = np.histogram(d, bins=bins, weights=d ** 3)
    return pd.DataFrame({
        "bin_lower_nm": np.round(edges[:-1], 3),
        "bin_upper_nm": np.round(edges[1:], 3),
        "bin_centre_nm": np.round(centres, 3),
        "count": counts,
        "number_fraction_percent": np.round(100 * counts / counts.sum(), 3),
        "cumulative_number_percent": np.round(
            100 * np.cumsum(counts) / counts.sum(), 3),
        "volume_fraction_percent": np.round(100 * vol / vol.sum(), 3),
    })


def _plot_sem_histogram(d, shape, loc, scale, s):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    counts, bins, _ = ax.hist(d, bins="fd", color="#B8C4D4",
                              edgecolor=cfg.COLORS["processed"], lw=0.8,
                              density=True, label="Measured features")
    xs = np.linspace(d.min() * 0.85, d.max() * 1.12, 400)
    ax.plot(xs, stats.lognorm.pdf(xs, shape, loc, scale),
            color=cfg.COLORS["accent"], lw=1.8,
            label=(f"Lognormal fit\nmedian = {scale:.1f} nm, "
                   f"GSD = {np.exp(shape):.2f}"))
    ax.axvline(s["mean_nm"], color=cfg.COLORS["fit"], ls="--", lw=1.3,
               label=f"Mean = {s['mean_nm']:.1f} nm")
    ax.axvline(s["median_nm"], color=cfg.COLORS["highlight"], ls=":", lw=1.5,
               label=f"Median = {s['median_nm']:.1f} nm")
    ax.set_xlabel("Equivalent circular diameter (nm)")
    ax.set_ylabel("Probability density (nm$^{-1}$)")
    ax.set_title(f"Figure 11. Particle-size distribution (n = {s['n']})")
    ax.legend(loc="upper right", fontsize=8)
    stamp(ax, loc="lower left")
    save_figure(fig, "fig11_sem_histogram")


def _plot_sem_boxplot(df, d, s):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.0))
    specs = [
        (axes[0], d, "Equivalent diameter (nm)", "#B8C4D4"),
        (axes[1], df["aspect_ratio"].to_numpy(float), "Aspect ratio", "#CFD8C4"),
        (axes[2], df["circularity"].to_numpy(float), "Circularity", "#D8CBD8"),
    ]
    for ax, values, label, colour in specs:
        bp = ax.boxplot(values, widths=0.45, patch_artist=True,
                        showmeans=True, meanline=False,
                        medianprops=dict(color=cfg.COLORS["accent"], lw=1.6),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor=cfg.COLORS["fit"],
                                       markersize=5),
                        flierprops=dict(marker="o", markersize=3.5,
                                        markerfacecolor="none",
                                        markeredgecolor="#888888"))
        bp["boxes"][0].set_facecolor(colour)
        bp["boxes"][0].set_edgecolor(cfg.COLORS["processed"])
        jitter = np.random.default_rng(7).normal(1.0, 0.035, size=len(values))
        ax.plot(jitter, values, "o", ms=2.6, color="#5A6B80", alpha=0.45)
        ax.set_ylabel(label)
        ax.set_xticks([])
    axes[0].text(0.03, 0.97,
                 f"median = {s['median_nm']:.1f}\n"
                 f"IQR = {s['iqr_nm']:.1f}\n"
                 f"n = {s['n']}",
                 transform=axes[0].transAxes, va="top", fontsize=8)
    fig.suptitle("Figure 12. Distribution of size and shape descriptors "
                 "(box = IQR, diamond = mean)", fontsize=10.5, y=1.01)
    stamp(axes[2], loc="lower right")
    save_figure(fig, "fig12_sem_boxplot")


def _plot_sem_distribution(d, binned, s):
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.bar(binned["bin_centre_nm"], binned["number_fraction_percent"],
           width=np.diff(binned[["bin_lower_nm", "bin_upper_nm"]].to_numpy(),
                         axis=1).ravel() * 0.92,
           color="#B8C4D4", edgecolor=cfg.COLORS["processed"], lw=0.7,
           label="Number-weighted")
    ax.bar(binned["bin_centre_nm"], binned["volume_fraction_percent"],
           width=np.diff(binned[["bin_lower_nm", "bin_upper_nm"]].to_numpy(),
                         axis=1).ravel() * 0.48,
           color=cfg.COLORS["accent"], alpha=0.8, lw=0,
           label="Volume-weighted ($\\propto d^3$)")
    ax.set_xlabel("Equivalent circular diameter (nm)")
    ax.set_ylabel("Fraction (%)")

    ax2 = ax.twinx()
    ax2.plot(binned["bin_upper_nm"], binned["cumulative_number_percent"],
             "-o", ms=3.5, lw=1.2, color=cfg.COLORS["neutral"],
             label="Cumulative (number)")
    ax2.set_ylabel("Cumulative number fraction (%)")
    ax2.set_ylim(0, 105)
    ax2.minorticks_off()

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("Figure 13. Number- vs volume-weighted size distribution")
    ax.text(0.97, 0.55,
            f"$D_{{[4,3]}}$ = {s['volume_weighted_D43_nm']:.1f} nm\n"
            f"$D_{{[3,2]}}$ = {s['surface_weighted_D32_nm']:.1f} nm\n"
            f"$\\bar{{d}}_{{number}}$ = {s['mean_nm']:.1f} nm",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
    stamp(ax, loc="lower right")
    save_figure(fig, "fig13_sem_size_distribution")


# ==========================================================================
# SECTION 8 -- TGA
# ==========================================================================
TGA_REGIONS = [
    dict(name="Region I", lo=25.0, hi=150.0,
         process="Loss of physisorbed and interparticle water",
         confidence="high"),
    dict(name="Region II", lo=150.0, hi=400.0,
         process="Loss of strongly bound water and surface dehydroxylation "
                 "(2 Fe-OH -> Fe-O-Fe + H2O)",
         confidence="medium"),
    dict(name="Region III", lo=400.0, hi=650.0,
         process="Residual dehydroxylation / loss of lattice-associated OH",
         confidence="low"),
    dict(name="Region IV", lo=650.0, hi=800.0,
         process="Comparatively stable; no further resolvable mass loss",
         confidence="high"),
]


def _analyse_tga_impl() -> dict:
    print("\n[TGA] ------------------------------------------------------")
    df = load_raw("tga_data.csv")
    T = df["temperature_C"].to_numpy(float)
    m = df["mass_percent"].to_numpy(float)
    print(f"  loaded {len(T)} points, {T.min():.0f}-{T.max():.0f} C")

    # --- Smoothing before differentiation (essential, and justified) -------
    # Differentiation amplifies noise as 1/dT. A Savitzky-Golay filter
    # computes the smoothed derivative analytically from the local polynomial,
    # which is far better conditioned than finite differences on raw data.
    step_C = float(np.median(np.diff(T)))
    m_smooth = savgol_filter(m, window_length=51, polyorder=3)

    # DTG = -dm/dT, expressed in % per degree C. The sign convention makes
    # mass-loss events appear as POSITIVE peaks.
    #
    # A wider window is used for the derivative than for the mass curve
    # itself: differentiation amplifies high-frequency noise, and an
    # under-smoothed DTG trace generates spurious maxima between the genuine
    # events.
    dtg = -savgol_filter(m, window_length=151, polyorder=3,
                         deriv=1, delta=step_C)

    # --- DTG peak detection ------------------------------------------------
    # Threshold on BOTH an absolute noise multiple and a fraction of the
    # largest DTG feature. The second criterion is what suppresses the
    # low-amplitude ripples that noise in the derivative always produces.
    noise_dtg = float(np.std(dtg[T > 700]))
    dtg_max = float(np.max(dtg))
    pk, props = find_peaks(dtg,
                           height=max(5.0 * noise_dtg, 0.04 * dtg_max),
                           prominence=max(4.0 * noise_dtg, 0.05 * dtg_max),
                           distance=int(40.0 / step_C))
    dtg_peaks = [dict(temperature_C=round(float(T[i]), 1),
                      dtg_percent_per_C=round(float(dtg[i]), 5))
                 for i in pk]
    print(f"  DTG maxima at: "
          f"{[p['temperature_C'] for p in dtg_peaks]} C "
          f"(noise = {noise_dtg:.2e} %/C)")

    # --- Mass loss by region ----------------------------------------------
    def mass_at(t):
        return float(np.interp(t, T, m_smooth))

    m_start, m_end = mass_at(T.min()), mass_at(T.max())
    total_loss = m_start - m_end

    regions = []
    for r in TGA_REGIONS:
        lo, hi = max(r["lo"], T.min()), min(r["hi"], T.max())
        loss = mass_at(lo) - mass_at(hi)
        sel = (T >= lo) & (T <= hi)
        peak_T = float(T[sel][np.argmax(dtg[sel])]) if sel.any() else np.nan
        regions.append(dict(
            region=r["name"],
            T_start_C=lo, T_end_C=hi,
            mass_at_start_percent=round(mass_at(lo), 3),
            mass_at_end_percent=round(mass_at(hi), 3),
            mass_loss_percent=round(loss, 3),
            fraction_of_total_loss_percent=round(100 * loss / total_loss, 1),
            dtg_max_temperature_C=round(peak_T, 1),
            assigned_process=r["process"],
            assignment_confidence=r["confidence"],
        ))
        print(f"  {r['name']:>10s} {lo:5.0f}-{hi:3.0f} C : "
              f"{loss:5.2f} %  ({100*loss/total_loss:4.1f} % of total)")

    region_df = pd.DataFrame(regions)

    # --- Model-based step deconvolution -----------------------------------
    # Region-based integration has an unavoidable weakness: the answer
    # depends on where the analyst draws the boundaries, and adjacent events
    # overlap across those boundaries. Fitting the whole curve as a sum of
    # sigmoidal steps removes that arbitrariness and returns the amplitude of
    # each EVENT rather than of each temperature interval.
    deconv = _deconvolve_tga_steps(T, m_smooth, dtg_peaks)
    if deconv["converged"]:
        print("  step deconvolution (boundary-independent):")
        for s in deconv["steps"]:
            print(f"    step at {s['midpoint_C']:6.1f} C : "
                  f"{s['amplitude_percent']:5.2f} % "
                  f"(width {s['width_C']:.1f} C)")

    # --- Cross-technique inference: surface hydroxyl/water coverage --------
    # Uses the XRD-derived crystallite size to convert a Region II mass loss
    # into an areal water density -- a genuinely useful combined quantity.
    D_nm = RESULTS.get("xrd", {}).get("williamson_hall_size_nm")
    coverage = None
    if D_nm:
        ssa_m2_g = 6.0 / (RHO_MAGNETITE_G_CM3 * D_nm * 1e-7) / 1e4  # m2/g
        region_ii = regions[1]["mass_loss_percent"]
        # per 1 g of sample:
        area_nm2 = ssa_m2_g * 1e18                      # nm^2 per gram
        molecules = (region_ii / 100.0) / M_WATER_G_MOL * N_AVOGADRO
        coverage = molecules / area_nm2
        print(f"  implied Region II water coverage = {coverage:.2f} "
              f"molecules/nm^2 (SSA = {ssa_m2_g:.0f} m2/g from XRD size)")

    processed = pd.DataFrame({
        "temperature_C": np.round(T, 2),
        "mass_percent_raw": np.round(m, 4),
        "mass_percent_smoothed": np.round(m_smooth, 4),
        "dtg_percent_per_C": np.round(dtg, 6),
    })
    save_processed(processed, "tga_processed.csv",
                   "Savitzky-Golay smoothed mass and DTG derivative")
    save_processed(region_df, "tga_mass_loss_regions.csv",
                   "region-resolved mass losses and assignments")

    _plot_tga(T, m, m_smooth, regions, total_loss)
    _plot_dtg(T, dtg, dtg_peaks, regions)
    _plot_tga_dtg_combined(T, m_smooth, dtg, regions, total_loss)

    return {
        "n_points": int(len(T)),
        "temperature_range_C": [float(T.min()), float(T.max())],
        "atmosphere_assumed": cfg.GROUND_TRUTH["tga_atmosphere"],
        "heating_rate_C_per_min": cfg.GROUND_TRUTH["tga_heating_rate_C_per_min"],
        "initial_mass_percent": round(m_start, 3),
        "final_mass_percent": round(m_end, 3),
        "total_mass_loss_percent": round(total_loss, 3),
        "residue_percent": round(m_end, 3),
        "dtg_peaks": dtg_peaks,
        "regions": regions,
        "step_deconvolution": deconv,
        "implied_water_coverage_molecules_per_nm2": (
            round(float(coverage), 2) if coverage else None),
        "thermal_stability_note": (
            "Above ~650 C the synthetic curve is flat to within balance noise, "
            "i.e. no further resolvable volatile loss in the assumed inert "
            "atmosphere."),
    }


def _deconvolve_tga_steps(T: np.ndarray, m: np.ndarray,
                          dtg_peaks: list[dict]) -> dict:
    """Fit the TG curve as a sum of logistic steps plus a linear drift.

        m(T) = m0 + drift * (T - T_min) - SUM_i  A_i / (1 + exp(-(T - c_i)/w_i))

    A_i is the amplitude of event i in % of initial mass, c_i its midpoint
    temperature and w_i its characteristic width. The linear drift term
    absorbs buoyancy, which otherwise biases every amplitude.

    The number of steps and their starting midpoints come from the DTG
    maxima, so the model is driven by the data rather than assumed.
    """
    centres0 = [p["temperature_C"] for p in dtg_peaks]
    if not centres0:
        return {"converged": False, "steps": [], "reason": "no DTG maxima"}
    n = len(centres0)

    def model(x, *params):
        m0, drift = params[0], params[1]
        y = m0 + drift * (x - x.min())
        for i in range(n):
            a, c, w = params[2 + 3 * i: 5 + 3 * i]
            y = y - a / (1.0 + np.exp(-(x - c) / w))
        return y

    p0 = [float(m[0]), 0.0]
    lo = [float(m[0]) - 3.0, -0.02]
    hi = [float(m[0]) + 3.0, 0.02]
    for c in centres0:
        p0 += [0.8, c, 30.0]
        lo += [0.0, c - 45.0, 5.0]
        hi += [12.0, c + 45.0, 160.0]

    try:
        popt, pcov = curve_fit(model, T, m, p0=p0, bounds=(lo, hi),
                               maxfev=80000)
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:                            # pragma: no cover
        return {"converged": False, "steps": [], "reason": str(exc)}

    fit = model(T, *popt)
    steps = []
    for i in range(n):
        a, c, w = popt[2 + 3 * i: 5 + 3 * i]
        a_err = perr[2 + 3 * i]
        steps.append(dict(
            amplitude_percent=round(float(a), 3),
            amplitude_err_percent=round(float(a_err), 4),
            midpoint_C=round(float(c), 1),
            width_C=round(float(w), 1),
        ))
    steps.sort(key=lambda s: s["midpoint_C"])

    return {
        "converged": True,
        "n_steps": n,
        "steps": steps,
        "summed_amplitude_percent": round(
            float(sum(s["amplitude_percent"] for s in steps)), 3),
        "linear_drift_percent_per_C": float(f"{popt[1]:.3e}"),
        "r_squared": round(r_squared(m, fit), 6),
        "note": ("Boundary-independent alternative to region integration. "
                 "The summed amplitude exceeds the measured total mass loss "
                 "because part of the lowest-temperature event occurs below "
                 "the starting temperature of the run."),
    }


def _plot_tga(T, m, m_smooth, regions, total_loss):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(T, m, color=cfg.COLORS["raw"], lw=0.8, label="Raw (synthetic)")
    ax.plot(T, m_smooth, color=cfg.COLORS["processed"], lw=1.5,
            label="Smoothed (SG 51 pt)")

    shades = ["#EDF2F7", "#E6EFE6", "#F5EFE6", "#F2EAF2"]
    for r, c in zip(regions, shades):
        ax.axvspan(r["T_start_C"], r["T_end_C"], color=c, zorder=0)
        mid = 0.5 * (r["T_start_C"] + r["T_end_C"])
        ax.annotate(f"{r['region']}\n$-${r['mass_loss_percent']:.2f} %",
                    xy=(mid, 99.9), ha="center", va="top", fontsize=8,
                    color="#444444")

    ax.set_xlabel("Temperature ($\\degree$C)")
    ax.set_ylabel("Mass (%)")
    ax.set_title("Figure 14. Synthetic TGA curve with annotated mass-loss "
                 "regions")
    ax.set_xlim(T.min(), T.max())
    ax.set_ylim(94.4, 100.4)
    ax.legend(loc="lower left")
    ax.text(0.985, 0.60,
            f"Total mass loss = {total_loss:.2f} %\n"
            f"Residue at 800 $\\degree$C = {m_smooth[-1]:.2f} %\n"
            "Atmosphere: N$_2$ (assumed)",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC",
                      lw=0.7))
    stamp(ax, loc="lower right")
    save_figure(fig, "fig14_tga_curve")


def _plot_dtg(T, dtg, dtg_peaks, regions):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(T, dtg, color=cfg.COLORS["accent"], lw=1.3)
    ax.axhline(0, color="#CCCCCC", lw=0.7)
    for p in dtg_peaks:
        ax.annotate(f"{p['temperature_C']:.0f} $\\degree$C",
                    xy=(p["temperature_C"], p["dtg_percent_per_C"]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8.5, color=cfg.COLORS["neutral"],
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"))
    ax.set_xlabel("Temperature ($\\degree$C)")
    ax.set_ylabel(r"DTG, $-\mathrm{d}m/\mathrm{d}T$ (% $\degree$C$^{-1}$)")
    ax.set_title("Figure 15. Derivative thermogravimetric (DTG) curve")
    ax.set_xlim(T.min(), T.max())
    stamp(ax, loc="upper right")
    save_figure(fig, "fig15_dtg_curve")


def _plot_tga_dtg_combined(T, m_smooth, dtg, regions, total_loss):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(T, m_smooth, color=cfg.COLORS["processed"], lw=1.6, label="TG")
    ax.set_xlabel("Temperature ($\\degree$C)")
    ax.set_ylabel("Mass (%)", color=cfg.COLORS["processed"])
    ax.tick_params(axis="y", colors=cfg.COLORS["processed"])
    ax.set_ylim(94.4, 100.4)

    ax2 = ax.twinx()
    ax2.plot(T, dtg, color=cfg.COLORS["accent"], lw=1.3, label="DTG")
    ax2.set_ylabel(r"DTG (% $\degree$C$^{-1}$)", color=cfg.COLORS["accent"])
    ax2.tick_params(axis="y", colors=cfg.COLORS["accent"])
    ax2.minorticks_on()

    for r in regions[:3]:
        ax.axvline(r["T_end_C"], color="#CCCCCC", ls=":", lw=0.9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("Figure 16. Combined TG and DTG traces")
    ax.set_xlim(T.min(), T.max())
    stamp(ax, loc="lower left")
    save_figure(fig, "fig16_tga_dtg_combined")


# ==========================================================================
# SECTION 9 -- Cross-technique synthesis
# ==========================================================================
TECHNIQUE_MATRIX = [
    dict(technique="XRD",
         probes="Long-range crystallographic order (bulk, ensemble)",
         main_information="Crystal structure, phase identity, lattice "
                          "parameter, crystallite size, microstrain",
         quantitative_output="d-spacings; a = lattice parameter (A); "
                             "D_Scherrer (nm); microstrain eps",
         key_limitation="Volume-weighted and blind to amorphous or "
                        "poorly-ordered material; cannot separate Fe3O4 from "
                        "gamma-Fe2O3 on peak position alone; Scherrer size is "
                        "a coherent-domain size, NOT a particle size"),
    dict(technique="FTIR",
         probes="Vibrational modes -- local bonding and surface chemistry",
         main_information="Fe-O lattice modes; adsorbed water; surface "
                          "hydroxyls; presence/absence of organics",
         quantitative_output="Band positions (cm-1); relative intensities; "
                             "band areas (semi-quantitative at best)",
         key_limitation="Heavily overlapped bands in the Fe-O region; "
                        "absorptivities are unknown so intensities are not "
                        "concentrations; sampling geometry (ATR vs KBr) "
                        "shifts band positions and intensities"),
    dict(technique="UV-Vis",
         probes="Electronic transitions (and, unavoidably, light scattering)",
         main_information="Charge-transfer absorption; qualitative optical "
                          "behaviour; dispersion stability",
         quantitative_output="Absorbance vs wavelength; apparent Tauc "
                             "intercepts (unreliable for this material)",
         key_limitation="For a dispersion, measured absorbance mixes true "
                        "absorption with Mie/Rayleigh turbidity; Fe3O4 is not "
                        "a wide-gap semiconductor, so a Tauc 'band gap' is "
                        "not a meaningful intrinsic property"),
    dict(technique="SEM",
         probes="Surface morphology of individual features (local)",
         main_information="Particle/aggregate size, shape, degree of "
                          "agglomeration, surface texture",
         quantitative_output="Size distribution statistics: mean, median, SD, "
                             "quartiles, D[4,3], D[3,2]",
         key_limitation="Measures whole features, so aggregates are counted "
                        "as particles; thresholding choices shift the mean by "
                        "several percent; limited field of view creates "
                        "sampling bias; conductive coating inflates diameters"),
    dict(technique="TGA",
         probes="Mass change on heating (bulk, destructive)",
         main_information="Volatile content: adsorbed water, surface "
                          "hydroxyls, organics (if present); thermal stability",
         quantitative_output="Mass loss per region (%); DTG peak temperatures "
                             "(C); total volatile content (%)",
         key_limitation="Gives mass change only, never chemical identity; "
                        "results depend strongly on atmosphere, heating rate, "
                        "sample mass and packing; in air, Fe3O4 oxidation "
                        "causes a mass GAIN that can mask losses"),
]


def _cross_technique_synthesis_impl() -> dict:
    print("\n[Cross-technique] ------------------------------------------")
    matrix = pd.DataFrame(TECHNIQUE_MATRIX)
    save_processed(matrix, "technique_comparison_matrix.csv",
                   "what each technique does and does not tell us")

    xrd = RESULTS["xrd"]
    sem = RESULTS["sem"]
    tga = RESULTS["tga"]

    d_xrd = xrd["williamson_hall_size_nm"]
    d_sem_num = sem["mean_nm"]
    d_sem_vol = sem["volume_weighted_D43_nm"]

    ratio_num = d_sem_num / d_xrd
    ratio_vol = d_sem_vol / d_xrd

    comparison = {
        "xrd_scherrer_mean_nm": xrd["scherrer_mean_nm"],
        "xrd_williamson_hall_nm": d_xrd,
        "sem_number_mean_nm": d_sem_num,
        "sem_median_nm": sem["median_nm"],
        "sem_volume_weighted_D43_nm": d_sem_vol,
        "ratio_sem_number_to_xrd": round(ratio_num, 2),
        "ratio_sem_volume_to_xrd": round(ratio_vol, 2),
        "interpretation": (
            f"The SEM number-mean diameter ({d_sem_num:.1f} nm) exceeds the "
            f"XRD coherent-domain size ({d_xrd:.1f} nm) by a factor of "
            f"{ratio_num:.2f}; on a like-for-like volume-weighted basis the "
            f"factor is {ratio_vol:.2f}. In the demonstration scenario this "
            "gap is what one would EXPECT, and it is informative rather than "
            "contradictory."),
        "reasons_for_divergence": [
            "Different measurands: XRD reports the size of coherently "
            "diffracting domains; SEM reports the outline of a visible "
            "feature. A single SEM feature may contain several domains.",
            "Different weightings: Scherrer sizes are volume-weighted while "
            "a raw SEM mean is number-weighted. Volume weighting emphasises "
            "the largest particles, so the two statistics are not comparable "
            "until one is converted (here D[4,3]).",
            "Surface disorder: a structurally disordered or hydrated "
            "surface shell contributes to the SEM outline but diffracts "
            "incoherently, so XRD does not count it.",
            "Agglomeration: touching particles are frequently segmented as "
            "one feature, inflating the SEM size.",
            "Microstrain: if strain broadening is attributed entirely to "
            "size, the Scherrer size is systematically UNDERESTIMATED. Here "
            f"Scherrer alone gives {xrd['scherrer_mean_nm']:.1f} nm while the "
            f"Williamson-Hall separation gives {d_xrd:.1f} nm.",
            "Imaging artefacts: conductive coating thickness and edge "
            "detection thresholds both bias SEM diameters upward.",
        ],
        "complementarity": (
            "None of these techniques is a substitute for another. XRD "
            "establishes what the crystalline phase IS; SEM establishes how "
            "the material is physically organised; FTIR reports what sits on "
            "the surface; TGA quantifies how much of it there is; UV-Vis "
            "reports the optical consequence. A defensible characterisation "
            "requires the set, and phase assignment in particular should be "
            "confirmed by a magnetic or spectroscopic method (Mossbauer, "
            "Raman, XPS) rather than by XRD peak positions alone."),
    }

    if tga.get("implied_water_coverage_molecules_per_nm2"):
        comparison["combined_xrd_tga_surface_coverage"] = (
            f"Combining the XRD crystallite size (specific surface area "
            f"{xrd['implied_ssa_m2_per_g']:.0f} m2/g for equivalent spheres) "
            f"with the Region II TGA mass loss gives "
            f"{tga['implied_water_coverage_molecules_per_nm2']:.1f} water "
            "molecules per nm2. That figure is in the range usually quoted "
            "for hydroxylated iron-oxide surfaces, which is a useful internal "
            "consistency check on the demonstration dataset.")

    print(f"  SEM/XRD size ratio: {ratio_num:.2f} (number-weighted), "
          f"{ratio_vol:.2f} (volume-weighted)")

    summary_rows = [
        ("XRD", "Phase + crystallite size",
         f"a = {xrd['lattice_parameter_ang']:.4f} A; "
         f"D = {d_xrd:.1f} nm; eps = {xrd['williamson_hall_microstrain']:.1e}"),
        ("FTIR", "Surface chemistry",
         f"{RESULTS['ftir']['n_bands_detected']} bands; Fe-O lattice modes "
         "+ O-H / H-O-H; no organic bands"),
        ("UV-Vis", "Optical behaviour",
         f"Broad CT absorption; scattering ~ lambda^-"
         f"{RESULTS['uvvis']['scattering_exponent_n_effective']:.1f}; "
         "no defensible band gap"),
        ("SEM", "Morphology + size distribution",
         f"n = {sem['n']}; mean = {d_sem_num:.1f} nm; "
         f"median = {sem['median_nm']:.1f} nm; GSD = {sem['lognormal_gsd']:.2f}"),
        ("TGA", "Volatile content",
         f"Total loss = {tga['total_mass_loss_percent']:.2f} %; "
         f"DTG maxima at "
         f"{', '.join(str(p['temperature_C']) for p in tga['dtg_peaks'])} C"),
    ]
    save_processed(
        pd.DataFrame(summary_rows,
                     columns=["technique", "role", "demonstration_outcome"]),
        "cross_technique_summary.csv", "headline result from each technique")

    _plot_cross_technique(xrd, sem, tga, comparison)
    _plot_technique_dashboard()

    return {"technique_matrix": TECHNIQUE_MATRIX,
            "size_comparison": comparison}


def _plot_cross_technique(xrd, sem, tga, comparison):
    """Size-scale comparison: the central cross-technique message."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    d = pd.read_csv(os.path.join(cfg.RAW_DIR, "sem_particle_measurements.csv"),
                    comment="#")["equivalent_diameter_nm"].to_numpy(float)

    parts = ax.violinplot([d], positions=[1], widths=0.7, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#B8C4D4")
        body.set_edgecolor(cfg.COLORS["processed"])
        body.set_alpha(0.75)
    ax.plot(np.random.default_rng(3).normal(1.0, 0.045, len(d)), d, "o",
            ms=3, color="#5A6B80", alpha=0.5)

    markers = [
        (xrd["scherrer_mean_nm"], "XRD Scherrer\n(mean of strong peaks)",
         cfg.COLORS["accent"], "s"),
        (xrd["williamson_hall_size_nm"], "XRD Williamson-Hall\n(size + strain)",
         cfg.COLORS["fit"], "D"),
        (sem["mean_nm"], "SEM number mean", cfg.COLORS["processed"], "o"),
        (sem["volume_weighted_D43_nm"], "SEM volume-weighted $D_{[4,3]}$",
         cfg.COLORS["highlight"], "^"),
    ]
    for i, (value, label, colour, mk) in enumerate(markers):
        ax.plot([1.55 + 0.001], [value], mk, color=colour, ms=9,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(f"{label}\n{value:.1f} nm", xy=(1.62, value),
                    xytext=(1.72, value), fontsize=8.5, va="center",
                    color=colour)
        ax.plot([1.0, 1.55], [value, value], ls=":", lw=0.8, color=colour,
                alpha=0.55)

    ax.set_xlim(0.5, 2.7)
    ax.set_xticks([1, 1.55])
    ax.set_xticklabels(["SEM feature\npopulation", "Derived\nsize metrics"])
    ax.set_ylabel("Diameter / coherent domain size (nm)")
    ax.set_title("Figure 17. Why XRD and SEM sizes differ -- "
                 "they measure different things")
    stamp(ax, loc="lower left")
    save_figure(fig, "fig17_size_comparison")


def _plot_technique_dashboard():
    """Single-page visual summary of all five techniques."""
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.07, right=0.975, top=0.87, bottom=0.08)

    # XRD
    ax = fig.add_subplot(gs[0, 0])
    p = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "xrd_processed.csv"),
                    comment="#")
    ax.plot(p["two_theta_deg"], p["intensity_bg_subtracted_counts"],
            color=cfg.COLORS["processed"], lw=0.8)
    ax.set_xlabel(r"2$\theta$ ($\degree$)"); ax.set_ylabel("Counts")
    ax.set_title("XRD - phase & crystallite size", fontsize=9.5)

    # FTIR
    ax = fig.add_subplot(gs[0, 1])
    p = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "ftir_processed.csv"),
                    comment="#")
    ax.plot(p["wavenumber_cm-1"], p["absorbance_smoothed"],
            color=cfg.COLORS["accent"], lw=0.9)
    ax.set_xlim(4000, 400)
    ax.set_xlabel(r"cm$^{-1}$"); ax.set_ylabel("Absorbance")
    ax.set_title("FTIR - surface chemistry", fontsize=9.5)

    # UV-Vis
    ax = fig.add_subplot(gs[0, 2])
    p = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "uvvis_processed.csv"),
                    comment="#")
    ax.plot(p["wavelength_nm"], p["absorbance_smoothed"],
            color=cfg.COLORS["highlight"], lw=1.1)
    ax.set_xlabel("nm"); ax.set_ylabel("Absorbance")
    ax.set_title("UV-Vis - optical response", fontsize=9.5)

    # SEM
    ax = fig.add_subplot(gs[1, 0])
    d = pd.read_csv(os.path.join(cfg.RAW_DIR,
                                 "sem_particle_measurements.csv"),
                    comment="#")["equivalent_diameter_nm"]
    ax.hist(d, bins="fd", color="#B8C4D4",
            edgecolor=cfg.COLORS["processed"], lw=0.7)
    ax.set_xlabel("Diameter (nm)"); ax.set_ylabel("Count")
    ax.set_title("SEM - size distribution", fontsize=9.5)

    # TGA
    ax = fig.add_subplot(gs[1, 1])
    p = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, "tga_processed.csv"),
                    comment="#")
    ax.plot(p["temperature_C"], p["mass_percent_smoothed"],
            color=cfg.COLORS["processed"], lw=1.3)
    ax2 = ax.twinx()
    ax2.plot(p["temperature_C"], p["dtg_percent_per_C"],
             color=cfg.COLORS["accent"], lw=1.0)
    ax2.set_ylabel("DTG", color=cfg.COLORS["accent"], fontsize=8)
    ax2.tick_params(labelsize=7.5, colors=cfg.COLORS["accent"])
    ax.set_xlabel(r"T ($\degree$C)"); ax.set_ylabel("Mass (%)")
    ax.set_title("TGA / DTG - volatile content", fontsize=9.5)

    # Summary text panel
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    xrd, sem, tga = RESULTS["xrd"], RESULTS["sem"], RESULTS["tga"]
    lines = [
        ("Demonstration outcomes", ""),
        ("Phase (XRD)", xrd["phase_assignment"]),
        ("Lattice parameter", f"{xrd['lattice_parameter_ang']:.4f} $\\AA$"),
        ("Crystallite size (W-H)",
         f"{xrd['williamson_hall_size_nm']:.1f} nm"),
        ("Microstrain",
         f"{xrd['williamson_hall_microstrain']:.1e}"),
        ("SEM mean diameter", f"{sem['mean_nm']:.1f} nm (n = {sem['n']})"),
        ("SEM / XRD ratio",
         f"{sem['mean_nm']/xrd['williamson_hall_size_nm']:.2f}"),
        ("Total mass loss (TGA)",
         f"{tga['total_mass_loss_percent']:.2f} %"),
        ("Optical band gap", "not defensible - see report"),
    ]
    y = 0.98
    for k, v in lines:
        if v == "":
            ax.text(0, y, k, fontsize=10, weight="bold",
                    transform=ax.transAxes, va="top")
            y -= 0.105
        else:
            ax.text(0, y, k, fontsize=8.5, color="#555555",
                    transform=ax.transAxes, va="top")
            ax.text(1.0, y, v, fontsize=8.5, ha="right",
                    color=cfg.COLORS["neutral"], transform=ax.transAxes,
                    va="top")
            y -= 0.095

    fig.suptitle("Figure 18. Cross-technique characterization summary - "
                 "Fe$_3$O$_4$ nanoparticles",
                 fontsize=13, y=0.965, weight="bold")
    fig.text(0.5, 0.925,
             "SYNTHETIC DEMONSTRATION DATASETS - not experimental measurements",
             ha="center", fontsize=9, color=cfg.COLORS["accent"], style="italic")
    save_figure(fig, "fig18_summary_dashboard")




# --------------------------------------------------------------------------
# Public entry points: each registers its result in RESULTS so that the
# functions can be called individually (e.g. from the notebook) without
# relying on main() to wire them together.
# --------------------------------------------------------------------------


def analyse_xrd() -> dict:
    out = _analyse_xrd_impl()
    RESULTS["xrd"] = out
    return out


def analyse_ftir() -> dict:
    out = _analyse_ftir_impl()
    RESULTS["ftir"] = out
    return out


def analyse_uvvis() -> dict:
    out = _analyse_uvvis_impl()
    RESULTS["uvvis"] = out
    return out


def analyse_sem() -> dict:
    out = _analyse_sem_impl()
    RESULTS["sem"] = out
    return out


def analyse_tga() -> dict:
    out = _analyse_tga_impl()
    RESULTS["tga"] = out
    return out


def cross_technique_synthesis() -> dict:
    out = _cross_technique_synthesis_impl()
    RESULTS["cross_technique"] = out
    return out


# ==========================================================================
# Entry point
# ==========================================================================
def main() -> None:
    print("=" * 70)
    print("IRON OXIDE NANOPARTICLE CHARACTERIZATION -- ANALYSIS PIPELINE")
    print(cfg.DISCLAIMER_LONG)
    print("=" * 70)

    analyse_xrd()
    analyse_ftir()
    analyse_uvvis()
    analyse_sem()
    analyse_tga()
    cross_technique_synthesis()

    out = os.path.join(cfg.REPORT_DIR, "analysis_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print(f"\nResults written to report/analysis_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
