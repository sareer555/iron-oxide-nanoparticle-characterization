"""
generate_synthetic_datasets.py
==============================
Builds every RAW demonstration dataset used in this portfolio project.

>>> ALL DATA PRODUCED BY THIS SCRIPT ARE SYNTHETIC. <<<
No sample was synthesized. No instrument was operated. Each generator uses an
explicit, documented physical model plus realistic noise, so that the datasets
are *plausible* and *internally consistent* -- never so that they can be
mistaken for measurements. Every output CSV carries a disclaimer header.

Generative models
-----------------
XRD   : pseudo-Voigt Bragg reflections of cubic Fe3O4 on a decaying
        background, with size + microstrain broadening convolved (in
        quadrature) with a Gaussian instrumental function; Poisson counting
        noise.
FTIR  : Gaussian/Lorentzian absorbance bands -> transmittance, on a drifting
        baseline, with detector noise and a deliberate atmospheric CO2 artefact.
UV-Vis: ligand-to-metal charge-transfer edge + Urbach tail + a Rayleigh/Mie
        turbidity term (lambda^-n), i.e. a scattering-contaminated spectrum.
SEM   : lognormal equivalent-diameter population plus correlated aspect-ratio
        and circularity descriptors; no image is fabricated.
TGA   : sum of logistic (sigmoidal) mass-loss steps in an assumed INERT
        atmosphere, plus buoyancy drift and balance noise.

Run:  python scripts/generate_synthetic_datasets.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402


# ==========================================================================
# Shared helpers
# ==========================================================================
def pseudo_voigt(x: np.ndarray, center: float, fwhm: float,
                 eta: float) -> np.ndarray:
    """Unit-height pseudo-Voigt profile.

    pV(x) = eta * L(x) + (1 - eta) * G(x)

    Both components are normalised to unit height and share the same FWHM,
    so the returned profile also has unit height and (approximately) the
    requested FWHM. `eta` is the Lorentzian fraction.
    """
    dx = (x - center) / fwhm
    lorentz = 1.0 / (1.0 + 4.0 * dx ** 2)
    gauss = np.exp(-4.0 * np.log(2.0) * dx ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def gaussian_band(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Unit-height Gaussian band."""
    return np.exp(-4.0 * np.log(2.0) * ((x - center) / fwhm) ** 2)


def lorentzian_band(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Unit-height Lorentzian band."""
    return 1.0 / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)


def logistic_step(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Logistic (sigmoidal) step rising from 0 to 1 around `center`.

    `width` is the characteristic transition width; the 10-90 % rise spans
    roughly 4.4 * width.
    """
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def _disclaimer_header(technique: str, model_notes: list[str]) -> str:
    """Comment block prepended to every raw CSV."""
    lines = [
        "# ============================================================",
        f"# {technique} -- SYNTHETIC DEMONSTRATION DATASET",
        "# ============================================================",
        f"# {cfg.DISCLAIMER_LONG}",
        "#",
        "# This file was produced by scripts/generate_synthetic_datasets.py",
        "# for a data-analysis PORTFOLIO project. It is NOT an instrument",
        "# export and must not be cited as experimental evidence.",
        "#",
        f"# Hypothetical sample : {cfg.SAMPLE_CONTEXT['material']}",
        f"# Random seed         : {cfg.RANDOM_SEED}",
        "#",
        "# Generative model:",
    ]
    lines += [f"#   - {n}" for n in model_notes]
    lines.append("# ============================================================")
    return "\n".join(lines) + "\n"


def write_csv_with_header(df: pd.DataFrame, path: str, technique: str,
                          model_notes: list[str]) -> None:
    """Write a dataframe to CSV preceded by the synthetic-data disclaimer."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(_disclaimer_header(technique, model_notes))
        df.to_csv(fh, index=False, lineterminator="\n")
    print(f"  wrote {os.path.relpath(path, cfg.PROJECT_ROOT)}  "
          f"({len(df)} rows)")


# ==========================================================================
# 1. XRD
# ==========================================================================
def generate_xrd(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic powder XRD pattern for nanocrystalline Fe3O4.

    Broadening model (this is the ground truth the analysis must recover):

        beta_sample(theta) = K*lambda / (D cos theta)  +  4 * eps * tan theta
        beta_observed      = sqrt( beta_sample^2 + beta_instrumental^2 )

    The first term is Scherrer size broadening, the second is uniform
    microstrain broadening -- the standard Williamson-Hall decomposition.
    Instrumental broadening is added in quadrature as a Gaussian contribution.
    """
    gt = cfg.GROUND_TRUTH
    two_theta_min, two_theta_max = cfg.ACQUISITION["xrd"]["range_2theta_deg"]
    step = cfg.ACQUISITION["xrd"]["step_deg"]

    two_theta = np.arange(two_theta_min, two_theta_max + 0.5 * step, step)

    D_nm = gt["xrd_crystallite_size_nm"]
    eps = gt["xrd_microstrain"]
    beta_instr_deg = gt["xrd_instrumental_fwhm_deg"]
    eta = gt["xrd_eta_pseudovoigt"]
    lam_nm = cfg.WAVELENGTH_NM
    K = cfg.SCHERRER_K

    # ---- Bragg peaks -----------------------------------------------------
    intensity = np.zeros_like(two_theta)
    peak_log = []

    for h, k, l, i_rel in cfg.MAGNETITE_REFLECTIONS:
        pos = cfg.two_theta_from_hkl(h, k, l)
        theta = np.radians(pos / 2.0)

        beta_size_rad = K * lam_nm / (D_nm * np.cos(theta))
        beta_strain_rad = 4.0 * eps * np.tan(theta)
        beta_sample_deg = np.degrees(beta_size_rad + beta_strain_rad)
        fwhm_deg = float(np.sqrt(beta_sample_deg ** 2 + beta_instr_deg ** 2))

        # Lorentz-polarisation factor: makes low-angle peaks relatively
        # stronger, as in a real Bragg-Brentano pattern.
        lp = (1.0 + np.cos(2.0 * theta) ** 2) / (
            np.sin(theta) ** 2 * np.cos(theta))
        lp_ref = None
        peak_log.append(dict(h=h, k=k, l=l, two_theta_deg=pos,
                             i_rel=i_rel, fwhm_deg=fwhm_deg, lp=lp))

    lp_311 = [p["lp"] for p in peak_log
              if (p["h"], p["k"], p["l"]) == (3, 1, 1)][0]

    scale = 980.0  # peak counts of the strongest (311) reflection
    for p in peak_log:
        amp = scale * (p["i_rel"] / 100.0) * (p["lp"] / lp_311)
        intensity += amp * pseudo_voigt(two_theta, p["two_theta_deg"],
                                        p["fwhm_deg"], eta)
        p["amplitude_counts"] = amp

    # ---- Background ------------------------------------------------------
    # (a) monotonic decay: air scatter + sample fluorescence (Fe with Cu Ka
    #     radiation fluoresces strongly -> genuinely high background)
    bg_decay = 145.0 * np.exp(-(two_theta - two_theta_min) / 95.0) + 58.0
    # (b) a broad, weak amorphous/hydrated-surface hump near 2theta ~ 27 deg
    bg_hump = 26.0 * gaussian_band(two_theta, 27.5, 22.0)
    background = bg_decay + bg_hump

    total = intensity + background

    # ---- Counting statistics --------------------------------------------
    # X-ray detection is a Poisson process: sigma = sqrt(counts).
    observed = rng.poisson(total).astype(float)

    df = pd.DataFrame({
        "two_theta_deg": np.round(two_theta, 3),
        "intensity_counts": observed,
    })
    return df, peak_log


# ==========================================================================
# 2. FTIR
# ==========================================================================
def generate_ftir(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic FTIR spectrum of surfactant-free Fe3O4 nanoparticles.

    Band set is deliberately minimal and defensible for an uncoated,
    air-exposed iron oxide:
      ~3400 cm-1  broad  nu(O-H)     adsorbed water + surface hydroxyls
      ~1630 cm-1  medium delta(H-O-H) molecular water bending
      ~ 570 cm-1  strong  nu(Fe-O)    tetrahedral-site stretch
      ~ 440 cm-1  medium  nu(Fe-O)    octahedral-site stretch
    Plus a sharp ~2340 cm-1 ATMOSPHERIC CO2 artefact, included on purpose so
    the analysis can demonstrate recognising an artefact rather than
    assigning it to the sample.

    NOTE: no C-H or C=O bands are generated, because the project design
    assumes NO organic capping agent. Goethite marker bands (~890/795 cm-1)
    are also absent by design.
    """
    lo, hi = 400, 4000
    wn = np.arange(lo, hi + 1, 1.0, dtype=float)

    # (center, fwhm, peak absorbance, shape)
    bands = [
        (3396.0, 420.0, 0.372, "gauss"),   # nu(O-H), H-bonded -> very broad
        (1628.0,  62.0, 0.121, "gauss"),   # delta(H-O-H)
        (1045.0, 110.0, 0.028, "gauss"),   # weak Fe-OH bending / surface
        (2341.0,   9.0, 0.043, "lorentz"), # ATMOSPHERIC CO2 -- artefact
        (571.0,   96.0, 0.545, "lorentz"), # nu(Fe-O) tetrahedral
        (441.0,   84.0, 0.318, "lorentz"), # nu(Fe-O) octahedral
    ]

    absorbance = np.zeros_like(wn)
    for center, fwhm, amp, shape in bands:
        profile = (gaussian_band(wn, center, fwhm) if shape == "gauss"
                   else lorentzian_band(wn, center, fwhm))
        absorbance += amp * profile

    # Sloping, slightly curved baseline: typical of ATR sampling / scattering
    # from a particulate powder.
    xnorm = (wn - lo) / (hi - lo)
    baseline = 0.052 - 0.030 * xnorm + 0.019 * xnorm ** 2

    # Detector noise grows modestly at the low-wavenumber end.
    noise_scale = 0.0016 * (1.0 + 1.8 * np.exp(-(wn - lo) / 380.0))
    noise = rng.normal(0.0, noise_scale)

    absorbance_total = absorbance + baseline + noise
    transmittance = 100.0 * np.power(10.0, -absorbance_total)

    # Conventional FTIR display order: high -> low wavenumber.
    order = np.argsort(-wn)
    df = pd.DataFrame({
        "wavenumber_cm-1": wn[order],
        "transmittance_percent": np.round(transmittance[order], 4),
        "absorbance": np.round(absorbance_total[order], 6),
    })
    return df


# ==========================================================================
# 3. UV-Vis
# ==========================================================================
def generate_uvvis(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic UV-Vis spectrum of an aqueous Fe3O4 nanoparticle dispersion.

    Physical model -- three additive contributions:
      1. O(2p) -> Fe(3d) ligand-to-metal charge transfer in the UV, modelled
         as a broad saturating edge.
      2. An Urbach-like exponential tail extending into the visible, which is
         what actually makes magnetite look black.
      3. A Rayleigh/Mie TURBIDITY term ~ lambda^-n (n = 2.8). This is
         scattering, NOT absorption, and it is included deliberately: it is
         the single biggest reason a Tauc analysis of a nanoparticle
         DISPERSION is unreliable.

    Because Fe3O4 is a mixed-valence, near-metallic conductor rather than a
    wide-gap semiconductor, this dataset is NOT designed to contain a
    well-defined optical band gap. The analysis exploits that fact.
    """
    lo, hi = cfg.ACQUISITION["uvvis"]["range_nm"]
    wl = np.arange(lo, hi + 1, cfg.ACQUISITION["uvvis"]["step_nm"], dtype=float)
    ev = 1239.841984 / wl  # photon energy (eV); h*c in eV*nm

    # 1. Charge-transfer edge (saturating toward the deep UV)
    a_ct = 1.62 / (1.0 + np.exp(-(ev - 3.55) / 0.42))

    # 2. Urbach-type sub-edge tail -> broad visible absorption
    a_urbach = 0.78 * np.exp((ev - 3.55) / 1.05)
    a_urbach = np.clip(a_urbach, 0.0, 0.95)

    # 3. Turbidity / scattering baseline (lambda^-n), n = 2.8
    a_scatter = 0.145 * (wl / 400.0) ** (-2.8)

    # Weak, very broad ligand-field / intervalence feature around 640 nm.
    a_ligand_field = 0.034 * gaussian_band(wl, 640.0, 210.0)

    absorbance = a_ct + a_urbach + a_scatter + a_ligand_field

    # Small instrument offset + photometric noise (worse at high absorbance
    # and in the deep UV where the source is weak).
    offset = 0.0121
    noise = rng.normal(0.0, 0.0011 + 0.0016 * np.exp(-(wl - lo) / 90.0))
    absorbance = absorbance + offset + noise

    df = pd.DataFrame({
        "wavelength_nm": wl,
        "absorbance": np.round(absorbance, 5),
    })
    return df


# ==========================================================================
# 4. SEM-derived particle measurements
# ==========================================================================
def generate_sem(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic particle measurement table.

    NO micrograph is fabricated. This table represents the kind of numeric
    output an analyst would obtain AFTER segmenting a real micrograph in
    ImageJ/Fiji -- i.e. the input to the statistical workflow being
    demonstrated.

    Equivalent circular diameter follows a lognormal distribution, which is
    the standard descriptor for particle populations produced by nucleation
    and growth. Aspect ratio and circularity are generated with a weak,
    physically sensible correlation to size (larger features in a real
    micrograph are more often touching/overlapping neighbours, so they
    appear slightly more elongated and less circular).
    """
    gt = cfg.GROUND_TRUTH
    n = gt["sem_n_particles"]
    median = gt["sem_lognormal_median_nm"]
    gsd = gt["sem_lognormal_gsd"]

    mu = np.log(median)         # lognormal location parameter
    sigma = np.log(gsd)         # lognormal shape parameter

    diameter = rng.lognormal(mean=mu, sigma=sigma, size=n)

    # Weak size-linked elongation, floored at 1.0 (aspect ratio >= 1).
    z = (np.log(diameter) - mu) / sigma
    aspect_ratio = 1.08 + 0.055 * z + rng.normal(0.0, 0.055, size=n)
    aspect_ratio = np.clip(aspect_ratio, 1.0, None)

    # Circularity: 4*pi*A / P^2, bounded above by 1 for a perfect circle.
    circularity = 0.94 - 0.030 * z + rng.normal(0.0, 0.028, size=n)
    circularity = np.clip(circularity, 0.45, 0.99)

    df = pd.DataFrame({
        "particle_id": [f"P{idx:03d}" for idx in range(1, n + 1)],
        "equivalent_diameter_nm": np.round(diameter, 2),
        "aspect_ratio": np.round(aspect_ratio, 3),
        "circularity": np.round(circularity, 3),
    })
    return df


# ==========================================================================
# 5. TGA
# ==========================================================================
def generate_tga(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic TGA curve for surfactant-free Fe3O4 nanoparticles in N2.

    Three logistic mass-loss steps, consistent with an UNCOATED oxide:
      Step 1 (~30-150 C) : physisorbed / interparticle water
      Step 2 (~150-400 C): strongly bound water + surface dehydroxylation
      Step 3 (~400-650 C): residual dehydroxylation / lattice-bound OH

    There is deliberately NO organic-decomposition step, because the project
    design assumes no capping agent.

    The atmosphere is assumed INERT (N2). This matters enormously: in AIR,
    Fe3O4 oxidises to gamma-Fe2O3 and then alpha-Fe2O3 above ~200 C, which
    produces a mass GAIN of up to +3.45 % and would change the curve's shape
    completely. That contrast is discussed in the report.
    """
    gt = cfg.GROUND_TRUTH
    lo, hi = cfg.ACQUISITION["tga"]["range_C"]
    step = cfg.ACQUISITION["tga"]["step_C"]
    temperature = np.arange(lo, hi + 0.5 * step, step)

    steps = [
        (gt["tga_step1_pct"],  78.0, 21.0),   # (amplitude %, center C, width C)
        (gt["tga_step2_pct"], 243.0, 44.0),
        (gt["tga_step3_pct"], 522.0, 38.0),
    ]

    mass = np.full_like(temperature, 100.0)
    for amplitude, center, width in steps:
        mass -= amplitude * logistic_step(temperature, center, width)

    # Buoyancy drift: gas density falls as the furnace heats, so the apparent
    # mass creeps upward slightly. A real, well-known TGA artefact.
    mass += 0.045 * (temperature - lo) / (hi - lo)

    # Microbalance noise.
    mass += rng.normal(0.0, 0.0045, size=temperature.size)

    initial_mg = gt["tga_initial_mass_mg"]
    df = pd.DataFrame({
        "temperature_C": np.round(temperature, 2),
        "mass_percent": np.round(mass, 4),
        "mass_mg": np.round(mass / 100.0 * initial_mg, 5),
    })
    return df


# ==========================================================================
# Entry point
# ==========================================================================
def main() -> None:
    print("=" * 70)
    print("GENERATING SYNTHETIC DEMONSTRATION DATASETS")
    print(cfg.DISCLAIMER_LONG)
    print("=" * 70)

    # Independent seeded streams -> each dataset reproducible on its own.
    seeds = np.random.SeedSequence(cfg.RANDOM_SEED).spawn(5)
    rngs = [np.random.default_rng(s) for s in seeds]

    print("\n[1/5] XRD")
    xrd, peak_log = generate_xrd(rngs[0])
    write_csv_with_header(
        xrd, os.path.join(cfg.RAW_DIR, "xrd_data.csv"),
        "XRD (Cu K-alpha1, 1.54060 A)",
        [
            "pseudo-Voigt Bragg peaks of cubic Fe3O4 (a = 8.396 A, Fd-3m)",
            f"crystallite size D = {cfg.GROUND_TRUTH['xrd_crystallite_size_nm']} nm "
            "(Scherrer broadening)",
            f"uniform microstrain eps = {cfg.GROUND_TRUTH['xrd_microstrain']:.1e}",
            f"Gaussian instrumental FWHM = "
            f"{cfg.GROUND_TRUTH['xrd_instrumental_fwhm_deg']} deg (quadrature)",
            "Lorentz-polarisation weighting; decaying Fe-fluorescence background",
            "Poisson counting noise",
        ])

    # A machine-readable record of what was injected -- used by validate.py
    pd.DataFrame(peak_log).to_csv(
        os.path.join(cfg.RAW_DIR, "_xrd_ground_truth_peaks.csv"), index=False)
    print("  wrote data/raw/_xrd_ground_truth_peaks.csv  (generator record)")

    print("\n[2/5] FTIR")
    ftir = generate_ftir(rngs[1])
    write_csv_with_header(
        ftir, os.path.join(cfg.RAW_DIR, "ftir_data.csv"),
        "FTIR (ATR, 4000-400 cm-1)",
        [
            "Gaussian/Lorentzian absorbance bands -> transmittance",
            "bands: 3396 nu(O-H), 1628 delta(H-O-H), 1045 surface Fe-OH,",
            "       571 + 441 nu(Fe-O) spinel lattice modes",
            "2341 cm-1 band is a DELIBERATE atmospheric CO2 artefact",
            "no C-H / C=O bands: design assumes NO capping agent",
            "curved baseline drift + wavenumber-dependent detector noise",
        ])

    print("\n[3/5] UV-Vis")
    uvvis = generate_uvvis(rngs[2])
    write_csv_with_header(
        uvvis, os.path.join(cfg.RAW_DIR, "uvvis_data.csv"),
        "UV-Vis (aqueous dispersion, 1 cm path)",
        [
            "O(2p)->Fe(3d) charge-transfer edge (saturating logistic)",
            "Urbach-type exponential sub-edge tail",
            "Rayleigh/Mie turbidity term proportional to lambda^-2.8",
            "weak ligand-field feature near 640 nm",
            "NOT designed to contain a well-defined optical band gap",
        ])

    print("\n[4/5] SEM particle measurements")
    sem = generate_sem(rngs[3])
    write_csv_with_header(
        sem, os.path.join(cfg.RAW_DIR, "sem_particle_measurements.csv"),
        "SEM-derived particle measurements (NO micrograph fabricated)",
        [
            f"lognormal equivalent diameter: median = "
            f"{cfg.GROUND_TRUTH['sem_lognormal_median_nm']} nm, "
            f"GSD = {cfg.GROUND_TRUTH['sem_lognormal_gsd']}",
            f"n = {cfg.GROUND_TRUTH['sem_n_particles']} features",
            "aspect ratio and circularity weakly correlated with size",
            "represents post-segmentation output, e.g. from ImageJ/Fiji",
        ])

    print("\n[5/5] TGA")
    tga = generate_tga(rngs[4])
    write_csv_with_header(
        tga, os.path.join(cfg.RAW_DIR, "tga_data.csv"),
        "TGA (N2 assumed, 10 C/min, 25-800 C)",
        [
            "three logistic mass-loss steps at ~78, ~243 and ~522 C",
            "amplitudes 2.60 / 1.70 / 0.60 % of initial mass",
            "NO organic-decomposition step: design assumes no capping agent",
            "INERT atmosphere assumed -> no oxidative mass gain",
            "buoyancy drift + microbalance noise included",
        ])

    print("\n" + "=" * 70)
    print("Done. All datasets are synthetic and labelled as such.")
    print("=" * 70)


if __name__ == "__main__":
    main()
