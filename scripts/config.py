"""
config.py
=========
Central configuration for the iron-oxide nanoparticle characterization
DEMONSTRATION project.

IMPORTANT -- SYNTHETIC DATA DISCLAIMER
--------------------------------------
Every dataset in this project is SYNTHETIC. The numbers below are the
"ground-truth" generative parameters used to build the demonstration data.
They are *not* measurements. No physical sample was synthesized and no
instrument was operated to produce anything in this repository.

The ground-truth block exists for one reason: it allows the analysis
pipeline to be validated. If the Scherrer / Williamson-Hall routines are
implemented correctly they must recover the values that were injected here.
`scripts/validate.py` performs exactly that check.

Author: portfolio demonstration project
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

for _d in (RAW_DIR, PROCESSED_DIR, FIGURE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

# Single seed for the whole project -> fully reproducible datasets.
RANDOM_SEED = 20240517

# --------------------------------------------------------------------------
# Hypothetical (NOT performed) sample context
# --------------------------------------------------------------------------
# The demonstration datasets are designed to be internally consistent with
# the following *hypothetical* scenario. This scenario was never carried out;
# it exists only to give the synthetic data a coherent physical story.
SAMPLE_CONTEXT = {
    "material": "Fe3O4 (magnetite), cubic inverse spinel, space group Fd-3m (No. 227)",
    "hypothetical_route": (
        "Surfactant-free aqueous co-precipitation of Fe(II)/Fe(III) chloride "
        "salts at a 1:2 molar ratio in excess NH4OH, 80 C, under N2, followed "
        "by magnetic decantation, washing to neutral pH, and vacuum drying at 60 C."
    ),
    "capping_agent": None,  # deliberately absent -> constrains the TGA design
    "capping_agent_note": (
        "No organic capping agent is assumed. Consequently the synthetic TGA "
        "dataset contains NO organic-decomposition step, and the synthetic FTIR "
        "dataset contains NO C-H / C=O bands. This is a design constraint, not "
        "an observation."
    ),
}

# --------------------------------------------------------------------------
# Crystallographic constants (literature reference values)
# --------------------------------------------------------------------------
# Magnetite lattice parameter. Reference value, not a measurement.
#   Fleet, M. E. Acta Crystallogr. B 1981, 37, 917-920.
#   ICDD PDF 19-0629 (magnetite).
A_MAGNETITE_ANG = 8.396          # angstrom
# Maghemite (gamma-Fe2O3) reference, used only for the phase-discrimination
# discussion.  Cornell & Schwertmann, The Iron Oxides, 2nd ed., 2003.
A_MAGHEMITE_ANG = 8.346          # angstrom

# Cu K-alpha1 radiation. The synthetic pattern is generated as if a
# monochromator / K-alpha2 stripping had already been applied, so a SINGLE
# wavelength is used throughout. This is stated explicitly in the report.
WAVELENGTH_ANG = 1.54060         # angstrom  (Cu K-alpha1)
WAVELENGTH_NM = WAVELENGTH_ANG / 10.0

SCHERRER_K = 0.9                 # spherical-crystallite shape factor

# Reflections used for the synthetic magnetite pattern.
# (h, k, l, relative intensity). Relative intensities follow the general
# pattern of ICDD PDF 19-0629 for magnetite.
MAGNETITE_REFLECTIONS = [
    (2, 2, 0, 30.0),
    (3, 1, 1, 100.0),
    (2, 2, 2, 8.0),
    (4, 0, 0, 20.0),
    (4, 2, 2, 10.0),
    (5, 1, 1, 30.0),
    (4, 4, 0, 40.0),
    (6, 2, 0, 4.0),
    (5, 3, 3, 10.0),
    (6, 2, 2, 4.0),
    (4, 4, 4, 3.0),
]

# --------------------------------------------------------------------------
# GROUND TRUTH used to synthesize the data (for validation only)
# --------------------------------------------------------------------------
GROUND_TRUTH = {
    # --- XRD ---
    "xrd_crystallite_size_nm": 12.0,      # volume-weighted mean, injected
    "xrd_microstrain": 8.0e-4,            # dimensionless, injected
    "xrd_instrumental_fwhm_deg": 0.080,   # Gaussian instrumental broadening
    "xrd_eta_pseudovoigt": 0.65,          # Lorentzian fraction of the profile
    "xrd_two_theta_zero_offset_deg": 0.0,
    "xrd_lattice_parameter_ang": A_MAGNETITE_ANG,

    # --- SEM particle sizes ---
    "sem_n_particles": 84,
    "sem_lognormal_median_nm": 18.5,      # geometric mean of the lognormal
    "sem_lognormal_gsd": 1.28,            # geometric standard deviation

    # --- TGA (mass-loss step amplitudes, % of initial mass) ---
    "tga_step1_pct": 2.60,   # physisorbed / interparticle water
    "tga_step2_pct": 1.70,   # strongly bound water + surface dehydroxylation
    "tga_step3_pct": 0.60,   # residual dehydroxylation / lattice water
    "tga_atmosphere": "N2 (inert) -- assumed, so NO oxidative mass gain",
    "tga_heating_rate_C_per_min": 10.0,
    "tga_initial_mass_mg": 9.84,
}

# --------------------------------------------------------------------------
# Instrument-style metadata for the HYPOTHETICAL acquisition
# --------------------------------------------------------------------------
# These describe the acquisition conditions the synthetic data were *designed
# to imitate*. They are simulation parameters, not instrument logs.
ACQUISITION = {
    "xrd": {
        "radiation": "Cu K-alpha1 (monochromated), 1.54060 angstrom",
        "range_2theta_deg": (20.0, 80.0),
        "step_deg": 0.02,
        "geometry": "Bragg-Brentano reflection",
    },
    "ftir": {
        "mode": "ATR (single-reflection diamond), assumed",
        "range_cm-1": (4000, 400),
        "resolution_cm-1": 4,
        "step_cm-1": 1.0,
    },
    "uvvis": {
        "mode": "dilute aqueous dispersion, 1 cm path length, assumed",
        "range_nm": (200, 900),
        "step_nm": 1.0,
    },
    "sem": {
        "mode": "measurements extracted from a hypothetical micrograph",
        "note": "NO image is presented as an experimental micrograph.",
    },
    "tga": {
        "atmosphere": "N2, 50 mL/min (assumed)",
        "heating_rate_C_per_min": 10.0,
        "range_C": (25, 800),
        "step_C": 0.5,
        "crucible": "Pt pan (assumed)",
    },
}

# --------------------------------------------------------------------------
# Figure style -- publication-oriented defaults
# --------------------------------------------------------------------------
FIGURE_DPI = 400
FIGURE_FORMATS = ("png", "pdf")

# A restrained, colour-blind-safe qualitative palette.
COLORS = {
    "raw": "#9AA3AE",
    "processed": "#1B3B6F",
    "accent": "#C1121F",
    "fit": "#E07A00",
    "background": "#6B8F71",
    "neutral": "#2B2B2B",
    "highlight": "#5B2C83",
}


def apply_plot_style() -> None:
    """Apply consistent publication-style Matplotlib settings."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "mathtext.default": "regular",
        # Embed TrueType (Type 42) rather than Type 3 fonts. Matplotlib
        # defaults to Type 3, which many publishers reject outright and which
        # some PDF viewers render poorly. Type 42 also keeps text selectable
        # and editable in Illustrator/Inkscape.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def two_theta_from_hkl(h: int, k: int, l: int,
                       a_ang: float = A_MAGNETITE_ANG,
                       wavelength_ang: float = WAVELENGTH_ANG) -> float:
    """Bragg angle 2-theta (degrees) for a cubic reflection.

    d(hkl) = a / sqrt(h^2 + k^2 + l^2)      [cubic system]
    lambda = 2 d sin(theta)                 [Bragg's law, n = 1]
    """
    import numpy as np

    d = a_ang / np.sqrt(h * h + k * k + l * l)
    sin_theta = wavelength_ang / (2.0 * d)
    if not -1.0 <= sin_theta <= 1.0:
        raise ValueError(f"Reflection ({h}{k}{l}) is outside the Ewald limit.")
    return float(2.0 * np.degrees(np.arcsin(sin_theta)))


DISCLAIMER_SHORT = "Synthetic portfolio dataset - not experimental data"
DISCLAIMER_LONG = (
    "DEMONSTRATION DATASET. Synthetically generated for portfolio purposes. "
    "These values were not measured on any instrument and do not describe a "
    "real sample."
)
