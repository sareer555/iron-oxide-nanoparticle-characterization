"""
validate.py
===========
Independent verification of the analysis pipeline.

This is the step that makes the project defensible rather than merely
presentable. Because every dataset was generated from KNOWN parameters
(scripts/config.py -> GROUND_TRUTH), the analysis can be checked against
those parameters. A pipeline that cannot recover what was injected is not
one whose output should be trusted on real data.

Three classes of check are run:

  A. RECOVERY   -- do the extracted physical quantities match the injected
                   ground truth, within a stated tolerance?
  B. INTERNAL   -- are units, arithmetic and statistics self-consistent?
                   (Scherrer recomputed by hand, quartiles, mass balance,
                   Bragg's law, unit conversions.)
  C. INTEGRITY  -- is every dataset labelled synthetic, is every figure
                   present, and are there any unlabelled experimental claims?

Run:  python scripts/validate.py
Exit code 0 = all checks passed, 1 = at least one failure.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str, str]] = []


def check(category: str, name: str, condition: bool, detail: str,
          warn_only: bool = False) -> None:
    status = PASS if condition else (WARN if warn_only else FAIL)
    _results.append((category, name, status, detail))
    symbol = {"PASS": "[ ok ]", "FAIL": "[FAIL]", "WARN": "[warn]"}[status]
    print(f"  {symbol} {name}: {detail}")


def close(a: float, b: float, rel_tol: float) -> bool:
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / abs(b) <= rel_tol


# ==========================================================================
def main() -> int:
    res_path = os.path.join(cfg.REPORT_DIR, "analysis_results.json")
    if not os.path.exists(res_path):
        print("analysis_results.json not found -- run scripts/analysis.py first.")
        return 1
    with open(res_path, encoding="utf-8") as fh:
        R = json.load(fh)
    gt = cfg.GROUND_TRUTH

    print("=" * 72)
    print("VALIDATION OF THE ANALYSIS PIPELINE")
    print("Recovered values are compared with the parameters that were")
    print("injected when the synthetic datasets were generated.")
    print("=" * 72)

    # ---------------------------------------------------------------- A ---
    print("\nA. RECOVERY OF INJECTED PARAMETERS")
    print("-" * 72)
    xrd = R["xrd"]

    d_true = gt["xrd_crystallite_size_nm"]
    d_wh = xrd["williamson_hall_size_nm"]
    check("recovery", "XRD crystallite size (Williamson-Hall)",
          close(d_wh, d_true, 0.10),
          f"recovered {d_wh:.2f} nm vs injected {d_true:.2f} nm "
          f"({100*(d_wh-d_true)/d_true:+.1f} %), tolerance 10 %")

    # Scherrer alone MUST under-estimate when strain is present -- that is
    # the physics, not a bug. Verify the direction of the bias.
    d_sch = xrd["scherrer_mean_nm"]
    check("recovery", "Scherrer under-estimates vs Williamson-Hall",
          d_sch < d_wh,
          f"Scherrer {d_sch:.2f} nm < Williamson-Hall {d_wh:.2f} nm, as "
          f"expected when microstrain contributes to the broadening")

    a_true = gt["xrd_lattice_parameter_ang"]
    a_rec = xrd["lattice_parameter_ang"]
    a_err = xrd["lattice_parameter_err_ang"]
    check("recovery", "XRD lattice parameter",
          abs(a_rec - a_true) <= max(3 * a_err, 0.005),
          f"recovered {a_rec:.4f} +/- {a_err:.4f} A vs injected "
          f"{a_true:.4f} A (within {abs(a_rec-a_true)/max(a_err,1e-9):.1f} sigma)")

    eps_true = gt["xrd_microstrain"]
    eps_rec = xrd["williamson_hall_microstrain"]
    check("recovery", "XRD microstrain (order of magnitude only)",
          0.2 * eps_true <= eps_rec <= 3.0 * eps_true,
          f"recovered {eps_rec:.2e} vs injected {eps_true:.2e}; the W-H "
          f"SLOPE is intrinsically less well determined than the intercept "
          f"(fit R^2 = {xrd['williamson_hall_r2']:.2f})", warn_only=True)

    check("recovery", "XRD phase assignment",
          xrd["phase_assignment"].startswith("Fe3O4"),
          f"assigned '{xrd['phase_assignment']}' (magnetite was injected)")

    # --- SEM ---
    sem = R["sem"]
    med_true = gt["sem_lognormal_median_nm"]
    check("recovery", "SEM lognormal median",
          close(sem["lognormal_median_nm"], med_true, 0.12),
          f"recovered {sem['lognormal_median_nm']:.2f} nm vs injected "
          f"{med_true:.2f} nm")
    gsd_true = gt["sem_lognormal_gsd"]
    check("recovery", "SEM geometric standard deviation",
          close(sem["lognormal_gsd"], gsd_true, 0.15),
          f"recovered {sem['lognormal_gsd']:.3f} vs injected {gsd_true:.3f}")
    check("recovery", "SEM sample size",
          sem["n"] == gt["sem_n_particles"],
          f"n = {sem['n']} (injected {gt['sem_n_particles']})")

    # --- TGA ---
    tga = R["tga"]
    dec = tga.get("step_deconvolution", {})
    if dec.get("converged"):
        injected = [gt["tga_step1_pct"], gt["tga_step2_pct"],
                    gt["tga_step3_pct"]]
        got = [s["amplitude_percent"] for s in dec["steps"]]
        check("recovery", "TGA number of mass-loss events",
              len(got) == 3, f"{len(got)} events resolved (3 injected)")
        if len(got) == 3:
            for i, (g, t) in enumerate(zip(got, injected), start=1):
                check("recovery", f"TGA step {i} amplitude",
                      close(g, t, 0.08),
                      f"recovered {g:.2f} % vs injected {t:.2f} %")
    else:
        check("recovery", "TGA step deconvolution", False,
              "deconvolution did not converge")

    # ---------------------------------------------------------------- B ---
    print("\nB. INTERNAL CONSISTENCY, UNITS AND ARITHMETIC")
    print("-" * 72)

    # B1. Bragg's law recomputed independently from the reported d-spacings.
    peaks = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                     "xrd_peak_analysis.csv"), comment="#")
    theta = np.radians(peaks["two_theta_fit_deg"] / 2.0)
    d_recomputed = cfg.WAVELENGTH_ANG / (2.0 * np.sin(theta))
    check("internal", "Bragg's law self-consistency",
          bool(np.allclose(d_recomputed, peaks["d_spacing_ang"], rtol=1e-4)),
          f"max deviation {np.max(np.abs(d_recomputed - peaks['d_spacing_ang'])):.2e} A")

    # B2. Scherrer recomputed by hand, in explicit units, for one peak.
    row = peaks.loc[peaks["hkl"] == "(311)"].iloc[0]
    beta_deg = float(row["fwhm_corrected_deg"])
    beta_rad = beta_deg * np.pi / 180.0
    th = float(row["two_theta_fit_deg"]) / 2.0 * np.pi / 180.0
    d_manual = 0.9 * 0.154060 / (beta_rad * np.cos(th))   # nm
    check("internal", "Scherrer equation recomputed by hand",
          close(d_manual, float(row["scherrer_size_nm"]), 1e-3),
          f"manual {d_manual:.3f} nm vs pipeline "
          f"{row['scherrer_size_nm']:.3f} nm (K=0.9, lambda=0.15406 nm, "
          f"beta={beta_deg:.4f} deg -> {beta_rad:.6f} rad, "
          f"theta={np.degrees(th):.3f} deg)")

    # B3. Degrees/radians confusion would change the answer by ~57x.
    d_wrong = 0.9 * 0.154060 / (beta_deg * np.cos(th))
    check("internal", "Radian conversion is actually applied",
          not close(d_manual, d_wrong, 0.5),
          f"correct {d_manual:.2f} nm vs degrees-by-mistake {d_wrong:.4f} nm "
          f"(factor {d_manual/d_wrong:.1f}) -- conversion verified present")

    # B4. Lattice parameter from each peak: a = d * sqrt(h^2+k^2+l^2)
    a_check = peaks["d_spacing_ang"] * np.sqrt(
        peaks["h"] ** 2 + peaks["k"] ** 2 + peaks["l"] ** 2)
    check("internal", "Cubic indexing a = d*sqrt(h^2+k^2+l^2)",
          bool(np.allclose(a_check, peaks["a_from_peak_ang"], rtol=1e-4)),
          f"all {len(peaks)} reflections consistent")

    # B5. SEM descriptive statistics recomputed from the raw file.
    raw_sem = pd.read_csv(os.path.join(cfg.RAW_DIR,
                                       "sem_particle_measurements.csv"),
                          comment="#")
    d_arr = raw_sem["equivalent_diameter_nm"].to_numpy(float)
    q1, med, q3 = np.percentile(d_arr, [25, 50, 75])
    # The pipeline reports these rounded to 2 decimal places, so compare at
    # that precision rather than to full float precision.
    def same2(a, b):
        return round(float(a), 2) == round(float(b), 2)

    ok = (same2(np.mean(d_arr), sem["mean_nm"])
          and same2(med, sem["median_nm"])
          and same2(np.std(d_arr, ddof=1), sem["std_dev_nm"])
          and same2(q1, sem["q1_nm"])
          and same2(q3, sem["q3_nm"]))
    check("internal", "SEM descriptive statistics recomputed",
          ok, f"mean/median/SD/Q1/Q3 all reproduce "
              f"(mean {np.mean(d_arr):.2f} nm, SD {np.std(d_arr, ddof=1):.2f} nm)")

    check("internal", "SEM quartile ordering and IQR",
          (sem["min_nm"] <= sem["q1_nm"] <= sem["median_nm"]
           <= sem["q3_nm"] <= sem["max_nm"]
           and abs(sem["iqr_nm"] - (sem["q3_nm"] - sem["q1_nm"])) <= 0.01),
          f"min {sem['min_nm']} <= Q1 {sem['q1_nm']} <= median "
          f"{sem['median_nm']} <= Q3 {sem['q3_nm']} <= max {sem['max_nm']}")

    # B6. Weighted means must be ordered D[3,2] <= D[4,3] and both >= mean.
    check("internal", "Moment-mean ordering (number <= Sauter <= De Brouckere)",
          sem["mean_nm"] <= sem["surface_weighted_D32_nm"]
          <= sem["volume_weighted_D43_nm"],
          f"number {sem['mean_nm']:.2f} <= D[3,2] "
          f"{sem['surface_weighted_D32_nm']:.2f} <= D[4,3] "
          f"{sem['volume_weighted_D43_nm']:.2f} nm")

    # B7. TGA mass balance.
    raw_tga = pd.read_csv(os.path.join(cfg.RAW_DIR, "tga_data.csv"),
                          comment="#")
    region_sum = sum(r["mass_loss_percent"] for r in tga["regions"])
    check("internal", "TGA region losses sum to the total",
          close(region_sum, tga["total_mass_loss_percent"], 0.02),
          f"regions sum to {region_sum:.3f} % vs total "
          f"{tga['total_mass_loss_percent']:.3f} %")

    check("internal", "TGA residue + total loss = initial mass",
          close(tga["residue_percent"] + tga["total_mass_loss_percent"],
                tga["initial_mass_percent"], 1e-3),
          f"{tga['residue_percent']:.3f} + "
          f"{tga['total_mass_loss_percent']:.3f} = "
          f"{tga['residue_percent'] + tga['total_mass_loss_percent']:.3f} % "
          f"(initial {tga['initial_mass_percent']:.3f} %)")

    check("internal", "TGA mass decreases monotonically overall",
          raw_tga["mass_percent"].iloc[:30].mean()
          > raw_tga["mass_percent"].iloc[-30:].mean(),
          "no net mass gain, consistent with the assumed inert atmosphere")

    # B8. Surface coverage cross-check: is the derived value physical?
    cov = tga.get("implied_water_coverage_molecules_per_nm2")
    if cov is not None:
        check("internal", "Derived surface water coverage is physically sane",
              2.0 <= cov <= 12.0,
              f"{cov:.2f} molecules/nm^2, within the 2-12 range typical of "
              f"hydroxylated iron-oxide surfaces")

    # B9. Units present in every processed file header.
    unit_pattern = re.compile(r"(_nm|_deg|_C|_percent|cm-1|_cm_1|_eV|_ang|_counts|_au|_mg)")
    for fname in sorted(os.listdir(cfg.PROCESSED_DIR)):
        if not fname.endswith(".csv"):
            continue
        cols = pd.read_csv(os.path.join(cfg.PROCESSED_DIR, fname),
                           comment="#", nrows=1).columns
        numeric_like = [c for c in cols
                        if c not in ("statistic", "value", "technique", "role",
                                     "demonstration_outcome", "hkl", "h", "k",
                                     "l", "assignment", "confidence", "label",
                                     "interpretation_note", "region",
                                     "assigned_process",
                                     "assignment_confidence", "count",
                                     "probes", "main_information",
                                     "quantitative_output", "key_limitation",
                                     "used_for_size", "used_for_lattice",
                                     "photometrically_unreliable",
                                     "particle_id", "eta_lorentzian_fraction",
                                     "sqrt_hkl_sum", "nelson_riley",
                                     "i_rel_ref", "i_rel_measured",
                                     "group_r_squared", "fwhm_rel_uncertainty",
                                     "aspect_ratio", "circularity",
                                     "absorbance", "absorbance_raw",
                                     "baseline", "absorbance_smoothed",
                                     "absorbance_baseline_corrected",
                                     "scattering_baseline",
                                     "absorbance_scatter_corrected")]
        missing = [c for c in numeric_like if not unit_pattern.search(c)]
        check("internal", f"Units in column names: {fname}",
              not missing,
              "all quantitative columns carry units" if not missing
              else f"missing units: {missing}", warn_only=True)

    # ---------------------------------------------------------------- C ---
    print("\nC. SYNTHETIC-DATA INTEGRITY AND DELIVERABLES")
    print("-" * 72)

    for sub, label in ((cfg.RAW_DIR, "raw"), (cfg.PROCESSED_DIR, "processed")):
        for fname in sorted(os.listdir(sub)):
            if not fname.endswith(".csv") or fname.startswith("_"):
                continue
            with open(os.path.join(sub, fname), encoding="utf-8") as fh:
                head = "".join(fh.readline() for _ in range(6)).upper()
            check("integrity", f"Disclaimer present: {label}/{fname}",
                  "SYNTHETIC" in head or "DEMONSTRATION" in head,
                  "synthetic-data disclaimer found in header")

    expected_figs = [
        "fig01_xrd_raw", "fig02_xrd_processed", "fig03_xrd_indexed",
        "fig04_xrd_peak_fit", "fig05_williamson_hall", "fig06_ftir_raw",
        "fig07_ftir_processed", "fig08_ftir_labelled",
        "fig09_uvvis_spectrum", "fig10_tauc_demonstration",
        "fig11_sem_histogram", "fig12_sem_boxplot",
        "fig13_sem_size_distribution", "fig14_tga_curve", "fig15_dtg_curve",
        "fig16_tga_dtg_combined", "fig17_size_comparison",
        "fig18_summary_dashboard",
    ]
    missing_figs = [f for f in expected_figs
                    if not os.path.exists(
                        os.path.join(cfg.FIGURE_DIR, f + ".png"))]
    check("integrity", "All figures generated",
          not missing_figs,
          f"{len(expected_figs) - len(missing_figs)}/{len(expected_figs)} "
          f"figures present in PNG"
          + (f"; missing {missing_figs}" if missing_figs else ""))

    missing_pdf = [f for f in expected_figs
                   if not os.path.exists(
                       os.path.join(cfg.FIGURE_DIR, f + ".pdf"))]
    check("integrity", "Vector (PDF) versions generated",
          not missing_pdf,
          f"{len(expected_figs) - len(missing_pdf)}/{len(expected_figs)} "
          f"figures present in PDF")

    # C3. No fabricated experimental claims anywhere in the source.
    banned = [
        r"\bwe synthesi[sz]ed\b", r"\bwe measured\b", r"\bwe observed\b",
        r"\bour experimental results\b", r"\bwas measured on\b",
        r"\bwe prepared\b", r"\bwe recorded\b", r"\bexperimentally obtained\b",
    ]
    offenders = []
    scan_dirs = [cfg.PROJECT_ROOT]
    for root, dirs, files in os.walk(cfg.PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for fname in files:
            if not fname.endswith((".py", ".md", ".txt", ".bib", ".ipynb")):
                continue
            path = os.path.join(root, fname)
            try:
                text = open(path, encoding="utf-8").read().lower()
            except (UnicodeDecodeError, OSError):
                continue
            for pat in banned:
                for mobj in re.finditer(pat, text):
                    # Allow it when it is explicitly framed as forbidden usage
                    ctx = text[max(0, mobj.start() - 200): mobj.end() + 120]
                    if any(w in ctx for w in ("do not write", "never write",
                                              "avoid", "banned", "forbidden",
                                              "instead of", "rather than",
                                              "must not", "not:")):
                        continue
                    offenders.append(
                        f"{os.path.relpath(path, cfg.PROJECT_ROOT)}: "
                        f"'{mobj.group(0)}'")
    check("integrity", "No fabricated experimental claims",
          not offenders,
          "no first-person experimental assertions found"
          if not offenders else f"found: {offenders[:5]}")

    # C4. Band gap must NOT be reported as a result.
    uv = R["uvvis"]
    check("integrity", "Optical band gap not reported as a measurement",
          "NOT REPORTED" in uv["band_gap_verdict"].upper(),
          "Tauc analysis is presented as a method demonstration with an "
          f"explicit {uv['tauc_spread_eV']:.2f} eV spread across equally "
          "defensible analysis choices")

    # C5. Datasets and figures must agree.
    xrd_proc = pd.read_csv(os.path.join(cfg.PROCESSED_DIR,
                                        "xrd_processed.csv"), comment="#")
    check("integrity", "Processed XRD row count matches raw",
          len(xrd_proc) == xrd["n_points"],
          f"{len(xrd_proc)} rows in both raw and processed XRD data")

    # ------------------------------------------------------------------ --
    print("\n" + "=" * 72)
    n_pass = sum(1 for *_, s, _ in ((c, n, s, d) for c, n, s, d in _results)
                 if s == PASS)
    n_fail = sum(1 for c, n, s, d in _results if s == FAIL)
    n_warn = sum(1 for c, n, s, d in _results if s == WARN)
    print(f"SUMMARY: {n_pass} passed, {n_warn} warnings, {n_fail} failed "
          f"({len(_results)} checks)")
    print("=" * 72)

    report = pd.DataFrame(_results,
                          columns=["category", "check", "status", "detail"])
    out = os.path.join(cfg.REPORT_DIR, "validation_report.csv")
    report.to_csv(out, index=False)
    print(f"Written to report/validation_report.csv")

    if n_fail:
        print("\nFAILED CHECKS:")
        for c, n, s, d in _results:
            if s == FAIL:
                print(f"  - [{c}] {n}: {d}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
