# Comprehensive Characterization and Python-Based Data Analysis of Iron Oxide Nanoparticles

A portfolio demonstration of a complete, reproducible five-technique
materials-characterization workflow — XRD, FTIR, UV–Vis, SEM and TGA —
implemented in Python and applied to clearly-labelled synthetic datasets.

---

## ⚠️ Synthetic-data disclaimer — read this first

**Every dataset in this repository is synthetic.** All five datasets were
generated computationally by `scripts/generate_synthetic_datasets.py` from
explicit physical models plus realistic noise.

- No material was synthesized.
- No instrument was operated.
- No measurement was performed.
- No experimental result is reported or implied.

This project demonstrates **data-analysis and interpretation capability**, not
laboratory work. The author's contribution is the analytical pipeline, the
statistical treatment and the scientific interpretation — not instrument
operation. Nothing here may be cited as evidence about any real material.

Every raw and processed CSV carries this disclaimer in its header, every
figure carries a visible synthetic-data stamp, and every page of the PDF
report carries a footer stating the same.

---

## Why synthetic data is the right choice here

Using synthetic data is not a workaround — it enables something real data
cannot offer. Because the generative parameters are **known**, the analysis
can be **validated**: `scripts/validate.py` compares every extracted quantity
against the value that was injected. A pipeline that cannot recover known
inputs should not be trusted on unknown ones.

| Quantity | Injected | Recovered |
|----------|----------|-----------|
| Crystallite size (Williamson–Hall) | 12.00 nm | 11.94 nm |
| Lattice parameter | 8.3960 Å | 8.3962 ± 0.0073 Å |
| Microstrain | 8.0 × 10⁻⁴ | 5.5 × 10⁻⁴ (order of magnitude only) |
| SEM lognormal median | 18.50 nm | 18.68 nm |
| SEM geometric SD | 1.280 | 1.245 |
| TGA step amplitudes | 2.60 / 1.70 / 0.60 % | 2.60 / 1.70 / 0.60 % |

**55 of 55 automated checks pass** (parameter recovery, unit and arithmetic
consistency, synthetic-data integrity).

---

## The hypothetical scenario

The five datasets were designed around a single internally consistent
*hypothetical* scenario. **This scenario was never carried out** — it exists
only to constrain data generation so the datasets tell a coherent story:

- **Material:** Fe₃O₄ (magnetite), cubic inverse spinel, *Fd*3̄*m*, a = 8.396 Å
- **Assumed route:** surfactant-free aqueous co-precipitation of Fe(II)/Fe(III)
  chlorides (1:2) in excess NH₄OH at 80 °C under N₂ — a Massart-type
  preparation
- **No capping agent** — a binding constraint: it is *why* the TGA data contain
  no organic-decomposition step and the FTIR data contain no C–H or C=O bands
- **Inert TGA atmosphere (N₂)** — in air, Fe₃O₄ oxidation would produce a mass
  *gain* of up to +3.45 %, changing the curve entirely

---

## Techniques and what each contributes

| Technique | Probes | Quantitative output | Principal limitation |
|-----------|--------|--------------------|---------------------|
| **XRD** | Long-range crystalline order | Lattice parameter, crystallite size, microstrain | Volume-weighted; blind to amorphous material; cannot separate Fe₃O₄ from γ-Fe₂O₃ on peak position alone |
| **FTIR** | Local bonding, surface chemistry | Band positions, relative intensities | Heavily overlapped Fe–O bands; intensities are not concentrations |
| **UV–Vis** | Electronic transitions (+ scattering) | Absorbance spectrum | Dispersion optics conflate absorption with turbidity; no meaningful band gap for this material |
| **SEM** | Morphology of individual features | Size distribution statistics | Measures features, not primary particles; thresholding and sampling bias |
| **TGA** | Mass change on heating | Mass loss per event, DTG maxima | Mass only, never identity; depends entirely on atmosphere and heating rate |

---

## Analysis workflow

```
raw CSV  →  cleaning  →  background / baseline  →  feature detection
         →  quantitative model  →  validation  →  figures + report
```

**XRD** — ALS background subtraction → peak detection → cubic indexing →
simultaneous pseudo-Voigt profile fitting of overlapping groups →
instrumental-broadening correction in quadrature → Scherrer equation →
Williamson–Hall size/strain separation → Nelson–Riley lattice refinement.

**FTIR** — ALS baseline correction → justified Savitzky–Golay smoothing →
peak detection → confidence-graded band assignment → tests for *absent*
diagnostic bands.

**UV–Vis** — photometric-range flagging → empirical λ⁻ⁿ turbidity correction →
Tauc analysis performed in full **as a cautionary demonstration**, with an
explicit sensitivity scan over baseline, transition and linear-window choices.

**SEM** — descriptive statistics → Tukey outlier fences → lognormal MLE fit
with goodness-of-fit testing → bootstrap confidence intervals →
number/surface/volume moment means.

**TGA** — Savitzky–Golay smoothed analytical derivative (DTG) → region
integration → boundary-independent logistic step deconvolution → combination
with the XRD crystallite size to yield surface water coverage.

---

## Major demonstration outputs

- Diffraction pattern indexed to cubic Fe₃O₄; lattice parameter refined to
  **8.3962 ± 0.0073 Å** by Nelson–Riley extrapolation
- Crystallite size **11.94 ± 0.59 nm** (Williamson–Hall), versus
  **11.12 ± 0.46 nm** from Scherrer alone — the difference quantifies the
  strain contribution that Scherrer misattributes to size
- Six FTIR bands resolved and graded by confidence, including one identified
  as an **atmospheric CO₂ artefact** and one explicitly left **unassigned**
- **No optical band gap reported.** Four equally defensible Tauc analyses of
  the same data span **0.64–3.00 eV**, all with R² ≥ 0.97
- Particle population of 84 features characterised; the data are shown to be
  **unable to distinguish** lognormal from normal at this sample size
- Three thermal events resolved and quantified, totalling **4.65 %** mass loss
- **SEM/XRD size ratio of 1.60**, explained rather than explained away
- Cross-technique result: **~5.4 H₂O molecules nm⁻²** surface coverage, from
  combining the XRD crystallite size with the TGA Region II mass loss

---

## Repository structure

```
Iron_Oxide_Nanoparticle_Characterization/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     5 synthetic datasets + generator record
│   └── processed/              11 processed / derived tables
├── notebooks/
│   └── iron_oxide_characterization_analysis.ipynb   (executed, with outputs)
├── scripts/
│   ├── config.py                paths, constants, ground truth, plot style
│   ├── generate_synthetic_datasets.py
│   ├── analysis.py              the full pipeline
│   ├── validate.py              55 automated checks
│   ├── build_report.py          renders the PDF from analysis_results.json
│   └── make_notebook.py         builds and executes the notebook
├── figures/                    18 figures, PNG at 400 dpi (vector PDFs
│                                regenerate locally -- see note below)
├── report/
│   ├── iron_oxide_characterization_report.pdf
│   ├── analysis_results.json
│   └── validation_report.csv
├── references/
│   └── references.bib          38 references, every DOI verified via Crossref
└── portfolio/
    └── fiverr_portfolio.md
```

---

### A note on figure formats

`scripts/analysis.py` saves every figure as both a 400 dpi PNG and a vector
PDF. Only the PNGs are committed. GitHub's file viewer cannot preview
Matplotlib-generated PDFs -- it reports "Unable to render code block" even
though the files are perfectly valid -- so committing them makes the
repository look broken to anyone browsing it. Running the pipeline regenerates
the vector PDFs locally, identically every time, for print or journal
submission.

Figure PDFs embed subsetted Type 42 (TrueType) fonts rather than Matplotlib's
Type 3 default, which many publishers reject.

---

## Reproducing everything

```bash
pip install -r requirements.txt
```

```bash
python scripts/generate_synthetic_datasets.py && python scripts/analysis.py && python scripts/validate.py && python scripts/build_report.py && python scripts/make_notebook.py
```

A single random seed (`RANDOM_SEED = 20240517` in `scripts/config.py`) makes
every dataset, figure, number and page reproducible. The PDF report
interpolates all of its quantitative statements from
`report/analysis_results.json` at build time, so the prose cannot drift out of
step with the data.

**Tested with:** Python 3.11, NumPy 2.4, pandas 3.0, SciPy 1.17, Matplotlib 3.11.

---

## Limitations

**The overriding limitation:** these data are synthetic and contain exactly
the physics the generative models encode. Real specimens bring preferred
orientation, anisotropic crystallite shapes, stacking faults, amorphous
fractions invisible to diffraction, inhomogeneity and instrument drift. None
of that is reproduced here.

**Method-specific:**

- **Scherrer** returns a volume-weighted *coherent-domain* size, not a
  particle size; *K* is uncertain at the ±10 % level; unreliable above ~100 nm
- **Single-peak fitting** — Rietveld refinement of the full pattern would
  constrain parameters better; single-peak methods were used to demonstrate
  them explicitly
- **Kα₂** — the pattern was generated for a single wavelength; real laboratory
  data contain the doublet and must be stripped before width analysis
- **FTIR** band intensities are not concentrations; ATR and KBr sampling give
  different positions and relative intensities
- **UV–Vis** — no band gap is reportable for a mixed-valence conductor
  measured as a dispersion
- **SEM** — n = 84 constrains the mean but not the distribution tails;
  agglomeration, thresholding, coating and field-of-view selection all bias
  measured sizes upward
- **TGA** — measures mass only, never chemical identity; results depend on
  atmosphere, heating rate, sample mass and packing, all of which are assumed

---

## References

38 references in `references/references.bib`, covering the Scherrer and
Williamson–Hall relations, Nelson–Riley extrapolation, iron-oxide
crystallography and surface chemistry, Tauc analysis and its misuse, SEM
particle sizing, thermal-analysis practice, and the numerical methods and
software used.

**Every DOI was programmatically resolved against the Crossref REST API** and
the returned author, year, journal and title checked against the entry. Two
entries carry no DOI because none was ever issued (Scherrer 1918 predates the
system; JMLR does not mint DOIs). Nothing is cited from memory.

---

## What this project demonstrates

The ability to build a reproducible analytical pipeline; apply established
physical relations with correct unit handling; fit non-linear models and
propagate uncertainty; apply appropriate statistics and state their limits;
validate results against known inputs; distinguish genuine signal from
instrumental artefact; grade confidence in interpretations rather than
asserting them; and **decline to report a number when the data do not support
one**.

*Skills: chemistry research · nanomaterials characterization · scientific data
analysis · Python (pandas, NumPy, SciPy, Matplotlib) · scientific
visualization · research interpretation · publication-quality figures ·
technical scientific reporting.*
