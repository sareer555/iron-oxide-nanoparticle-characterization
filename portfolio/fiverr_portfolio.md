# Fiverr Portfolio Presentation

**Project:** Comprehensive Characterization and Python-Based Data Analysis of
Iron Oxide Nanoparticles

---

> ### 🔒 Honesty guardrail — applies to every word below
>
> This project demonstrates **data analysis and interpretation**. It does not
> demonstrate instrument operation, and nothing in this document claims
> otherwise.
>
> **Never say or imply:** that you ran an XRD, FTIR, UV–Vis, SEM or TGA
> instrument; that you synthesized these nanoparticles; that these are
> experimental results; that you have hands-on instrumentation experience.
>
> **Do say:** that you analyse and interpret characterization data; that you
> build reproducible Python pipelines; that you know the physics behind the
> equations and the limits of each method.
>
> If a client asks directly whether you have operated the instruments, answer
> plainly: *"No — my background is MSc Chemistry (Physical Chemistry), with
> research experience on iron oxide nanoparticles. What I offer is the data
> analysis, interpretation and reporting side. I work from your instrument
> exports."* That answer loses you nothing and protects your reputation
> permanently.

---

## A) Short project description

A complete, reproducible materials-characterization data-analysis workflow
covering five techniques — XRD, FTIR, UV–Vis, SEM-derived particle sizing and
TGA — built in Python and demonstrated end to end on synthetic Fe₃O₄
nanoparticle datasets. The project covers raw-data processing, background and
baseline correction, non-linear peak fitting, quantitative parameter
extraction with uncertainties, statistical analysis, publication-quality
figures, cross-technique reconciliation and a 15-page scientific report.
Because the datasets were generated from known parameters, the entire pipeline
is validated against ground truth: 55 of 55 automated checks pass.

---

## B) Three-sentence portfolio summary

I built a five-technique nanomaterials characterization analysis pipeline in
Python — XRD, FTIR, UV–Vis, SEM particle sizing and TGA — covering everything
from raw-file processing and baseline correction through non-linear peak
fitting, Scherrer and Williamson–Hall crystallite-size analysis, distribution
statistics and thermal-event deconvolution, to publication-quality figures and
a full scientific report. The workflow runs on clearly-labelled synthetic
datasets so that every extracted value can be checked against the parameters
that generated it: crystallite size, lattice parameter, particle-size
distribution and thermal mass-loss amplitudes are all recovered correctly, and
55 of 55 automated validation checks pass. The project also demonstrates
analytical judgement rather than just computation — it explains why the XRD
and SEM size metrics legitimately differ by a factor of 1.6, flags one
spectroscopic band as an instrument artefact, leaves another unassigned, and
declines to report an optical band gap because four equally defensible
analyses of the same data span 0.64–3.00 eV.

---

## C) Skills demonstrated

**Scientific data analysis**
- Raw instrument-format data processing and cleaning
- Background estimation (asymmetric least squares) and baseline correction
- Justified smoothing — Savitzky–Golay with filter widths chosen against
  feature widths, not by default
- Peak detection with noise-referenced thresholds
- Non-linear least-squares curve fitting (pseudo-Voigt, logistic, power-law)
- Simultaneous multi-peak fitting of overlapping features
- Analytical derivatives from local polynomial fits (DTG)
- Uncertainty propagation and weighted regression
- Model-based deconvolution as an alternative to boundary-dependent integration

**Materials characterization interpretation**
- Powder XRD indexing, phase identification and lattice-parameter refinement
  (Nelson–Riley extrapolation)
- Scherrer crystallite-size analysis with instrumental-broadening correction
- Williamson–Hall size/strain separation
- Vibrational band assignment with explicit confidence grading
- Optical spectroscopy of dispersions, including scattering correction
- Particle-size distribution statistics and moment means
- Thermogravimetric event resolution and assignment
- Cross-technique reconciliation of conflicting size metrics

**Statistics**
- Descriptive statistics, quartiles, Tukey outlier fences
- Distribution fitting by maximum likelihood; Kolmogorov–Smirnov testing
- Bootstrap confidence intervals (distribution-free)
- Number-, surface- and volume-weighted moment means
- Correct interpretation of non-significant test results

**Python engineering**
- Modular, documented, reproducible codebase driven by a single random seed
- Automated validation suite (55 checks) run against known ground truth
- Report generation that interpolates every number from a results file, so
  prose cannot drift from data
- Programmatic notebook construction and execution

**Scientific communication**
- Publication-quality figures (400 dpi PNG + vector PDF)
- A 15-page technical report with figures, tables and verified references
- Clear separation of what data show from what they cannot show

---

## D) Tools used

| Tool | Role |
|------|------|
| **Python 3.11** | Entire workflow |
| **NumPy** | Numerical computation, array operations |
| **pandas** | Data structures, tabular processing, CSV I/O |
| **SciPy** | `optimize.curve_fit`, `signal` (peak finding, Savitzky–Golay), `stats`, sparse linear algebra for ALS baselines |
| **Matplotlib** | All 18 publication-quality figures |
| **ReportLab** | Programmatic PDF report generation |
| **nbformat / nbclient** | Notebook construction and execution |
| **Crossref REST API** | Programmatic verification of every cited DOI |
| **scikit-learn** | *Installed but deliberately not used* — no step here is a machine-learning problem |

*That last row is worth keeping. Clients who know the field read unnecessary
ML as a red flag, and knowing when not to reach for a tool is itself a
selling point.*

---

## E) Deliverables

| Deliverable | Detail |
|-------------|--------|
| **5 raw datasets** | CSV, each with a documented generative model and disclaimer header |
| **11 processed datasets** | Peak tables, fitted parameters, statistics, region analyses, comparison matrices |
| **Executed Jupyter notebook** | 52 cells, sectioned narrative with inline outputs and figures |
| **Python codebase** | 6 modules: config, data generation, analysis, validation, report builder, notebook builder |
| **18 figures** | 400 dpi PNG + vector PDF, consistent styling |
| **15-page PDF report** | Abstract, 12 sections, 9 tables, 11 embedded figures, 34 references |
| **Validation report** | 55 automated checks with pass/fail detail, exported to CSV |
| **Bibliography** | 38 BibTeX entries, every DOI verified against Crossref |
| **README** | Purpose, disclaimer, workflow, outputs, limitations, reproduction instructions |

---

## F) Scientific methods

| Method | Applied to |
|--------|-----------|
| Bragg's law, cubic indexing (*d* = *a*/√(h²+k²+l²)) | XRD phase identification |
| Asymmetric least squares baseline (Eilers & Boelens) | XRD background, FTIR baseline |
| Pseudo-Voigt profile fitting (Thompson–Cox–Hastings form) | XRD peak shape |
| Scherrer equation, *D* = *Kλ*/(*β*cos*θ*) | Crystallite size |
| Quadrature instrumental-broadening correction | XRD width correction |
| Williamson–Hall analysis, *β*cos*θ* = *Kλ*/*D* + 4*ε*sin*θ* | Size/strain separation |
| Nelson–Riley extrapolation | Lattice-parameter refinement |
| Savitzky–Golay smoothing and differentiation | FTIR, UV–Vis, DTG |
| Group-frequency band assignment with confidence grading | FTIR |
| Empirical λ⁻ⁿ turbidity correction | UV–Vis dispersion spectra |
| Tauc analysis with multi-dimensional sensitivity scan | UV–Vis (as a cautionary demonstration) |
| Lognormal MLE fitting, KS testing, bootstrap CIs | SEM particle statistics |
| Moment means D[4,3], D[3,2] | Number vs volume weighting |
| Logistic step deconvolution | TGA event quantification |
| Surface-area/coverage calculation | Combined XRD + TGA |

---

## G) Key demonstration findings

> Phrase these as *demonstration outcomes*, never as experimental findings.

1. **Phase and structure.** The synthetic pattern indexes cleanly to cubic
   Fe₃O₄; Nelson–Riley refinement gives *a* = 8.3962 ± 0.0073 Å, one standard
   error from the magnetite reference and seven from maghemite.

2. **Crystallite size, done two ways.** Scherrer alone gives 11.12 ± 0.46 nm;
   Williamson–Hall, which separates strain from size, gives 11.94 ± 0.59 nm
   against an injected truth of 12.00 nm. The gap between them *is* the strain
   contribution that Scherrer misattributes to size.

3. **Honest uncertainty.** The Williamson–Hall *intercept* (size) is well
   constrained; the *slope* (strain) is not, with only five usable reflections.
   Microstrain is therefore reported as an order of magnitude (~10⁻⁴), not a
   value — the kind of restraint that distinguishes an analyst from a script.

4. **Artefact recognition.** Of six FTIR bands resolved, one is identified as
   atmospheric CO₂ — an instrument artefact, not a sample feature — and one
   weak band is explicitly left **unassigned** because the data cannot
   distinguish between three plausible origins.

5. **Evidence from absence.** Diagnostic bands for goethite, organics, nitrate
   and sulfate are shown absent using a resolved-local-maximum test, rather
   than the naive "is absorbance low here" test that a neighbouring band's
   tail would defeat.

6. **A refusal to report.** Four equally defensible Tauc analyses of the same
   UV–Vis data give apparent band gaps of 0.66, 2.23, 2.37 and 3.00 eV —
   every fit with R² ≥ 0.97. No band gap is reported, and the reasons are
   documented.

7. **Statistical restraint.** With n = 84, a KS test rejects neither a
   lognormal nor a normal model. The lognormal fit is reported as physically
   motivated, *not* as demonstrated by the data.

8. **Thermal events resolved.** Three mass-loss events totalling 4.65 %;
   boundary-independent deconvolution recovers the injected amplitudes
   (2.60 / 1.70 / 0.60 %) exactly.

9. **Cross-technique quantification.** Combining the XRD crystallite size with
   the TGA Region II loss gives ~5.4 H₂O molecules nm⁻², within the range
   quoted for hydroxylated iron-oxide surfaces — an internal consistency check
   neither technique could provide alone.

10. **The headline result: reconciling a disagreement.** SEM gives a
    number-mean of 19.1 nm against an XRD coherent-domain size of 11.9 nm — a
    factor of 1.60. The project explains why this is *expected*: different
    measurands, different weightings, surface disorder, agglomeration and
    imaging bias. Knowing this is why a client should not simply average the
    two numbers.

---

## H) Limitations

State these openly. Clients who matter will trust you more, not less.

- **All data are synthetic.** Generated from explicit physical models; they
  contain that physics and no more. Real specimens bring preferred
  orientation, anisotropic shapes, stacking faults, amorphous fractions,
  inhomogeneity and instrument drift.
- **No instrument was operated** and no material was synthesized in producing
  this project.
- **No SEM micrograph is presented.** Fabricating an image and calling it a
  micrograph would be misrepresentation regardless of disclaimers. The
  analysis works from a post-segmentation measurement table — the numeric
  input a real workflow produces.
- **Scherrer analysis** returns a volume-weighted coherent-domain size, not a
  particle size; *K* is uncertain at ±10 %; unreliable above ~100 nm.
- **Single-peak fitting** rather than Rietveld refinement, chosen to make the
  underlying methods explicit.
- **Kα₂** was not simulated; real laboratory patterns require doublet
  stripping before width analysis.
- **FTIR intensities** are not concentrations; ATR and KBr sampling differ.
- **UV–Vis** on a dispersion cannot give intrinsic electronic properties.
- **n = 84** constrains a mean but not distribution tails.
- **TGA** measures mass only; atmosphere, heating rate and packing are assumed.

---

## I) Suggested Fiverr portfolio title

**Primary:**
> Nanomaterials Characterization Data Analysis in Python — XRD, FTIR, UV–Vis,
> SEM & TGA

**Alternatives:**
- Python Data Analysis & Publication-Quality Figures for Materials
  Characterization
- XRD, FTIR, TGA & Particle-Size Data Analysis with Scientific Reporting
- Crystallite Size, Spectra & Thermal Data Analysis for Nanomaterials Research

---

## J) Suggested Fiverr portfolio description

> **Nanomaterials characterization data analysis in Python — XRD, FTIR,
> UV–Vis, SEM and TGA**
>
> I turn raw characterization exports into quantitative results,
> publication-ready figures and a report you can put in front of a supervisor,
> a reviewer or a client.
>
> This portfolio project demonstrates a complete five-technique workflow on
> iron oxide (Fe₃O₄) nanoparticles: background and baseline correction,
> non-linear peak fitting, Scherrer and Williamson–Hall crystallite-size
> analysis, spectroscopic band assignment, particle-size distribution
> statistics, and thermogravimetric event deconvolution — ending in 18
> publication-quality figures and a 15-page scientific report.
>
> **Please note:** the datasets in this demonstration are synthetic and
> clearly labelled as such. That is deliberate. Because the data were
> generated from known parameters, the entire analysis pipeline could be
> validated against ground truth — crystallite size, lattice parameter,
> size distribution and thermal mass-loss amplitudes are all recovered
> correctly, and 55 of 55 automated checks pass. You can see not just that the
> analysis produces numbers, but that the numbers are right.
>
> **What I offer is data analysis and interpretation, not instrument
> operation.** You provide the instrument output — I handle the processing,
> the quantification, the statistics, the figures and the write-up.
>
> **What you get:**
> - Cleaned, processed datasets with every step documented
> - Quantitative results with genuine uncertainties, not just point values
> - Publication-quality figures (300–400 dpi raster + vector) to your journal's
>   requirements
> - A clear written interpretation, including what your data *cannot* support
> - Fully reproducible Python code, so you can re-run or extend the analysis
>
> **Background:** MSc Chemistry (Physical Chemistry specialisation), with
> research experience on iron oxide nanoparticles. Python, pandas, NumPy,
> SciPy, scikit-learn, Matplotlib, Excel.
>
> One thing worth knowing about how I work: this project declines to report an
> optical band gap, because four equally defensible analyses of the same data
> disagree by more than 2 eV. I would rather tell you a number is unsupportable
> than hand you one that falls apart under review.
>
> Message me with your data and what you need — I'll tell you honestly what it
> can and cannot show before you order.

---

## K) Image / thumbnail concepts

**Recommended thumbnail — use `figures/fig18_summary_dashboard.png`.** It shows
all five techniques at once, is immediately legible as real scientific work,
and carries the synthetic-data notice.

| # | Concept | Source | Why it works |
|---|---------|--------|-------------|
| 1 | **Five-panel dashboard** — XRD, FTIR, UV–Vis, SEM histogram, TGA/DTG with a results table | `fig18_summary_dashboard.png` | Communicates breadth instantly; looks like a real report page |
| 2 | **Indexed XRD pattern** with Miller indices labelled | `fig03_xrd_indexed.png` | The single most recognisable materials-science image; signals domain credibility |
| 3 | **Before → after split** — raw noisy pattern beside processed, indexed result | `fig01` + `fig03` side by side | Shows the transformation the client is actually buying |
| 4 | **Peak fit with residuals** | `fig04_xrd_peak_fit.png` | Reads as genuine analysis rather than plotting; residual panel signals rigour |
| 5 | **Size-metric reconciliation** — violin plot with XRD/SEM markers | `fig17_size_comparison.png` | Differentiator: shows interpretation, not just computation |
| 6 | **Technique icon strip** — "XRD · FTIR · UV–Vis · SEM · TGA" over a faded spectrum | Custom, using `fig03` as background | Scannable in a crowded gig grid |

**Design rules for any custom thumbnail:**
- Large, legible text — most browsing happens on mobile
- At most 6 words of overlay text; e.g. *"XRD · FTIR · UV–Vis · SEM · TGA"*
- Keep the *"Synthetic demonstration data"* stamp visible; it is a credibility
  signal, not a weakness
- Avoid stock "science" imagery — bubbling flasks and glowing molecules read as
  amateur to technical buyers
- Consistent palette across all gig images (the project palette is in
  `scripts/config.py`)

---

## L) Recommended Fiverr Gig connection

### Which parts legitimately support your first gig

**Fully supported — lead with these:**

| Claim you can make | Evidence in this project |
|--------------------|-------------------------|
| "I process raw characterization data files" | 5 raw datasets loaded, cleaned, background/baseline corrected |
| "I fit peaks and extract quantitative parameters" | Pseudo-Voigt fitting with uncertainties, `xrd_peak_analysis.csv` |
| "I calculate crystallite size from XRD" | Scherrer + Williamson–Hall, validated against ground truth |
| "I produce publication-quality figures" | 18 figures, 400 dpi PNG + vector PDF |
| "I write technical scientific reports" | 15-page PDF with tables, figures, verified references |
| "I do particle-size distribution statistics" | Full descriptive + distributional analysis with bootstrap CIs |
| "I analyse TGA curves and compute DTG" | Region integration + step deconvolution |
| "I write reproducible Python analysis code" | Seeded, modular, documented, with a 55-check validation suite |
| "I verify references properly" | Every DOI resolved against Crossref |

**Supported with correct framing:**

| Claim | Correct framing |
|-------|----------------|
| Iron oxide nanoparticle expertise | "MSc Chemistry with research background in iron oxide nanoparticles" — true and stated. Do not extend it to instrument operation. |
| Five-technique characterization | "I analyse and interpret data from five techniques" — never "I performed five characterizations." |
| SEM analysis | "SEM image-derived particle-size statistics" — you analyse measurements extracted from micrographs. |

**Not supported — never claim:**

- Operating XRD, FTIR, UV–Vis, SEM or TGA instruments
- Synthesizing these nanoparticles
- That any result here is experimental
- Rietveld refinement (single-peak fitting was used instead)
- Machine learning applied to characterization data (deliberately absent)

---

### Recommended gig structure

**Gig title:** *I will analyse your XRD, FTIR, TGA and particle size data in
Python with publication-ready figures*

| Tier | Scope | Deliverables |
|------|-------|-------------|
| **Basic** | One technique, one sample | Processed data, 2–3 publication-quality figures, key quantitative results, short summary |
| **Standard** | Up to three techniques, or one technique across multiple samples | Above plus peak tables with uncertainties, comparative figures, 2–3 page interpretation |
| **Premium** | Full multi-technique package | Above plus cross-technique reconciliation, full report, reproducible Python code and notebook |

**Gig FAQ entries — write these yourself, they pre-empt the hard questions:**

- *"Have you operated these instruments?"* → "No. My background is MSc
  Chemistry (Physical Chemistry) with research on iron oxide nanoparticles. I
  specialise in the analysis and interpretation side, working from your
  instrument exports."
- *"Is the portfolio data real?"* → "No, and it says so throughout. It's
  synthetic, which let me validate the entire pipeline against known values —
  55 of 55 checks pass. On your project I work with your real data."
- *"What formats can you accept?"* → CSV, TXT, ASC, XY, DAT, XLSX, and most
  instrument text exports. Ask about binary formats.
- *"Can you do Rietveld refinement?"* → Be honest about your current answer.
  If not yet, say "not currently — I use single-peak profile fitting, which
  suits crystallite-size work; for full structure refinement you want a
  GSAS-II/FullProf specialist."

---

### How to use this project on your profile

1. **Upload the dashboard figure as the primary gig image** — it communicates
   scope in one glance.
2. **Attach the PDF report as a portfolio sample.** Its synthetic-data
   disclaimer on every page protects you, and its depth does the selling.
3. **Put the repository on GitHub** and link it. The validation suite is the
   strongest single credibility signal you have — very few freelancers can
   show that their analysis recovers known inputs.
4. **Lead your gig description with the validation result**, not the
   technique list. "55 of 55 automated checks pass" is a claim competitors
   cannot match; "I know XRD and FTIR" is one hundreds of people make.
5. **Keep the band-gap refusal in your description.** It is counter-intuitive
   marketing and it works: it signals that your numbers are trustworthy
   precisely because you are willing to withhold one.
6. **Next portfolio project** — build the same rigour on *real public data*
   (e.g. an open crystallographic or spectroscopic repository). Pairing this
   validated-synthetic project with a real-data project covers both "the
   method is correct" and "it works on messy real data," which together are
   close to unanswerable.
