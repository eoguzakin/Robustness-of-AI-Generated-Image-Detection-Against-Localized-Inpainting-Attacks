# Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks

Code and experimental pipelines for the thesis:

> **Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks**  
> Oğuz Akın — Saarland University / CISPA Helmholtz Center for Information Security (2025)

- Thesis PDF: [Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks.pdf](./Robustness%20of%20AI-Generated%20Image%20Detection%20Against%20Localized%20Inpainting%20Attacks.pdf)
- Curated datasets: https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks

---

## Overview

This work evaluates the **robustness of AI-generated image (AIGI) detectors** under **localized inpainting attacks**.
Unlike global post-processing, localized inpainting overwrites *only selected regions*, creating hybrid images that mix authentic and generated pixels (or watermarked and non-watermarked regions).

### Key findings (high level)

- A consistent dichotomy appears across detector families:
  - On edited real photos, many detectors still predict **Real** (high **ASRReal** / high “real pass rate”).
  - On edited fakes, many detectors remain robust: inpainting often does **not** dramatically improve evasion (low **ASRFake** / low “fake evasion rate”).
- Passive + training-free detectors (e.g., UFD, DIMD, WaRPAD, AEROBLADE) commonly show high pass rates on inpainted reals under a fixed-threshold policy.
- Watermarking methods (Tree-Ring, Stable Signature) tend to stay robust for small/moderate edits, but degrade more noticeably as inpainted area increases (e.g., 25–40% masks).
- Baseline AUC is not a reliable predictor of robustness.

**Conclusion:** robustness should be evaluated under a clear threat model and a fixed-threshold protocol; clean benchmark performance alone is insufficient for deployment decisions.

---

## Repository layout

```
.
├─ configs/
│  ├─ paths_aeroblade.yaml
│  ├─ paths_dimd.yaml
│  ├─ paths_stablesig.yaml
│  ├─ paths_treering.yaml
│  ├─ paths_ufd.yaml
│  └─ paths_warpad.yaml
├─ scripts/
│  ├─ config_utils.py
│  ├─ ufd_score_baseline.py
│  ├─ ufd_score_robust.py
│  ├─ ufd_analyze.py
│  ├─ ufd_common.py
│  ├─ dimd_score.py
│  ├─ dimd_score_baseline.py
│  ├─ dimd_score_robust.py
│  ├─ dimd_analyze.py
│  ├─ dimd_common.py
│  ├─ warpad_score_baseline.py
│  ├─ warpad_score_robust.py
│  ├─ warpad_analyze.py
│  ├─ aeroblade_score_baseline.py
│  ├─ aeroblade_score_robust.py
│  ├─ aeroblade_analyze.py
│  ├─ aeroblade_common.py
│  ├─ stablesig_score_baseline.py
│  ├─ stablesig_score_robust.py
│  ├─ stablesig_analyze.py
│  ├─ treering_score_baseline.py
│  ├─ treering_score_robust.py
│  ├─ treering_analyze.py
│  └─ make_ufd_jobs.py
├─ Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks.pdf
└─ README.md
```

---

## How to run the experiments

All detectors follow the same high-level workflow:

1. **Baseline scoring** (clean reals vs. clean fakes)
2. **Threshold calibration** on baseline only (fixed threshold is then locked)
3. **Robustness scoring** (inpainted splits)
4. **Evaluation** (AUC/ΔAUC + ASRReal/ASRFake under the locked threshold)

### Configuration (paths)

Each detector reads dataset locations from its YAML file in `configs/`.
For example, UFD defaults to `configs/paths_ufd.yaml` and uses keys like:

- `ufd.baseline.reals`
- `ufd.baseline.fakes`
- (robustness keys are detector-specific and referenced by the `*_score_robust.py` scripts)

The YAML loader expands environment variables (e.g., `$DATA_ROOT`) via `scripts/config_utils.py`.

### Quickstart

Run scripts directly (examples):

```bash
# UFD
python scripts/ufd_score_baseline.py --paths configs/paths_ufd.yaml
python scripts/ufd_score_robust.py --paths configs/paths_ufd.yaml
python scripts/ufd_analyze.py

# WaRPAD
python scripts/warpad_score_baseline.py --paths configs/paths_warpad.yaml
python scripts/warpad_score_robust.py --paths configs/paths_warpad.yaml
python scripts/warpad_analyze.py
```

For detector-specific options (checkpoints, third-party repos, device, output folders), use:

```bash
python scripts/ufd_score_baseline.py --help
```

### Outputs

Pipelines generally produce:

- Raw per-image score CSVs (per split/condition)
- Summary tables with **AUC**, **ΔAUC**, and attack success rates (**ASRReal**, **ASRFake**)

Default output root is typically `outputs/<detector>/...` (see each script’s `--out_dir`).

---

## Datasets

All curated datasets are hosted on Hugging Face:

https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks

- Baseline sets: clean reals vs. clean fakes (used for calibration)
- Robustness sets: attacked (inpainted) variants
- Attack splits include:
  - Semantic masks (from Semi-Truths)
  - Random blobs (area bins: 0–3%, 3–10%, 10–25%, 25–40%)
  - Random rectangles

---

## Detectors evaluated

Implemented using official codebases and weights (or official verification procedures):

- Passive detectors: UFD, DIMD
- Training-free methods: AEROBLADE, WaRPAD
- Watermarking approaches: Stable Signature, Tree-Ring

---

## Metrics

- **AUC**: separability of real vs. fake.
- **ΔAUC**: degradation from baseline to robustness condition.
- **ASR (Attack Success Rate)**:
  - **ASRFake** (fake evasion rate): % of inpainted fakes misclassified as Real.
  - **ASRReal** (real pass rate): % of inpainted reals still classified as Real.

ASR is computed on the baseline-correct subset (only images correctly classified on clean baseline are counted), using the threshold fixed during calibration.

---

## Reproducibility

- Datasets: Hugging Face link above
- Code: `scripts/`
- Paths/config: `configs/`
- Thesis PDF: linked at the top

All robustness results use a fixed threshold chosen on clean baseline data (no re-tuning on attacked images).

---

## Citation

```bibtex
@thesis{akin2025robustness,
  title  = {Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks},
  author = {Ak{\i}n, O{\u{g}}uz},
  year   = {2025},
  school = {Saarland University, CISPA Helmholtz Center for Information Security}
}
```