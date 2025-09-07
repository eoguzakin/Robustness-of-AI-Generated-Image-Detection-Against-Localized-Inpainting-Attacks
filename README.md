# Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks

This repository contains the **code and experimental pipelines** used in the thesis:

> **Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks**
> Oğuz Akın, Saarland University / CISPA Helmholtz Center for Information Security (2025)

📄 [Thesis PDF](./Thesis.pdf)
📊 [Curated Datasets on Hugging Face](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)

---

## 📌 Overview

This thesis systematically evaluates the **robustness of AI-generated image (AIGI) detectors** against **localized inpainting attacks**.

Most detectors perform well on clean benchmark datasets, but real-world deployment involves post-processing edits such as inpainting, cropping, or compression. Inpainting, in particular, can erase watermarks, sanitize generative fingerprints, or create false positives on authentic photos.

### Key Findings

* **Reactive detectors** exhibit a *security–usability trade-off*:

  * **DIMD**: strong against evasion (low ASR on fakes), but fails on edited reals (high false positives).
  * **AEROBLADE**: tolerant on reals, but catastrophically vulnerable to evasion (ASR often 100%).
  * **UFD**: excellent baseline AUC but collapses under inpainting (ASR >95% in many cases).
* **Training-free methods** (DIRE, AEROBLADE) show **unpredictable behavior** and high false positives.
* **Watermarking methods** (Stable Signature, Tree-Ring) are **threshold fragile**: robust at conservative thresholds (t90/t99) but collapse at balanced thresholds.
* **Baseline AUC is a poor predictor of robustness** — detectors that excel on clean data can fail under realistic manipulations.

📢 **Conclusion**: No universally robust detector exists. Future research must adopt *threat-model-specific* and *threshold-aware* evaluations.

---

## 🗂 Repository Structure

```
├── scripts/                          # Experimental pipelines and evaluation code
│   ├── preparation of datasets and detectors.zip
│   ├── ufd_scripts.zip
│   ├── dimd_thesis_scripts_full_bundle.zip
│   ├── dire_scripts_bundle.zip
│   ├── aeroblade_scripts_py_full.zip
│   ├── stablesig_tools_bundle.zip
│   ├── treering_scripts_bundle_full.zip
│   └── watermark_inpaint_pipeline.zip
├── Thesis.pdf                        # Final thesis document
└── README.md                         # You are here
```

### Script Bundles

* **preparation of datasets and detectors.zip** – dataset curation, preprocessing, environment setup
* **ufd\_scripts.zip** – UniversalFakeDetect (semantic passive detection)
* **dimd\_thesis\_scripts\_full\_bundle.zip** – DIMD (artifact-based passive detection)
* **dire\_scripts\_bundle.zip** – DIRE (diffusion reconstruction, training-free)
* **aeroblade\_scripts\_py\_full.zip** – AEROBLADE (autoencoder reconstruction, training-free)
* **stablesig\_tools\_bundle.zip** – Stable Signature watermark generation and detection
* **treering\_scripts\_bundle\_full.zip** – Tree-Ring watermark detection pipeline
* **watermark\_inpaint\_pipeline.zip** – standardized inpainting attacks (LaMa, ZITS, mask generation)

Each bundle contains the **standalone code and configs** to replicate the respective experiments.

---

## 📊 Datasets

All curated datasets are hosted on Hugging Face:

👉 [Robustness of AIGI Detection Against Localized Inpainting Attacks](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)

* **Baseline sets**: reals vs. fully synthetic fakes (for calibration)
* **Robustness sets**: matched pairs of reals and their inpainted versions
* **Attack splits**:

  * Semantic masks (from Semi-Truths dataset)
  * Random blobs (area bins: 0–3%, 3–10%, 10–25%, 25–40%)
  * Random rectangles

---

## ⚙️ Detectors Evaluated

The following detectors were implemented using official codebases and weights:

* **Passive detectors**:

  * *UFD* (semantic, CLIP-based)
  * *DIMD* (artifact-based CNN)
* **Training-free methods**:

  * *DIRE* (diffusion reconstruction error)
  * *AEROBLADE* (autoencoder reconstruction error)
* **Watermarking approaches**:

  * *Stable Signature* (latent-space watermark)
  * *Tree-Ring* (frequency-domain watermark)

Each detector was evaluated in its own isolated environment.

**Evaluation protocol**:

1. **Baseline calibration** – reals vs. fakes → determine decision threshold \*t\*\*\*.
2. **Robustness test** – evaluate on inpainted sets at fixed \*t\*\*\*.

---

## 📈 Metrics

* **AUC** – separability of real vs. fake.
* **ΔAUC** – degradation from baseline to robustness.
* **ASR (Attack Success Rate)**:

  * *Fake evasion*: % of inpainted fakes misclassified as Real (detector fooled).
  * *Real misclassification*: % of inpainted reals incorrectly passed as clean Real (detector fails to flag edits).
* **Watermarking ASR** – % of attacked images where watermark is undetectable, measured at fixed TPRs (t90, t99) and baseline thresholds.

---

## 🔬 Reproducibility

* Datasets: [Hugging Face](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)
* Scripts: in this repository (`scripts/` bundles)
* Thesis: included as [Thesis.pdf](./Thesis.pdf)

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@thesis{akin2025robustness,
  title={Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks},
  author={Ak{\i}n, Oğuz},
  year={2025},
  school={Saarland University, CISPA Helmholtz Center for Information Security}
}
```

---