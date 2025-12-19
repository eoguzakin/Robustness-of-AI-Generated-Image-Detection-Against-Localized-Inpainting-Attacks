The README below follows your **old sectioning/format**, but updates the detector list + conclusions to match the **final thesis** (fixed-threshold protocol; ASRReal/ASRFake framing; dichotomy between edited reals vs edited fakes; watermark robustness degrading with larger masks) and reflects the **new scripts layout** (per-detector runnable pipelines / script files instead of zip bundles).[1][2][3][4][5][6]

```markdown
# Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks


This repository contains the **code and experimental pipelines** used in the thesis:


> **Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks**  
> Oğuz Akın, Saarland University / CISPA Helmholtz Center for Information Security (2025)


📄 [Thesis PDF](./Thesis.pdf)  
📊 [Curated Datasets on Hugging Face](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)


---


## 📌 Overview


This thesis systematically evaluates the **robustness of AI-generated image (AIGI) detectors** against **localized inpainting attacks**.

Most detectors perform well on clean benchmark datasets, but real-world deployment involves post-processing edits. Localized inpainting is particularly important because it **overwrites only selected regions**, creating *hybrid images* that mix authentic and generated pixels (or watermarked and non-watermarked regions).


### Key Findings


* **A consistent dichotomy appears across detector families**:
  * On **edited real photos**, many detectors **fail to flag localized inpainting** and still classify them as Real (high **ASRReal** / high “real pass rate”).
  * On **edited fakes**, many detectors remain **surprisingly robust**: localized inpainting often does **not** significantly help an attacker evade detection (low **ASRFake** / low “fake evasion rate”).
* **Passive + training-free detectors** (e.g., UFD, DIMD, WaRPAD, AEROBLADE) commonly show **high pass rates on inpainted reals**, meaning localized edits can go unnoticed under a fixed-threshold policy.
* **Watermarking methods** (Tree-Ring, Stable Signature) tend to stay **robust for small and moderate edits**, but their failure rates **increase more noticeably** as the inpainted area grows (especially for large masks such as 25–40%).
* **Baseline AUC is not a reliable predictor of robustness**: strong clean-data separation does not guarantee stability under localized edits.

📢 **Conclusion**: Robustness must be evaluated under a clear threat model and a fixed-threshold protocol; “clean benchmark performance” alone is insufficient for deployment decisions.


---


## 🗂 Repository Structure


```
├── scripts/                             # Experimental pipelines and evaluation code
│   ├── ufd-scripts.txt                  # UFD: baseline + robustness scoring + analysis templates
│   ├── dimd-scripts.txt                 # DIMD: scoring + bin evaluation (AUC/ASR) templates
│   ├── warpad-scripts.txt               # WaRPAD: scoring + analysis pipeline
│   ├── aeroblade-scripts.txt            # AEROBLADE (official): run + aggregate + print results
│   ├── stablesig-scripts.txt            # Stable Signature: decode-score + eval pipeline
│   └── treering-scripts.txt             # Tree-Ring: key estimation + scoring + eval pipeline
├── Thesis.pdf                            # Final thesis document
└── README.md                             # You are here
```


### Script Bundles (new layout)


Instead of large zipped bundles, the repository now ships **detector-specific runnable pipelines / templates** in `scripts/*.txt`:

* **ufd-scripts.txt** – UFD scoring on baseline + robustness splits + fixed-threshold evaluation
* **dimd-scripts.txt** – DIMD directory scorer + summary evaluator (AUC/ΔAUC/ASR)
* **warpad-scripts.txt** – WaRPAD scoring + analysis pipeline (fixed-threshold ASR tables)
* **aeroblade-scripts.txt** – AEROBLADE official pipeline + aggregation + final ASR tables
* **stablesig-scripts.txt** – Stable Signature decode scoring + robustness evaluator (with LaMa name mapping)
* **treering-scripts.txt** – Tree-Ring key estimation + scoring + robustness evaluator

Each file contains an end-to-end recipe: environment notes, dataset paths, scoring, and evaluation outputs.


---


## 🚀 How to run the experiments


All detectors follow the same high-level workflow:

1. **Baseline scoring** (clean reals vs clean fakes)  
2. **Threshold calibration** on baseline only (fixed threshold is then locked)  
3. **Robustness scoring** (inpainted splits)  
4. **Evaluation** (AUC/ΔAUC + ASRReal/ASRFake under the locked threshold)

### Quickstart

Pick a detector and follow its script file:

- UFD → `scripts/ufd-scripts.txt`
- DIMD → `scripts/dimd-scripts.txt`
- WaRPAD → `scripts/warpad-scripts.txt`
- AEROBLADE → `scripts/aeroblade-scripts.txt`
- Stable Signature → `scripts/stablesig-scripts.txt`
- Tree-Ring → `scripts/treering-scripts.txt`

### Outputs

All pipelines are designed to produce:

- Raw score CSVs (per condition/split)
- A summary table with **AUC**, **ΔAUC**, and attack success rates (**ASRReal**, **ASRFake**)
- (Optional) confidence intervals depending on the evaluator script


---


## 📊 Datasets


All curated datasets are hosted on Hugging Face:

👉 [Robustness of AIGI Detection Against Localized Inpainting Attacks](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)


* **Baseline sets**: clean reals vs. clean fakes (used for calibration)
* **Robustness sets**: attacked (inpainted) variants
* **Attack splits**:
  * Semantic masks (from Semi-Truths)
  * Random blobs (area bins: 0–3%, 3–10%, 10–25%, 25–40%)
  * Random rectangles


---


## ⚙️ Detectors Evaluated


The following detectors were implemented using official codebases and weights (or official verification procedures):

* **Passive detectors**:
  * *UFD* (CLIP-based universal detector)
  * *DIMD* (artifact-based diffusion detector)
* **Training-free methods**:
  * *AEROBLADE* (autoencoder reconstruction-based)
  * *WaRPAD* (feature-consistency based)
* **Watermarking approaches**:
  * *Stable Signature* (latent-space / decoder-based verification)
  * *Tree-Ring* (frequency-domain / inversion-based verification)

Each detector is evaluated under the same fixed-threshold protocol described below.


---


## 📈 Metrics


* **AUC** – separability of real vs. fake.
* **ΔAUC** – degradation from baseline to robustness condition.
* **ASR (Attack Success Rate)**:
  * **ASRFake (Fake evasion rate)**: % of **inpainted fakes** misclassified as Real (attacker successfully evades).
  * **ASRReal (Real pass rate)**: % of **inpainted reals** still classified as Real (detector fails to flag localized edits).

**Important:** ASR metrics are computed on the *baseline-correct subset* (only images correctly classified on clean baseline are counted in ASR evaluation), using the threshold fixed during calibration.


---


## 🔬 Reproducibility


* Datasets: [Hugging Face](https://huggingface.co/datasets/eoguzakin/Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks)
* Scripts: in this repository (`scripts/*.txt`)
* Thesis: included as [Thesis.pdf](./Thesis.pdf)

All robustness results are produced with a **fixed threshold chosen on clean baseline data** (no re-tuning on attacked images).


---


## 📜 Citation


If you use this work, please cite:


```
@thesis{akin2025robustness,
  title={Robustness of AI-Generated Image Detection Against Localized Inpainting Attacks},
  author={Ak{\i}n, O\u{g}uz},
  year={2025},
  school={Saarland University, CISPA Helmholtz Center for Information Security}
}
```
