# Robustness-of-AI-Generated-Image-Detection-Against-Localized-Inpainting-Attacks
This repository contains the code, data, and written work for my Bachelor's thesis. The goal of this project is to systematically evaluate the robustness of different families of AI-generated image detectors against adversarial inpainting attacks.

---

## Repository Structure

*   `/exposé/`: Contains the LaTeX source code and final PDF of the thesis exposé.
*   `/preliminary_experiment/`: Contains the self-contained code and results for a preliminary experiment run in Google Colab to validate the methodology.
*   `/final_experiments/`: (Work in Progress) Will contain the scripts and results for the full-scale experiments run on the main compute cluster.

---

## Preliminary Results (August 17th, 2025)

A preliminary experiment was conducted to validate the methodology on a 400-image benchmark (200 real, 200 inpainted). The `UniversalFakeDetect` (CLIP) model was evaluated.

*   **Preliminary Accuracy:** **97.00%**
*   **For full details and reproducible code, please see the [`/preliminary_experiment/`](./preliminary_experiment) directory.**
