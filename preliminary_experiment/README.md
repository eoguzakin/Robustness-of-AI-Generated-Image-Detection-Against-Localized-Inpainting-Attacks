# Preliminary Experiment: UniversalFakeDetect on a 200-Image Benchmark

This directory contains the code and results for a preliminary experiment conducted on August 17th, 2025. The goal was to validate the core methodology of the thesis on a small-scale benchmark while awaiting access to the main compute cluster.

---

### Contents

*   `preliminary_experiment_200_images.ipynb`: A Google Colab notebook that contains the complete, self-contained code to reproduce this experiment.
*   `preliminary_results.txt`: A text file containing the final accuracy score from the experiment run.

---

### How to Reproduce

To ensure full reproducibility, the experiment should be run in a clean Google Colab environment.

1.  **Open the Notebook:** Upload the `preliminary_experiment_200_images.ipynb` file to Google Colab.
2.  **Set the Runtime:** In the menu, go to `Runtime` -> `Change runtime type` and select `T4 GPU`.
3.  **Run the Script:** Run the single, large code cell in the notebook.

The script will automatically perform all necessary steps:
- Install all dependencies in a clean environment.
- Download 200 real images from the "beans" dataset.
- Generate 200 corresponding "fake" images by applying a Stable Diffusion inpainting attack.
- Evaluate the `UniversalFakeDetect` (CLIP) model on this 400-image benchmark.
- Print the final accuracy score.

---

### Preliminary Results

The experiment was run on a benchmark of 200 real and 200 inpainted images.

**Final Accuracy:** The `UniversalFakeDetect` model achieved a preliminary accuracy of **97.00%**.
