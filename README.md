# Zero-Shot Single-Reference Face Identity Transfer

This repository package contains the final project report and source code for:

**Zero-Shot Single-Reference Face Identity Transfer via Reference Alignment and Region-Aware Candidate Selection**

The project performs single-reference face identity transfer without training LoRA, DreamBooth, or identity-specific model weights. It uses a frozen SDXL inpainting pipeline, ControlNet Canny conditioning, IP-Adapter face image prompting, InsightFace face analysis, reference alignment, multi-candidate generation, region-aware scoring, and adaptive candidate search.

## Repository Structure

```text
final_submission/
  README.md
  requirements.txt
  dataset_manifest_template.csv
  submission_steps_zh.md
  code/
    zero-shot-paper-dataset-ablation.ipynb
    zero-shot-experiment-demo-with-pilot-output.ipynb
    upgrade_zero_shot_dataset_ablation.py
  report/
    main.tex
    references.bib
    figures/
      qualitative_triplet.png
```

## Main Files

- `code/zero-shot-paper-dataset-ablation.ipynb`: paper-ready notebook for dataset-level batch runs and ablations.
- `code/zero-shot-experiment-demo-with-pilot-output.ipynb`: executed single-pair demo notebook with pilot output logs.
- `code/upgrade_zero_shot_dataset_ablation.py`: helper script used to generate the paper-mode notebook from the original experiment notebook.
- `report/main.tex`: IEEE-style final report draft.
- `dataset_manifest_template.csv`: example manifest format for dataset-level evaluation.

## Installation

The notebook is designed for Kaggle with GPU T4 x2. In Kaggle:

1. Create a new notebook.
2. Set **Accelerator** to **GPU T4 x2**.
3. Upload or attach a dataset containing source/reference face pairs.
4. Upload the notebook from `code/zero-shot-paper-dataset-ablation.ipynb`.
5. Run the first install cell, or install the packages listed in `requirements.txt`.

The original install cell uses:

```bash
pip install -U \
  diffusers==0.35.1 \
  transformers==4.44.2 \
  accelerate==0.34.2 \
  insightface \
  onnxruntime \
  opencv-python-headless \
  peft \
  safetensors \
  controlnet_aux \
  huggingface_hub
```

Kaggle usually already provides a compatible PyTorch/CUDA runtime. If running locally, install a CUDA-enabled PyTorch build that matches your GPU driver before installing the remaining packages.

## Running the Single-Pair Demo

Use `zero-shot-experiment-demo-with-pilot-output.ipynb` if you want to inspect the original workflow and pilot output logs.

1. Set `SOURCE_IMAGE` and `REFERENCE_IMAGE`, or leave them blank to auto-detect two images under `/kaggle/input`.
2. Run the notebook from top to bottom.
3. Inspect `debug_reference_aligned_crop.png`.
4. Inspect `candidate_contact_sheet.png`.
5. Check `auto_refine_leaderboard.csv` and `candidate_scores.json`.
6. Use `MANUAL_SELECTED_CANDIDATE` only if visual judgment disagrees with the automatic ranking.

## Running Dataset-Level Ablations

Use `zero-shot-paper-dataset-ablation.ipynb`.

1. Prepare a CSV manifest following `dataset_manifest_template.csv`.
2. Set `DATASET_MANIFEST` to the manifest path, or fill `DATASET_PAIRS` directly in the notebook.
3. Choose the ablation variants in `ABLATION_GRID`.
4. Set `RUN_DATASET_BATCH = True`.
5. Run the dataset batch cells.

The runner writes:

- `dataset_ablation_results.csv`
- `dataset_ablation_results.json`
- `ablation_summary.csv`
- `ablation_summary.json`
- `paired_ablation_comparisons.csv`
- `paired_ablation_comparisons.json`
- per-run final images, selected crops, debug crops, candidate contact sheets, and score JSON files.

## Dataset Information

The expected manifest columns are:

- `pair_id`: stable identifier for the source/reference pair.
- `source_image`: path to the source/body/pose image.
- `reference_image`: path to the reference identity image.
- optional metadata such as `split`, `source_id`, `reference_id`, and `notes`.

Use only consented images. If images are private, do not publish them in the GitHub repository. Instead, publish the manifest format, reproduction instructions, and aggregate result tables.

## Report

The report is in `report/main.tex` and uses IEEE conference style. On Overleaf:

1. Create an IEEE conference project.
2. Upload `main.tex`, `references.bib`, and the `figures/` folder.
3. Compile with pdfLaTeX.
4. Replace the placeholder GitHub URL in the Data Availability Statement with the real repository link.
5. Fill the ablation results table after running the dataset batch.

## Ethical Use

This project is for research and educational use. Do not use it to create deceptive, impersonating, or non-consensual media. Generated images should be clearly labeled as synthetic.
