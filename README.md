# Zero-Shot Single-Reference Face Identity Transfer

This repository package contains the final project report and source code for:

**Zero-Shot Single-Reference Face Identity Transfer via Reference Alignment and Region-Aware Candidate Selection**

The project performs single-reference face identity transfer without training LoRA, DreamBooth, or identity-specific model weights. It uses a frozen SDXL inpainting pipeline, ControlNet Canny conditioning, IP-Adapter face image prompting, InsightFace face analysis, reference alignment, multi-candidate generation, region-aware scoring, and adaptive candidate search.

## Repository Structure

```text
Zero-Shot-Single-Reference Face Identity Transfer/
  README.md
  requirements.txt
  dataset_manifest_template.csv
  final_zero_shot_transfer_and_ablation.ipynb
   
  report/
    figures/
      qualitative_triplet.png
```

## Main Files

- `final_zero_shot_transfer_and_ablation.ipynb`: The core main program of this project. It integrates a single-pair zero-shot face transfer demonstration (Single-Pair Demo) and fully automated dataset ablation experiments.
- `dataset_manifest_template.csv`: example manifest format for dataset-level evaluation.

## Installation

The notebook is designed for Kaggle with GPU T4 x2. In Kaggle:

1. Create a new notebook.
2. Set **Accelerator** to **GPU T4 x2**.
3. Upload or attach a dataset containing source/reference face pairs.
4. Upload the notebook from `zero-shot-paper-dataset-ablation.ipynb`.
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

## Kaggle T4 x2 Face Swap Notebook Execution Guide

This guide will help you smoothly run this Face Swap Notebook in the Kaggle environment. This notebook has been specifically optimized for Kaggle's **GPU T4 x2** setup, fixing path and hardware scheduling configurations.

### I. Prerequisites

Before running any code, ensure your Kaggle environment is configured correctly:

*   **Hardware Accelerator Settings:** Go to the Settings panel on the right side of your Kaggle Notebook and set the **Accelerator** to **GPU T4 x2**.
*   **Prepare Image Dataset:** Add the images you want to use for the face swap as a **Kaggle Dataset**.
*   **Hugging Face Token (Optional):** If you need to download models that require authorization, create a secret named `HF_TOKEN` in Kaggle Secrets and enter your Hugging Face token. If this secret is not set, public models can still be downloaded normally.

---

### II. Code Execution Steps and Configuration

#### Step 1: Install Dependencies (Cell 1)
Run the first code block directly. The system will use `!pip install` to automatically install required libraries, including `diffusers`, `transformers`, `insightface`, `controlnet_aux`, and others.

#### Step 2: Load Modules and Log in to Hugging Face (Cell 2)
Execute the second block. The code sets the PyTorch memory option (`expandable_segments:True`) to prevent out-of-memory errors and attempts to read the Kaggle Secret `HF_TOKEN` to log in to Hugging Face.

#### Step 3: Core Parameter Settings (Cell 3)
The third block contains all the parameter configurations for image generation and face swapping. Please modify these according to your needs:

*   **Image Path Settings:** You can fill in the paths of your mounted images in `SOURCE_IMAGE` and `REFERENCE_IMAGE`.
*   **Auto-Detection:** If left blank `""`, the program will automatically pick the first two images under the `/kaggle/input` directory as the source and reference images.
*   **Output Directory:** The generated images will be saved to `/kaggle/working` by default.
*   **Other Advanced Settings:** Settings such as face detection size, adaptive search (`ADAPTIVE_SEARCH_ENABLED`), automatic scoring mechanism (`AUTO_SCORE_ENABLED`), and eye refinement profiles can also be adjusted within this block.

#### Step 4: Verify Runtime Environment (Cell 4)
Execute `print_runtime_info()`. This will print the current PyTorch version and the number of GPUs detected by the system, allowing you to confirm that the dual T4 GPUs are correctly enabled.

#### Step 5: Load Functions and Preview Input Images (Cell 5 & 6)
*   The system assigns the InsightFace face recognition Provider to `CPUExecutionProvider` by default to save valuable GPU memory resources.
*   The program will resolve the image paths you set or automatically found, and output small preview images for you to verify that the source image (`SOURCE_IMAGE`) and reference image (`REFERENCE_IMAGE`) are loaded correctly.

#### Step 6: Start Inference
Continue executing the remaining Cells in order. The Notebook will perform the face swap inference and output the final results to the output directory.

---

### III. Important Notes

*   **Dual-GPU Device Map Issues:** In the parameter block, `USE_DUAL_GPU_DEVICE_MAP = False` is set by default. When using SDXL ControlNet and IP-Adapter, enabling the dual-GPU device map sometimes causes Kaggle to allocate some layers to the CPU, triggering an "Expected all tensors to be on the same device" error. It is recommended to keep it as False for stable execution.

### Dataset Ablations
The runner writes:

- `dataset_ablation_results.csv`
- `dataset_ablation_results.json`
- `ablation_summary.csv`
- `ablation_summary.json`
- `paired_ablation_comparisons.csv`
- `paired_ablation_comparisons.json`
- per-run final images, selected crops, debug crops, candidate contact sheets, and score JSON files.

## Ethical Use

This project is for research and educational use. Do not use it to create deceptive, impersonating, or non-consensual media. Generated images should be clearly labeled as synthetic.
