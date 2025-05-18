# Multimodal VQA with Amazon Berkeley Objects

**Course**: AIM825 – VR Mini Project Two  

A compact Visual Question Answering pipeline built on the **Amazon Berkeley Objects (ABO)** dataset.  
This project covers the full lifecycle from raw metadata curation to baseline evaluation, parameter-efficient fine-tuning (LoRA), and inference deployment.

---

## Features

- **Data Preparation & Curation**  
  - Parse ABO metadata  
  - Generate single-word QA pairs  
  - Clean and validate answer distributions  
- **Baseline Evaluation**  
  - Off-the-shelf VQA models (PaLI-Gemma, Qwen2-VL, etc.)  
  - Automated inference scripts & performance logging  
- **LoRA Fine-Tuning**  
  - Parameter-efficient adaptation on VQA tasks  
  - Custom learning rate schedules & early-stopping  
- **Inference & Deployment**  
  - Compare baseline vs fine-tuned model outputs  
  - Upload final weights to Hugging Face Hub  
  - Ready-to-use Jupyter demos

---

## Repository Layout

```text
.
├── Data_Prep_And_Curation_Notebooks/   # Metadata parsing & QA CSV generation
├── Datasets/                           # Curated Q&A Dataset for finetuning
├── Finetuning_Notebooks/               # LoRA fine-tuning experiments
├── Inference_Notebooks/                # Model inference demos
│   ├── Baseline/                       # Off-the-shelf model benchmarks
│   └── Finetuned/                      # LoRA-tuned weights inference
├── uploading-model-to-hf.ipynb         # Script to push weights to HF
├── Report.pdf                          # Detailed project report & results
├── inference.py                        # Inference Script for final Evaluation
├── requirements.txt                    # Python dependencies
└── README.md                           # This overview
