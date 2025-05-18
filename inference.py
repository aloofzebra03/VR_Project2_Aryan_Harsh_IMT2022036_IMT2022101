import argparse
import os
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login

def main():
    login(token = "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")
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',  type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path',   type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path, dtype=str)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the base PaliGemma model in FP16
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-mix-224",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    lora_dir = "aloofzebra03/VR2_finetuned-paligemma-lora"

    # 2) Plug in your fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # 3) Reload the processor so it matches your base model’s preproc
    processor = AutoProcessor.from_pretrained(
        "google/paligemma-3b-mix-224",
        local_files_only=False
    )

    generated_answers = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PaliGemma Inference"):
        img_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row["question"]).strip()
        prompt_text = f"<image> Answer in exactly one word: {question}"

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

            # decode only the answer portion
            answer_ids = output_ids[0, input_len:]
            pred = processor.decode(answer_ids, skip_special_tokens=True).strip().split()[0].lower()
        except Exception:
            pred = "error"

        generated_answers.append(pred)

    # Append to DataFrame and save
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
