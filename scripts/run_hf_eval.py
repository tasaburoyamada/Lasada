import sys
import os
import torch
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# Add scripts dir to path to import hf registration
sys.path.append(os.path.abspath("scripts"))
import register_hf_model

def run_huggingface_eval(model_path, tasks=["hellaswag", "arc_easy"]):
    print("==================================================")
    print(f" Running HuggingFace Benchmark Evaluation ")
    print(f" Model Path: {model_path}")
    print(f" Target Tasks: {tasks}")
    print("==================================================")

    tokenizer = register_hf_model.Symbol32HFTokenizer()
    lm = HFLM(pretrained=model_path, tokenizer=tokenizer, device="cpu", batch_size=1)
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=10
    )

    print("\n==================================================")
    print(" HuggingFace Benchmark Results:")
    print("==================================================")
    for task_name, task_results in results["results"].items():
        acc = task_results.get("acc,none", task_results.get("acc_norm,none", "N/A"))
        print(f"  - [{task_name}]: Accuracy = {acc}")

if __name__ == "__main__":
    model_dir = "/home/tasaburoyamada/models/Lasada-BitMoE-31B-Base"
    run_huggingface_eval(model_dir)
