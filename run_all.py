import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

# Add the project root to python path so we can import experiments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import CifarCNN
from src.formats.int_fmt import FormatINT8, FormatINT4
from src.formats.fp_fmt import FormatFP8_E4M3, FormatFP4_E2M1
from src.formats.nf4_fmt import FormatNF4
from src.formats.posit_fmt import FormatPosit8

from src.utils.metrics import get_correct_predictions, mcnemar_test, calculate_accuracy
import importlib
train_module = importlib.import_module("experiments.01_train_fp32")
train_baseline_model = train_module.train_baseline_model

benchmark_module = importlib.import_module("experiments.02_benchmark")
quantize_weights_in_place = benchmark_module.quantize_weights_in_place

def main():
  seeds = [42, 100, 2023, 777, 999]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Running Faza R1 on device: {device}")
  
  # Train all models if they don't exist
  for seed in seeds:
    model_path = f"results/models/model_seed_{seed}.pth"
    if not os.path.exists(model_path):
      print(f"Model for seed {seed} not found. Training...")
      train_baseline_model(seed)
      
  # Load dataset
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)
  
  # Define formats
  formats = {
    "FP32 Baseline": None, # Special case
    "INT8": (FormatINT8(), 8),
    "FP8 (E4M3)": (FormatFP8_E4M3(), 8),
    "Posit8": (FormatPosit8(es = 0), 8),
    "NF4": (FormatNF4(), 4),
    "INT4": (FormatINT4(), 4),
    "FP4 (E2M1)": (FormatFP4_E2M1(), 4)
  }
  
  # To store accuracies
  accuracies = defaultdict(list)
  
  # To store predictions for McNemar
  preds_for_mcnemar = {}

  # Benchmark all models
  for seed in seeds:
    print(f"\n--- Evaluating Seed {seed} ---")
    base_model = CifarCNN().to(device)
    base_model.load_state_dict(torch.load(f"results/models/model_seed_{seed}.pth", map_location=device, weights_only=True))
    base_model.eval()
    
    # Get FP32 predictions
    preds_base = get_correct_predictions(base_model, test_loader, device)
    acc_base = 100.0 * preds_base.sum().item() / len(preds_base)
    accuracies["FP32 Baseline"].append(acc_base)
    
    if seed == seeds[0]:
      preds_for_mcnemar["FP32 Baseline"] = preds_base
    
    for name, fmt_info in formats.items():
      if name == "FP32 Baseline":
        continue
        
      fmt, bits = fmt_info
      q_model = quantize_weights_in_place(base_model, fmt)
      
      preds = get_correct_predictions(q_model, test_loader, device)
      acc = 100.0 * preds.sum().item() / len(preds)
      accuracies[name].append(acc)
      
      if seed == seeds[0]:
        preds_for_mcnemar[name] = preds

  # Print Results (Mean ± Std Dev)
  print("\n==================================================")
  print(" Faza R1: Statistical Results (5 Seeds)")
  print("==================================================")
  print(f"{'Format':<15} | {'Accuracy (Mean ± Std)':<25}")
  print("-" * 45)
  
  # Sort by mean accuracy
  sorted_formats = sorted(accuracies.keys(), key=lambda k: np.mean(accuracies[k]), reverse=True)
  
  for name in sorted_formats:
    mean_acc = np.mean(accuracies[name])
    std_acc = np.std(accuracies[name])
    print(f"{name:<15} | {mean_acc:>7.2f}% ± {std_acc:.2f}%")

  # McNemar's Test on Seed 1
  print("\n==================================================")
  print(f" McNemar's Test (Seed {seeds[0]})")
  print("==================================================")
  # Compare each format against INT8 as a baseline
  base_format = "INT8"
  if base_format in preds_for_mcnemar:
    preds_int8 = preds_for_mcnemar[base_format]
    for name in formats.keys():
      if name not in [base_format, "FP32 Baseline"]:
        stat, p = mcnemar_test(preds_int8, preds_for_mcnemar[name])
        sig = "Significant" if p < 0.05 else "Not Significant"
        print(f"{base_format} vs {name:<12} | p-value: {p:.4f} ({sig})")

if __name__ == "__main__":
  main()
