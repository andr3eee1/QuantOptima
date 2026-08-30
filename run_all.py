from src.quantization.apply import apply_optimal_codebooks
from src.utils.metrics import get_correct_predictions, mcnemar_test, calculate_accuracy, calculate_mse
from src.formats.posit_fmt import FormatPosit8
from src.formats.nf4_fmt import FormatNF4
from src.formats.fp_fmt import FormatFP8_E4M3, FormatFP4_E2M1
from src.formats.int_fmt import FormatINT8, FormatINT4
from src.model.network import CifarCNN
import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import importlib

# Add the project root to python path so we can import experiments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


kurtosis_module = importlib.import_module("experiments.04_kurtosis")
run_kurtosis_experiment = kurtosis_module.run_kurtosis_experiment

activations_module = importlib.import_module("experiments.05_activations")
run_activations_analysis = activations_module.run_activations_analysis

train_module = importlib.import_module("experiments.01_train_fp32")
train_baseline_model = train_module.train_baseline_model

benchmark_module = importlib.import_module("experiments.02_benchmark")
quantize_weights_in_place = benchmark_module.quantize_weights_in_place


def run_faza_r1(seeds):
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
  test_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform)
  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=64, shuffle=False)

  # Define formats
  formats = {
      "FP32 Baseline": None,
      "INT8": (FormatINT8(), 8),
      "FP8 (E4M3)": (FormatFP8_E4M3(), 8),
      "Posit8": (FormatPosit8(es=0), 8),
      "NF4": (FormatNF4(), 4),
      "INT4": (FormatINT4(), 4),
      "FP4 (E2M1)": (FormatFP4_E2M1(), 4),
      "OPTIM-8": 8,
      "OPTIM-4": 4
  }

  accuracies = defaultdict(list)
  mses = defaultdict(list)
  preds_for_mcnemar = {}

  for seed in seeds:
    print(f"\n--- Evaluating Seed {seed} ---")
    base_model = CifarCNN().to(device)
    base_model.load_state_dict(torch.load(
        f"results/models/model_seed_{seed}.pth", map_location=device, weights_only=True))
    base_model.eval()

    preds_base = get_correct_predictions(base_model, test_loader, device)
    acc_base = 100.0 * preds_base.sum().item() / len(preds_base)
    accuracies["FP32 Baseline"].append(acc_base)
    mses["FP32 Baseline"].append(0.0)

    if seed == seeds[0]:
      preds_for_mcnemar["FP32 Baseline"] = preds_base

    for name, fmt_info in formats.items():
      if name == "FP32 Baseline":
        continue

      if name.startswith("OPTIM"):
        q_model = apply_optimal_codebooks(base_model, bits=fmt_info)
      else:
        fmt, bits = fmt_info
        q_model = quantize_weights_in_place(base_model, fmt)

      mse = 0.0
      for (p_name, p_base), (q_name, p_q) in zip(base_model.named_parameters(), q_model.named_parameters()):
        if 'weight' in p_name:
          mse += calculate_mse(p_base, p_q)
      mses[name].append(mse)

      preds = get_correct_predictions(q_model, test_loader, device)
      acc = 100.0 * preds.sum().item() / len(preds)
      accuracies[name].append(acc)

      if seed == seeds[0]:
        preds_for_mcnemar[name] = preds

  print("\n==================================================")
  print(" Faza R1: Statistical Results (5 Seeds)")
  print("==================================================")
  print(f"{'Format':<15} | {'Mean MSE':<10} | {'Accuracy (Mean ± Std)':<25}")
  print("-" * 55)

  sorted_formats = sorted(
      accuracies.keys(), key=lambda k: np.mean(accuracies[k]), reverse=True)

  for name in sorted_formats:
    mean_acc = np.mean(accuracies[name])
    std_acc = np.std(accuracies[name])
    mean_mse = np.mean(mses[name])
    print(f"{name:<15} | {mean_mse:<10.4f} | {mean_acc:>7.2f}% ± {std_acc:.2f}%")

  print("\n==================================================")
  print(f" McNemar's Test (Seed {seeds[0]})")
  print("==================================================")
  base_format = "INT8"
  if base_format in preds_for_mcnemar:
    preds_int8 = preds_for_mcnemar[base_format]
    for name in formats.keys():
      if name not in [base_format, "FP32 Baseline"]:
        stat, p = mcnemar_test(preds_int8, preds_for_mcnemar[name])
        sig = "Significant" if p < 0.05 else "Not Significant"
        print(f"{base_format} vs {name:<12} | p-value: {p:.4f} ({sig})")

  print("\n==================================================")
  print(" Generating the Star Graph (Accuracy vs Bits)")
  print("==================================================")

  plt.figure(figsize=(10, 6))

  plt.errorbar(
      [32, 8, 4],
      [np.mean(accuracies["FP32 Baseline"]), np.mean(
          accuracies["INT8"]), np.mean(accuracies["INT4"])],
      yerr=[np.std(accuracies["FP32 Baseline"]), np.std(
          accuracies["INT8"]), np.std(accuracies["INT4"])],
      marker='o', label="Standard (INT)", color='blue', linewidth=2, capsize=5
  )

  plt.errorbar(
      [32, 8, 4],
      [np.mean(accuracies["FP32 Baseline"]), np.mean(
          accuracies["OPTIM-8"]), np.mean(accuracies["OPTIM-4"])],
      yerr=[np.std(accuracies["FP32 Baseline"]), np.std(
          accuracies["OPTIM-8"]), np.std(accuracies["OPTIM-4"])],
      marker='x', linestyle='--', label="OPTIM (Lloyd-Max)", color='red', linewidth=2, capsize=5
  )

  plt.errorbar(
      [32, 8, 4],
      [np.mean(accuracies["FP32 Baseline"]), np.mean(
          accuracies["FP8 (E4M3)"]), np.mean(accuracies["FP4 (E2M1)"])],
      yerr=[np.std(accuracies["FP32 Baseline"]), np.std(
          accuracies["FP8 (E4M3)"]), np.std(accuracies["FP4 (E2M1)"])],
      marker='s', label="FP Formats", color='green', linewidth=2, capsize=5
  )

  plt.title("The Star Graph: Model Accuracy vs. Bit-Width (5 Seeds)")
  plt.xlabel("Bits per Weight")
  plt.ylabel("Test Accuracy (%)")

  plt.xlim(35, 2)

  plt.legend()
  plt.grid(True)

  os.makedirs("results/plots", exist_ok=True)
  plt.savefig("results/plots/star_graph_accuracy.png")
  print("Saved to results/plots/star_graph_accuracy.png!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seeds", type=int, nargs="+",
                      default=[42, 100, 2023, 777, 999])
  args = parser.parse_args()

  # Run the accuracy benchmarks and generate Star Graph
  run_faza_r1(args.seeds)

  # Run the kurtosis sweep and generate the Kurtosis Crossover Graph
  run_kurtosis_experiment(args.seeds[0])
  
  # Run the activations bonus
  run_activations_analysis(args.seeds[0])
