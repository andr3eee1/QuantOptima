import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from src.formats.int_fmt import FormatINT8
from src.formats.fp_fmt import FormatFP8_E4M3
from src.utils.metrics import calculate_mse
from src.utils.seed import fix_seed
from src.model.network import CifarCNN


def calculate_kurtosis(tensor: torch.Tensor) -> float:
  """Calculates the excess kurtosis of a 1D tensor."""
  mean = torch.mean(tensor)
  std = torch.std(tensor)
  # The 4th moment over the standard deviation squared (Standard Kurtosis)
  kurt = torch.mean(((tensor - mean) / std) ** 4)
  return kurt.item()


def run_kurtosis_experiment(seed: int):
  print("========================================")
  print(" PHASE 3: The Kurtosis Crossover")
  print("========================================\n")

  fix_seed(seed)
  n_elements = 100_000

  fmt_int8 = FormatINT8()
  fmt_fp8 = FormatFP8_E4M3()

  df_values = np.logspace(np.log10(30), np.log10(2.1), num=20)

  kurtosis_vals = []
  mse_int8 = []
  mse_fp8 = []

  print("Sweeping Student-t degrees of freedom (fatness of tails)...")
  for df in df_values:
    m = torch.distributions.StudentT(df=df)
    tensor = m.sample([n_elements]).squeeze()

    kurt_val = calculate_kurtosis(tensor)

    q_int8 = fmt_int8.fake_quantize(tensor)
    q_fp8 = fmt_fp8.fake_quantize(tensor)

    err_int8 = calculate_mse(tensor, q_int8)
    err_fp8 = calculate_mse(tensor, q_fp8)

    kurtosis_vals.append(kurt_val)
    mse_int8.append(err_int8)
    mse_fp8.append(err_fp8)

    print(f"DF: {df:5.1f} | Kurtosis: {kurt_val:8.2f} | INT8 MSE: {err_int8:.6f} | FP8 MSE: {err_fp8:.6f}")

  # Sort by kurtosis for plotting
  sort_idx = np.argsort(kurtosis_vals)
  kurtosis_vals = np.array(kurtosis_vals)[sort_idx]
  mse_int8 = np.array(mse_int8)[sort_idx]
  mse_fp8 = np.array(mse_fp8)[sort_idx]

  # Load Real Weights to plot as Stars
  device = torch.device("cpu")
  model = CifarCNN()
  model.load_state_dict(torch.load(
      "results/models/model_seed_42.pth", map_location=device, weights_only=True))

  real_kurtosis = {}
  for name, param in model.named_parameters():
    if 'weight' in name:
      kurt = calculate_kurtosis(param.detach().view(-1))
      real_kurtosis[name] = kurt
      print(f"Real Layer: {name:<12} | Kurtosis: {kurt:.2f}")

  # Plotting
  os.makedirs("results/plots", exist_ok=True)
  plt.figure(figsize=(10, 6))

  plt.plot(kurtosis_vals, mse_int8, label='INT8 (Symmetric, scaled)',
           color='blue', linewidth=2)
  plt.plot(kurtosis_vals, mse_fp8, label='FP8 (E4M3, scaled)',
           color='orange', linewidth=2)

  # Add Stars for real weights
  y_star_baseline = 0.0  # Just plot them near the bottom axis or on the INT8 curve
  for name, kurt in real_kurtosis.items():
    # Interpolate the Y value on the INT8 curve for visual appeal
    y_val = np.interp(kurt, kurtosis_vals, mse_int8)
    plt.plot(kurt, y_val, marker='*', markersize=15, label=f'Real: {name}')

  plt.title("Phase 3: MSE vs Kurtosis (The True Crossover)")
  plt.xlabel("Kurtosis (Outlier Heaviness)")
  plt.ylabel("Mean Squared Error (MSE)")
  plt.yscale('log')
  plt.xscale('log')
  plt.grid(True, which="both", ls="--", alpha=0.5)
  plt.legend()

  save_path = "results/plots/kurtosis_crossover.png"
  plt.savefig(save_path)
  print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  run_kurtosis_experiment(args.seed)
