import argparse
import torch

from src.formats.int_fmt import FormatINT8
from src.formats.fp_fmt import FormatFP8_E4M3
from src.formats.nf4_fmt import FormatNF4
from src.formats.posit_fmt import FormatPosit8

from src.utils.metrics import calculate_mse
from src.utils.seed import fix_seed


def run_random_experiment(seed: int):
  """Runs a simulated neural network weight quantization benchmark."""
  print("========================================")
  print(" PHASE 0: Random Distribution Showdown")
  print("========================================\n")
  
  # Lock the RNG to ensure deterministic results across benchmark runs.
  fix_seed(seed)
  n_elements = 1_000_000
  
  # Simulate a layer of neural network weights following a Gaussian distribution.
  weights = torch.randn(n_elements) * 0.5
  
  print(f"Active Seed: {seed}")
  print(f"Generated {n_elements:,} random weights.")
  print(f"Distribution: Mean = {weights.mean():.4f}, Std = {weights.std():.4f}")
  print(f"Min = {weights.min():.4f}, Max = {weights.max():.4f}\n")
  
  formats = {
    "INT8 (Uniform)": FormatINT8(),
    "FP8 (E4M3)": FormatFP8_E4M3(),
    "Posit8 (es = 0)": FormatPosit8(es = 0),
    "NF4 (4-bit Codebook)": FormatNF4()
  }
  
  results = {}
  
  for name, fmt in formats.items():
    # Pass the weights through the fake quantization cycle.
    quantized = fmt.fake_quantize(weights)
    
    # Evaluate the quantization loss.
    mse = calculate_mse(weights, quantized)
    
    results[name] = mse
      
  print("Quantization Error (MSE) - Lower is better:")
  print("-" * 45)
  
  # Sort results by MSE to determine the most accurate format.
  sorted_results = sorted(results.items(), key = lambda item: item[1])
  
  for rank, (name, mse) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:<20} | MSE: {mse:.6f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Run the random distribution quantization benchmark.")
  parser.add_argument("--seed", type = int, default = 42, help = "The random seed for data generation.")
  args = parser.parse_args()
  
  run_random_experiment(args.seed)