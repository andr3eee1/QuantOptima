import argparse
import torch

from src.formats.int_fmt import FormatINT8
from src.formats.fp_fmt import FormatFP8_E4M3
from src.formats.nf4_fmt import FormatNF4
from src.formats.posit_fmt import FormatPosit8

from src.utils.metrics import calculate_mse
from src.utils.seed import fix_seed

def calculate_kurtosis(tensor: torch.Tensor) -> float:
  """Calculates the excess kurtosis of a 1D tensor."""
  mean = torch.mean(tensor)
  # Calculate the standard deviation
  std = torch.std(tensor)
  
  # The 4th moment over the standard deviation squared, minus 3
  kurt = torch.mean(((tensor - mean) / std) ** 4) - 3.0
  
  return kurt.item()


def generate_synthetic_distribution(dist_type: str, n_elements: int) -> torch.Tensor:
  """Generates a 1D array with specific tail properties."""
  
  if dist_type == "uniform":
    # Flat block. No tails, no outliers. Lowest possible kurtosis.
    return torch.rand(n_elements) * 2.0 - 1.0
  elif dist_type == "normal":
    # Standard bell curve. The baseline.
    return torch.randn(n_elements)
  elif dist_type == "laplace":
    # Pointy center, fat tails. High kurtosis.
    m = torch.distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    return m.sample([n_elements]).squeeze()
  elif dist_type == "cauchy":
    # Insanely fat tails. Extreme outliers. 
    # We must clamp it because Cauchy values can literally reach infinity.
    m = torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
    return torch.clamp(m.sample([n_elements]).squeeze(), min = -20.0, max = 20.0)  
  else:
    raise ValueError("Unknown distribution type")


def run_kurtosis_experiment(seed: int):
  print("========================================")
  print(" PHASE 3: The Kurtosis Showdown")
  print("========================================\n")
  
  fix_seed(seed)
  n_elements = 500_000
  
  # We test them from lowest kurtosis to highest kurtosis
  distributions = ["uniform", "normal", "laplace", "cauchy"]
  
  formats = {
    "INT8 (Uniform)": FormatINT8(),
    "FP8 (E4M3)": FormatFP8_E4M3(),
    "Posit8 (es = 0)": FormatPosit8(es = 0),
    "NF4 (4-bit Codebook)": FormatNF4()
  }
  
  for dist_name in distributions:
    print(f"\nTesting Distribution: {dist_name.upper()}")
    print("-" * 50)
    
    tensor = generate_synthetic_distribution(dist_name, n_elements)
    
    kurt_val = calculate_kurtosis(tensor)
    print(f"Calculated Excess Kurtosis: {kurt_val:.2f}\n")
    
    results = []
    for fmt_name, fmt_engine in formats.items():
      q_tensor = fmt_engine.fake_quantize(tensor)
      mse = calculate_mse(tensor, q_tensor)
      results.append((fmt_name, mse))
      
    # Sort results by MSE to determine the winner for this specific shape
    results.sort(key = lambda x: x[1])
    
    for rank, (fmt_name, mse) in enumerate(results, 1):
      print(f"{rank}. {fmt_name:<20} | MSE: {mse:.6f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Test how kurtosis changes format rankings.")
  parser.add_argument("--seed", type = int, default = 42, help = "The random seed for data generation.")
  args = parser.parse_args()
  
  run_kurtosis_experiment(args.seed)