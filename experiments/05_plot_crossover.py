import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.formats.int_fmt import FormatINT8
from src.formats.fp_fmt import FormatFP8_E4M3
from src.formats.posit_fmt import FormatPosit8
from src.utils.metrics import calculate_mse

def calculate_kurtosis(tensor: torch.Tensor) -> float:
  mean = torch.mean(tensor)
  std = torch.std(tensor)
  kurt = torch.mean(((tensor - mean) / std) ** 4) - 3.0
  return kurt.item()


def generate_mixed_distribution(alpha: float, n_elements: int) -> torch.Tensor:
  """
  Blends a uniform distribution (low kurtosis) with a Cauchy distribution (extreme kurtosis)
  using the alpha parameter [0.0, 1.0] to create a sliding scale of fat tails.
  """
  uniform = torch.rand(n_elements) * 2.0 - 1.0
  
  m = torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
  cauchy = torch.clamp(m.sample([n_elements]).squeeze(), min = -20.0, max = 20.0)
  
  return (1.0 - alpha) * uniform + alpha * cauchy


def plot_crossover():
  print("Generating Kurtosis Crossover Graph...")
  n_elements = 100_000
  steps = 20
  
  kurtosis_vals = []
  mse_int8 = []
  mse_fp8 = []
  mse_posit = []
  
  int8_engine = FormatINT8()
  fp8_engine = FormatFP8_E4M3()
  posit_engine = FormatPosit8(es = 0)
  
  # Slide alpha from 0.0 to 0.4 to slowly inject more extreme outliers
  for i in range(steps):
    alpha = i / (steps - 1) * 0.4 
    tensor = generate_mixed_distribution(alpha, n_elements)
    
    kurtosis_vals.append(calculate_kurtosis(tensor))
    mse_int8.append(calculate_mse(tensor, int8_engine.fake_quantize(tensor)))
    mse_fp8.append(calculate_mse(tensor, fp8_engine.fake_quantize(tensor)))
    mse_posit.append(calculate_mse(tensor, posit_engine.fake_quantize(tensor)))
    
  plt.figure(figsize = (10, 6))
  plt.plot(kurtosis_vals, mse_int8, label = "INT8", color = 'blue', linewidth = 2)
  plt.plot(kurtosis_vals, mse_fp8, label = "FP8 (E4M3)", color = 'red', linewidth = 2)
  plt.plot(kurtosis_vals, mse_posit, label = "Posit8", color = 'green', linewidth = 2)
  
  plt.title("Format Supremacy Inversion based on Kurtosis")
  plt.xlabel("Excess Kurtosis (κ)")
  plt.ylabel("Mean Squared Error (MSE)")
  plt.legend()
  plt.grid(True)
  
  os.makedirs("results/figures", exist_ok = True)
  plt.savefig("results/figures/kurtosis_crossover.png")
  print("Graph saved to results/figures/kurtosis_crossover.png!")


if __name__ == "__main__":
  plot_crossover()