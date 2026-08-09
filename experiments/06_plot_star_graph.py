import os
import matplotlib.pyplot as plt

def plot_star_graph():
  print("Generating the Star Graph (Accuracy vs Bits)...")
  
  # The exact numbers from our benchmark results
  bits = [32, 8, 4]
  
  # FP32 -> FP8 (E4M3) -> Missing 4-bit FP so we drop it
  acc_fp = [75.05, 74.68, None] 
  
  # FP32 -> INT8 -> NF4 (Best standard integer/normal mapping)
  acc_std = [75.05, 75.07, 74.31]
  
  # FP32 -> OPTIM 8 -> OPTIM 4
  acc_optim = [75.05, 74.98, 74.08]
  
  plt.figure(figsize = (10, 6))
  
  # Plot the lines
  plt.plot([32, 8, 4], acc_std, marker = 'o', label = "Standard (INT8 / NF4)", color = 'blue', linewidth = 2)
  plt.plot([32, 8, 4], acc_optim, marker = 'x', linestyle = '--', label = "OPTIM (Lloyd-Max)", color = 'red', linewidth = 2)
  
  # We only have FP8, so we plot a segment
  plt.plot([32, 8], [75.05, 74.68], marker = 's', label = "FP Formats (FP8)", color = 'green', linewidth = 2)

  plt.title("The Star Graph: Model Accuracy vs. Bit-Width")
  plt.xlabel("Bits per Weight")
  plt.ylabel("Test Accuracy (%)")
  
  # Invert X axis so it goes 32 -> 8 -> 4
  plt.xlim(35, 2) 
  
  plt.legend()
  plt.grid(True)
  
  os.makedirs("results/figures", exist_ok = True)
  plt.savefig("results/figures/star_graph_accuracy.png")
  print("Saved to results/figures/star_graph_accuracy.png!")

if __name__ == "__main__":
  plot_star_graph()