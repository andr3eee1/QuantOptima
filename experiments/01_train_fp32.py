import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.utils.seed import fix_seed
from src.model.network import CifarCNN


def plot_weight_histograms(model):
  """Extracts weights and generates histograms in results/histograms/."""
  os.makedirs("results/histograms", exist_ok = True)
  
  for name, param in model.named_parameters():
    if 'weight' in name:
      plt.figure(figsize = (8, 5))
      # Flatten the tensor into a 1D contiguous array for the histogram
      weights = param.detach().cpu().numpy().flatten()
      
      plt.hist(weights, bins = 150, alpha = 0.75, color = 'blue')
      plt.title(f"Weight Distribution: {name}")
      plt.xlabel("Value")
      plt.ylabel("Frequency")
      plt.grid(True)
      
      safe_name = name.replace(".", "_")
      save_path = f"results/histograms/{safe_name}.png"
      plt.savefig(save_path)
      plt.close()
      
  print("Histograms generated and saved to results/histograms/")


def train_baseline_model(seed: int):
  print("========================================")
  print(" PHASE 1: Training FP32 Baseline & Histograms")
  print("========================================\n")
  
  fix_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Active Seed: {seed}")
  print(f"Training on device: {device}")

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 2)

  model = CifarCNN().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = 0.001)

  epochs = 5
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      
      optimizer.zero_grad() # Reset gradients
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
      if i % 200 == 199:
        print(f"[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 200:.3f}")
        running_loss = 0.0

  os.makedirs("results/models", exist_ok = True)
  save_path = "results/models/cifar_fp32_baseline.pth"
  torch.save(model.state_dict(), save_path)
  print(f"\nTraining Complete! Baseline FP32 weights saved to: {save_path}")
  
  print("\nExtracting weights and generating histograms...")
  plot_weight_histograms(model)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Train the FP32 baseline model on CIFAR-10.")
  parser.add_argument("--seed", type = int, default = 42, help = "The random seed for training reproducibility.")
  args = parser.parse_args()
  
  train_baseline_model(args.seed)