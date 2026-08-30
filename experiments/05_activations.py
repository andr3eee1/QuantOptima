import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from src.model.network import CifarCNN


def calculate_kurtosis(tensor: torch.Tensor) -> float:
  mean = torch.mean(tensor)
  std = torch.std(tensor)
  if std == 0:
    return 0.0
  kurt = torch.mean(((tensor - mean) / std) ** 4)
  return kurt.item()


def run_activations_analysis(seed: int = 42):
  print("\n==================================================")
  print(" PHASE 3 BONUS: Activations Kurtosis Analysis")
  print("==================================================")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = CifarCNN().to(device)
  model.load_state_dict(torch.load(
      f"results/models/model_seed_{seed}.pth", map_location=device, weights_only=True))
  model.eval()

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  test_dataset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform)
  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=128, shuffle=False)

  activations = {}

  def get_activation(name):
    def hook(model, input, output):
      if name not in activations:
        activations[name] = input[0].detach().cpu().flatten()
      else:
        activations[name] = torch.cat(
            (activations[name], input[0].detach().cpu().flatten()))
    return hook

  hooks = []
  for name, layer in model.named_modules():
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
      hooks.append(layer.register_forward_hook(get_activation(name)))

  print("Extracting activations from the first 5 batches...")
  with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
      if i >= 5:
        break
      images = images.to(device)
      model(images)

  for h in hooks:
    h.remove()

  os.makedirs("results/plots", exist_ok=True)

  print(f"\n{'Layer (Inputs)':<15} | {'Weight Kurtosis':<15} | {'Activation Kurtosis':<20}")
  print("-" * 55)

  for name, act_tensor in activations.items():
    act_kurtosis = calculate_kurtosis(act_tensor)

    weight_tensor = dict(model.named_parameters())[
        f"{name}.weight"].detach().cpu().flatten()
    weight_kurtosis = calculate_kurtosis(weight_tensor)

    print(f"{name:<15} | {weight_kurtosis:<15.2f} | {act_kurtosis:<20.2f}")

    plt.figure(figsize=(8, 5))
    plt.hist(act_tensor.numpy(), bins=150,
             alpha=0.75, color='purple', log=True)
    plt.title(
        f"Activation Distribution: {name} (Kurtosis: {act_kurtosis:.2f})")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/plots/activation_hist_{name}.png")
    plt.close()

  print("\nActivation histograms saved to results/plots/")


if __name__ == "__main__":
  run_activations_analysis()
