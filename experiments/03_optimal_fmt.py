import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from src.model.network import CifarCNN
from src.quantization.apply import apply_optimal_codebooks
from src.utils.metrics import calculate_accuracy
from src.utils.seed import fix_seed

def run_optimal_benchmark(seed: int):
  print("========================================")
  print(" PHASE 2: Testing Optimal Codebooks")
  print("========================================\n")
  
  fix_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Active Seed: {seed}")
  
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)
  
  base_model = CifarCNN().to(device)
  base_model.load_state_dict(torch.load("results/models/cifar_fp32_baseline.pth", map_location = device))
  base_model.eval()
  
  print("Evaluating FP32 Baseline...")
  base_acc = calculate_accuracy(base_model, test_loader, device)
  print(f"Baseline Accuracy: {base_acc:.2f}%\n")
  
  print("Calculating OPTIM 8-bit codebooks...")
  optim8_model = apply_optimal_codebooks(base_model, bits = 8)
  acc_8 = calculate_accuracy(optim8_model, test_loader, device)
  
  print("\nCalculating OPTIM 4-bit codebooks...")
  optim4_model = apply_optimal_codebooks(base_model, bits = 4)
  acc_4 = calculate_accuracy(optim4_model, test_loader, device)
  
  print("\nPhase 2 Final Results:")
  print("-" * 45)
  print(f"{'Format':<15} | {'Bits':<5} | {'Accuracy (%)':<15}")
  print("-" * 45)
  print(f"{'OPTIM (Lloyd)':<15} | {'8':<5} | {acc_8:>6.2f}%")
  print(f"{'OPTIM (Lloyd)':<15} | {'4':<5} | {acc_4:>6.2f}%")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Test optimal quantization formats.")
  parser.add_argument("--seed", type = int, default = 42, help = "Random seed.")
  args = parser.parse_args()
  
  run_optimal_benchmark(args.seed)