import argparse
import copy
import torch
import torchvision
import torchvision.transforms as transforms

from src.model.network import CifarCNN
from src.formats.int_fmt import FormatINT8, FormatINT4
from src.formats.fp_fmt import FormatFP8_E4M3, FormatFP4_E2M1
from src.formats.nf4_fmt import FormatNF4
from src.formats.posit_fmt import FormatPosit8

from src.utils.metrics import calculate_accuracy
from src.utils.seed import fix_seed


def quantize_weights_in_place(model, format_engine):
  """Iterates over the network and overwrites FP32 arrays with quantized values."""
  q_model = copy.deepcopy(model)
  with torch.no_grad():
    for name, param in q_model.named_parameters():
      if 'weight' in name:
        param.copy_(format_engine.fake_quantize(param))
  return q_model


def run_benchmark(seed: int):
  print("========================================")
  print(" PHASE 1: Benchmarking Standard Formats")
  print("========================================\n")
  
  fix_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Active Seed: {seed}")
  
  # Load the test dataset (the images the model has NEVER seen during training)
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False)
  
  # Load the trained FP32 memory state
  base_model = CifarCNN().to(device)
  base_model.load_state_dict(torch.load(f"results/models/model_seed_{seed}.pth", map_location = device))
  base_model.eval()
  
  print("Evaluating FP32 Baseline...")
  base_acc = calculate_accuracy(base_model, test_loader, device)
  print(f"Baseline Accuracy: {base_acc:.2f}%\n")
  
  # Define the quantization engines
  formats = {
    "INT8": (FormatINT8(), 8),
    "FP8 (E4M3)": (FormatFP8_E4M3(), 8),
    "Posit8": (FormatPosit8(es = 0), 8),
    "NF4": (FormatNF4(), 4),
    "INT4": (FormatINT4(), 4),
    "FP4 (E2M1)": (FormatFP4_E2M1(), 4)
  }
  
  results = []
  
  for name, (fmt, bits) in formats.items():
    print(f"Applying bitwise quantization: {name}...")
    q_model = quantize_weights_in_place(base_model, fmt)
    acc = calculate_accuracy(q_model, test_loader, device)
    results.append((name, bits, acc))
    
  print("\nQuantization Benchmark Results:")
  print("-" * 50)
  print(f"{'Format':<15} | {'Bits':<5} | {'Accuracy (%)':<15}")
  print("-" * 50)
  
  # Sort results descending by accuracy to easily spot the winner
  results.sort(key = lambda x: x[2], reverse = True)
  
  for name, bits, acc in results:
    drop = base_acc - acc
    print(f"{name:<15} | {bits:<5} | {acc:>6.2f}% (Drop: {drop:.2f}%)")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Benchmark quantization formats on the trained model.")
  parser.add_argument("--seed", type = int, default = 42, help = "The random seed for data loading reproducibility.")
  args = parser.parse_args()
  
  run_benchmark(args.seed)