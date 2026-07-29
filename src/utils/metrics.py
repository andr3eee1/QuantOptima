import torch

def calculate_mse(original_tensor, quantized_tensor):
  """Calculates the mean squared error between two tensors."""
  return torch.mean((original_tensor - quantized_tensor) ** 2).item()

def calculate_accuracy(model, data_loader, device):
  """Runs the model on the dataset and returns the % of correct predictions."""
  model.eval()
  correct = 0
  total = 0
  
  with torch.no_grad():
    for data in data_loader:
      inputs, labels = data[0].to(device), data[1].to(device)
      outputs = model(inputs)
      
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      
  return 100.0 * correct / total