import torch

def calculate_mse(original_tensor, quantized_tensor):
  """Calculates the mean squared error between two tensors."""
  return torch.mean((original_tensor - quantized_tensor) ** 2).item()

def calculate_accuracy(model, data_loader):
  """Runs the model on the dataset and returns the % of correct predictions."""
  pass