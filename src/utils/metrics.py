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

def get_correct_predictions(model, data_loader, device):
  """Returns a boolean tensor of correct predictions for each image in the dataset."""
  model.eval()
  all_correct = []
  
  with torch.no_grad():
    for data in data_loader:
      inputs, labels = data[0].to(device), data[1].to(device)
      outputs = model(inputs)
      
      _, predicted = torch.max(outputs.data, 1)
      all_correct.append(predicted == labels)
      
  return torch.cat(all_correct)

def mcnemar_test(preds_a, preds_b):
  """
  Calculates McNemar's test statistic and p-value for two sets of boolean predictions.
  preds_a, preds_b: 1D boolean tensors of the same length.
  """
  import scipy.stats as stats
  
  # b: model A correct, model B incorrect
  b = torch.sum(preds_a & ~preds_b).item()
  
  # c: model A incorrect, model B correct
  c = torch.sum(~preds_a & preds_b).item()
  
  if b + c == 0:
    return 0.0, 1.0 # No difference
    
  statistic = ((abs(b - c) - 1.0) ** 2) / (b + c) # with continuity correction
  p_value = stats.chi2.sf(statistic, 1) # chi-squared survival function for 1 degree of freedom
  
  return statistic, p_value