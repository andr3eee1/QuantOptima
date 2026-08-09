import copy
import torch

from src.quantization.lloyd_max import LloydMaxQuantizer

def apply_optimal_codebooks(model, bits):
  """
  Iterates throught the network layers and calculates a custom optimal 
  codebook for each weight matrix using the Lloyd-Max algorithm.
  """
  q_model = copy.deepcopy(model)
  quantizer = LloydMaxQuantizer(bits = bits)
  
  with torch.no_grad():
    for name, param in q_model.named_parameters():
      if 'weight' in name:
        print(f"Optimizing codebook for {name} ({param.numel()} weights)...")
        optimal_tensor = quantizer.fake_quantize(param)
        param.copy_(optimal_tensor)
        
  return q_model