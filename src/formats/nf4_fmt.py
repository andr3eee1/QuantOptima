from dataclasses import dataclass
import torch

@dataclass
class QuantizedTensor:
  data: torch.Tensor
  abs_max: float

class FormatNF4:
  def __init__(self):
    self.nf4_table=torch.tensor([
      -1.0, -0.69619280, -0.52507305, -0.39491749, 
      -0.28444138, -0.18477343, -0.09105004, 0.0, 
      0.07958030, 0.16093020, 0.24611230, 0.33791524, 
      0.44070983, 0.56261700, 0.72295684, 1.0
    ])

  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    abs_max = fp32_tensor.abs().max().item()
    if abs_max == 0.0:
      abs_max = 1e-8

    normalized = fp32_tensor / abs_max
    distances = torch.abs(normalized.unsqueeze(-1) - self.nf4_table)
    quantized_indices = torch.argmin(distances, dim = -1).to(torch.uint8)
    
    return QuantizedTensor(data = quantized_indices, abs_max = abs_max)

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    normalized = self.nf4_table[quantized_packet.data.to(torch.long)]
    
    return normalized * quantized_packet.abs_max

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet=self.encode(fp32_tensor)
    return self.decode(q_packet)