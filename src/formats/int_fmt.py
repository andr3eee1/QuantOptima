from dataclasses import dataclass
import torch

@dataclass
class QuantizedTensor:
  data: torch.Tensor
  min_val: float
  max_val: float

class FormatINT8:
  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    min_val = fp32_tensor.min()
    max_val = fp32_tensor.max()

    denominator = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
    fp32_tensor = ((fp32_tensor - min_val) / denominator) * 254 - 127

    return QuantizedTensor(data = torch.round(fp32_tensor).to(torch.int8), min_val = min_val, max_val = max_val)

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    fp32_tensor = quantized_packet.data.to(torch.float32)
    range_val = quantized_packet.max_val - quantized_packet.min_val

    return ((fp32_tensor + 127) / 254) * range_val + quantized_packet.min_val

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet = self.encode(fp32_tensor)
    return self.decode(q_packet)