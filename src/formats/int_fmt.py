from dataclasses import dataclass
import torch

@dataclass
class QuantizedTensor:
  data: torch.Tensor
  min_val: float
  max_val: float

class FormatINT8:
  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    absmax = fp32_tensor.abs().max()
    # scale factor ensures the max value maps to 127
    scale = absmax / 127.0 if absmax != 0 else torch.tensor(1e-8, device=fp32_tensor.device)
    
    # Symmetric quantization: divide by scale and round
    q_tensor = torch.round(fp32_tensor / scale)
    q_tensor = torch.clamp(q_tensor, min=-127.0, max=127.0)
    
    return QuantizedTensor(data=q_tensor.to(torch.int8), min_val=-absmax.item(), max_val=absmax.item())

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    # Recover scale from min_val/max_val trick
    scale = quantized_packet.max_val / 127.0
    
    fp32_tensor = quantized_packet.data.to(torch.float32)
    return fp32_tensor * scale

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet = self.encode(fp32_tensor)
    return self.decode(q_packet)


class FormatINT4:
  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    absmax = torch.nan_to_num(fp32_tensor).abs().max()
    scale = absmax / 7.0 if absmax != 0 else torch.tensor(1e-8, device=fp32_tensor.device)
    
    # Symmetric quantization
    q_tensor = torch.round(fp32_tensor / scale)
    q_tensor = torch.clamp(q_tensor, min=-7.0, max=7.0)
    
    return QuantizedTensor(data=q_tensor.to(torch.int8), min_val=-absmax.item(), max_val=absmax.item())

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    scale = quantized_packet.max_val / 7.0
    fp32_tensor = quantized_packet.data.to(torch.float32)
    return fp32_tensor * scale

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet = self.encode(fp32_tensor)
    return self.decode(q_packet)