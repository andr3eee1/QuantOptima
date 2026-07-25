from dataclasses import dataclass
import torch

@dataclass
class QuantizedTensor:
  data: torch.Tensor
  min_val: float
  max_val: float

class FormatFP8_E4M3:
  def __init__(self):
    self.bias = 7
    self.m_bits = 3
    self.m_steps = 2 ** self.m_bits
    self.max_val = 448.0

  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    sign_bit = (fp32_tensor < 0).to(torch.uint8)
    
    abs_x = torch.abs(fp32_tensor)
    abs_x = torch.clamp(abs_x, max = self.max_val)
    
    zero_mask = (abs_x == 0.0)
    abs_x[zero_mask] = 1e-9
    
    e = torch.floor(torch.log2(abs_x))
    e = torch.clamp(e, min = -6.0, max = 8.0)
    exp_stored = (e + self.bias).to(torch.uint8)
    
    m = abs_x / (2.0 ** e)
    m_stored = torch.clamp(torch.round((m - 1.0) * self.m_steps), max = 7.0).to(torch.uint8)
    
    encoded_byte = (sign_bit << 7) | (exp_stored << 3) | m_stored
    
    encoded_byte[zero_mask] = 0
    
    return QuantizedTensor(data = encoded_byte, min_val = -self.max_val, max_val = self.max_val)

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    b = quantized_packet.data
    
    sign_bit = (b >> 7) & 1
    exp_stored = (b >> 3) & 0b1111
    m_stored = b & 0b111
    
    sign_mult = 1.0 - (2.0 * sign_bit.to(torch.float32))
    e = exp_stored.to(torch.float32) - self.bias
    m = 1.0 + (m_stored.to(torch.float32) / self.m_steps)
    
    fp32_tensor = sign_mult * (2.0 ** e) * m
    
    zero_mask = ((b & 0b01111111) == 0)
    fp32_tensor[zero_mask] = 0.0
    
    return fp32_tensor

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet=self.encode(fp32_tensor)
    return self.decode(q_packet)