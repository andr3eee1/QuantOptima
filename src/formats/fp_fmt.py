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
    # Per-tensor scaling
    absmax = fp32_tensor.abs().max()
    scale = absmax / self.max_val if absmax != 0 else torch.tensor(1e-8, device=fp32_tensor.device)
    scaled_tensor = fp32_tensor / scale
    
    sign_bit = (scaled_tensor < 0).to(torch.uint8)
    
    abs_x = torch.abs(scaled_tensor)
    abs_x = torch.clamp(abs_x, max=self.max_val)
    
    zero_mask = (abs_x == 0.0)
    
    # Calculate exponent for normals
    # We add a tiny epsilon to avoid log2(0) for exactly 0, even though it's masked later
    e = torch.floor(torch.log2(abs_x + 1e-12))
    
    # Subnormal mask (values smaller than 2^-6)
    subnormal_mask = (e < -6.0) & ~zero_mask
    
    e_clamped = torch.clamp(e, min=-6.0, max=8.0)
    exp_stored = (e_clamped + self.bias).to(torch.uint8)
    
    m_normal = abs_x / (2.0 ** e_clamped)
    m_stored = torch.clamp(torch.round((m_normal - 1.0) * self.m_steps), min=0.0, max=7.0).to(torch.uint8)
    
    # Fix subnormals
    exp_stored[subnormal_mask] = 0
    m_subnormal = abs_x[subnormal_mask] / (2.0 ** -9.0)
    m_stored[subnormal_mask] = torch.clamp(torch.round(m_subnormal), min=0.0, max=7.0).to(torch.uint8)
    
    encoded_byte = (sign_bit << 7) | (exp_stored << 3) | m_stored
    encoded_byte[zero_mask] = 0
    
    # Store the scale inside max_val and min_val so we can recover it
    return QuantizedTensor(data=encoded_byte, min_val=-absmax.item(), max_val=absmax.item())

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    b = quantized_packet.data
    
    sign_bit = (b >> 7) & 1
    exp_stored = (b >> 3) & 0b1111
    m_stored = b & 0b111
    
    sign_mult = 1.0 - (2.0 * sign_bit.to(torch.float32))
    
    subnormal_mask = (exp_stored == 0)
    
    # Normal decoding
    e = exp_stored.to(torch.float32) - self.bias
    m = 1.0 + (m_stored.to(torch.float32) / self.m_steps)
    fp32_tensor = sign_mult * (2.0 ** e) * m
    
    # Subnormal decoding
    fp32_tensor[subnormal_mask] = sign_mult[subnormal_mask] * (m_stored[subnormal_mask].to(torch.float32) * (2.0 ** -9.0))
    
    zero_mask = ((b & 0b01111111) == 0)
    fp32_tensor[zero_mask] = 0.0
    
    # 2. Per-tensor un-scaling
    scale = quantized_packet.max_val / self.max_val
    return fp32_tensor * scale

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet=self.encode(fp32_tensor)
    return self.decode(q_packet)


class FormatFP4_E2M1:
  def __init__(self):
    self.bias = 1
    self.m_bits = 1
    self.m_steps = 2 ** self.m_bits
    self.max_val = 6.0

  def encode(self, fp32_tensor: torch.Tensor) -> QuantizedTensor:
    absmax = torch.nan_to_num(fp32_tensor).abs().max()
    scale = absmax / self.max_val if absmax != 0 else torch.tensor(1e-8, device=fp32_tensor.device)
    scaled_tensor = fp32_tensor / scale
    
    sign_bit = (scaled_tensor < 0).to(torch.uint8)
    
    abs_x = torch.abs(scaled_tensor)
    abs_x = torch.clamp(abs_x, max=self.max_val)
    
    zero_mask = (abs_x == 0.0)
    
    e = torch.floor(torch.log2(abs_x + 1e-12))
    subnormal_mask = (e < 0.0) & ~zero_mask
    
    e_clamped = torch.clamp(e, min=0.0, max=2.0)
    exp_stored = (e_clamped + self.bias).to(torch.uint8)
    
    m_normal = abs_x / (2.0 ** e_clamped)
    m_stored = torch.clamp(torch.round((m_normal - 1.0) * self.m_steps), min=0.0, max=1.0).to(torch.uint8)
    
    exp_stored[subnormal_mask] = 0
    m_subnormal = abs_x[subnormal_mask] / (2.0 ** 0.0) # Subnormals in E2M1 use 2^0 as multiplier (2^(1-bias))
    m_stored[subnormal_mask] = torch.clamp(torch.round(m_subnormal * self.m_steps), min=0.0, max=1.0).to(torch.uint8)
    
    encoded_byte = (sign_bit << 3) | (exp_stored << 1) | m_stored
    encoded_byte[zero_mask] = 0
    
    return QuantizedTensor(data=encoded_byte, min_val=-absmax.item(), max_val=absmax.item())

  def decode(self, quantized_packet: QuantizedTensor) -> torch.Tensor:
    b = quantized_packet.data
    
    sign_bit = (b >> 3) & 1
    exp_stored = (b >> 1) & 0b11
    m_stored = b & 0b1
    
    sign_mult = 1.0 - (2.0 * sign_bit.to(torch.float32))
    
    subnormal_mask = (exp_stored == 0)
    
    e = exp_stored.to(torch.float32) - self.bias
    m = 1.0 + (m_stored.to(torch.float32) / self.m_steps)
    fp32_tensor = sign_mult * (2.0 ** e) * m
    
    fp32_tensor[subnormal_mask] = sign_mult[subnormal_mask] * (m_stored[subnormal_mask].to(torch.float32) / self.m_steps * (2.0 ** 0.0))
    
    zero_mask = ((b & 0b01111) == 0)
    fp32_tensor[zero_mask] = 0.0
    
    scale = quantized_packet.max_val / self.max_val
    return fp32_tensor * scale

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet=self.encode(fp32_tensor)
    return self.decode(q_packet)