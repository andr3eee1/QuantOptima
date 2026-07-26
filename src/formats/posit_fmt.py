from dataclasses import dataclass
import torch

class FormatPosit8:
  def __init__(self, es: int = 0):
    self.es = es
    self.useed = 2 ** (2 ** self.es)

    valid_floats = []
    for b in range(256):
      valid_floats.append(self._decode_byte(b))
    self.lut = torch.tensor(valid_floats, dtype = torch.float32)

  def encode(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    safe_lut = self.lut.clone()
    safe_lut[128] = float('inf')

    nan_mask = torch.isnan(fp32_tensor)
    distances = torch.abs(fp32_tensor.unsqueeze(-1) - safe_lut)
    quantized_bytes = torch.argmin(distances, dim = -1).to(torch.uint8)
    quantized_bytes[nan_mask] = 128
    
    return quantized_bytes

  def _decode_byte(self, b: int) -> float:
    b &= 255

    if b == 0:
      return 0.0
    if b == 128:
      return float('nan')

    sign = (b >> 7) & 1
    if sign == 1:
      b = (-b) & 255

    regime_bit = (b >> 6) & 1
    k = 0
    bit_idx = 5
    while bit_idx >= 0 and ((b >> bit_idx) & 1) == regime_bit:
      k += 1
      bit_idx -= 1

    if regime_bit == 0:
      k = -(k + 1)

    bit_idx -= 1

    exp = 0
    if self.es > 0 and bit_idx >= 0:
      exp_bits = min(self.es, bit_idx + 1)
      exp = (b >> (bit_idx + 1 - exp_bits)) & ((1 << exp_bits) - 1)
      exp <<= (self.es - exp_bits)
      bit_idx -= exp_bits

    f = 0.0
    if bit_idx >= 0:
      f_bits = bit_idx + 1
      f_int = b & ((1 << f_bits) - 1)
      f = f_int / (1 << f_bits)

    val = (self.useed ** k) * (2 ** exp) * (1.0 + f)

    return (1 - 2 * sign) * val

  def decode(self, quantized_packet: torch.Tensor) -> torch.Tensor:
    raw_bytes = quantized_packet.view(-1).cpu().tolist()

    decoded_floats = []
    for b in raw_bytes:
      decoded_floats.append(self._decode_byte(b))
    
    return torch.tensor(decoded_floats, dtype = torch.float32).view(quantized_packet.shape)

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    q_packet=self.encode(fp32_tensor)
    return self.decode(q_packet)