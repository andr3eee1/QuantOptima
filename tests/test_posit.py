import math
import torch
from src.formats.posit_fmt import FormatPosit8


def test_posit_special_values():
  """Tests if Zero and NaR (Not a Real) are handled correctly according to the standard."""
  fmt = FormatPosit8(es = 0)
  
  # 0x00 is 0.0, and 0x80 is NaR (NaN in Python)
  t = torch.tensor([0.0, float('nan')])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 0.0, "Absolute zero was lost!"
  assert math.isnan(res[1].item()), "NaR (Not a Real) handling failed!"


def test_posit_golden_zone():
  """Tests if 1.0 and -1.0 are encoded with absolute perfection."""
  fmt = FormatPosit8(es = 0)
  
  # Posits are mathematically designed to have maximum accuracy exactly at 1.0 and -1.0.
  t = torch.tensor([1.0, -1.0])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 1.0, "Posit failed to represent 1.0 perfectly!"
  assert res[1].item() == -1.0, "Posit failed to represent -1.0 perfectly!"


def test_posit_max_bounds():
  """Tests the absolute limits of Posit8 with es = 0."""
  fmt = FormatPosit8(es = 0)
  
  # The maximum representable value in an 8-bit Posit with es = 0 is exactly 64.0.
  # Any massive outlier must snap to 64.0 or -64.0.
  t = torch.tensor([1000.0, -5000.0])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 64.0, "Upper bound clipping failed! Did not snap to 64.0."
  assert res[1].item() == -64.0, "Lower bound clipping failed! Did not snap to -64.0."


def test_posit_tapered_precision():
  """Tests dynamic bit allocation: huge numbers lose decimal precision, small numbers keep it."""
  fmt = FormatPosit8(es = 0)
  
  # 1.5 is near the center. The Regime is short, leaving plenty of bits for the Fraction.
  # It should be represented perfectly with no error.
  t_small = torch.tensor([1.5]) 
  res_small = fmt.fake_quantize(t_small)
  
  # 24.2 is a large outlier. The Regime eats 6 bits: [0 11111 0 ...]
  # This leaves exactly 1 bit for the fraction. 
  # The valid values here are 16.0 and 24.0. It loses the decimal and snaps to 24.0.
  t_large = torch.tensor([24.2])
  res_large = fmt.fake_quantize(t_large)
  
  assert res_small[0].item() == 1.5, "Lost precision in the high-accuracy golden zone!"
  assert res_large[0].item() == 24.0, "Tapered precision failed! Outlier did not lose its fraction."