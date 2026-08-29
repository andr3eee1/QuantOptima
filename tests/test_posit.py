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
  """Tests the absolute limits of Posit8 with es = 0 under per-tensor scaling."""
  fmt = FormatPosit8(es = 0)
  
  # With per-tensor scaling, -5000 is perfectly recovered (modulo floating noise)
  t = torch.tensor([1000.0, -5000.0])
  res = fmt.fake_quantize(t)
  
  assert torch.abs(res[1] - (-5000.0)) < 1.0, "The max outlier was not perfectly scaled and recovered!"

def test_posit_tapered_precision():
  """Tests dynamic bit allocation: huge numbers lose decimal precision, small numbers keep it."""
  fmt = FormatPosit8(es = 0)
  
  # By adding 64.0, we force the scale factor to be exactly 1.0 (since max_val is 64.0)
  t_small = torch.tensor([1.5, 64.0])
  res_small = fmt.fake_quantize(t_small)
  
  # 24.2 loses its fraction and snaps to 24.0 when scale = 1.0
  t_large = torch.tensor([24.2, 64.0])
  res_large = fmt.fake_quantize(t_large)
  
  assert res_small[0].item() == 1.5, "Lost precision in the high-accuracy golden zone!"
  assert res_large[0].item() == 24.0, "Tapered precision failed! Outlier did not lose its fraction."