import torch
from src.formats.fp_fmt import FormatFP8_E4M3

def test_zero_handling():
  """Tests if positive and negative zeros are handled without crashing."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([0.0, -0.0])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 0.0, "Absolute zero was lost!"
  assert res[1].item() == 0.0, "Negative zero is not handled correctly!"


def test_clipping_bounds():
  """Tests if massive outliers are properly clipped to the E4M3 limits."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([1000.0, -5000.0])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 448.0, "Clipping failed at the upper bound!"
  assert res[1].item() == -448.0, "Clipping failed at the lower bound!"


def test_known_quantization_error():
  """Tests a specific known fractional value to verify mantissa precision."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([14.5])
  res = fmt.fake_quantize(t)
  
  assert res[0].item() == 14.0, "Banker's rounding failed! Did not map 14.5 to 14.0"


def test_exact_powers_of_two():
  """Tests if exact powers of two maintain perfect precision."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([2.0, 8.0, 128.0])
  res = fmt.fake_quantize(t)
  
  assert torch.equal(res, t), "Powers of 2 suffered precision degradation!"

def test_subnormals():
  """Tests if subnormal values are handled correctly, as required by C2."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([0.001])
  res = fmt.fake_quantize(t)
  
  # Before fixing, 0.001 decodes to 0.015625
  # E4M3 min subnormal is 2^-9 (0.001953125). 
  # Depending on rounding, it should be 0.0 or 0.001953125.
  assert res[0].item() < 0.01, f"Expected subnormal handling, got {res[0].item()}"