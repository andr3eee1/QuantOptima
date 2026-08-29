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
  """Tests if massive outliers scale the whole tensor safely instead of clipping."""
  fmt = FormatFP8_E4M3()
  t = torch.tensor([1000.0, -5000.0])
  res = fmt.fake_quantize(t)
  
  # With per-tensor scaling, -5000.0 determines the scale and maps to -448.0 internally.
  # So -5000.0 is perfectly recovered (modulo some FP math noise), and 1000.0 is scaled down.
  assert torch.abs(res[1] - (-5000.0)) < 1.0, "The max outlier was not perfectly scaled and recovered!"

def test_known_quantization_error():
  """Tests a specific known fractional value to verify mantissa precision."""
  fmt = FormatFP8_E4M3()
  # By adding 448.0, we force the scale factor to be exactly 1.0
  # This allows us to test raw E4M3 banker's rounding on 14.5
  t = torch.tensor([14.5, 448.0])
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
  # Add 448.0 to the tensor so per-tensor scaling leaves 0.001 as 0.001
  t = torch.tensor([0.001, 448.0])
  res = fmt.fake_quantize(t)
  
  # Before fixing, 0.001 decodes to 0.015625
  # E4M3 min subnormal is 2^-9 (0.001953125). 
  # Depending on rounding, it should be 0.0 or 0.001953125.
  assert res[0].item() < 0.01, f"Expected subnormal handling, got {res[0].item()}"