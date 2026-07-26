import torch
from src.formats.nf4_fmt import FormatNF4

def test_nf4_exact_mapping():
  """If we input exactly the scaled codebook values, the error must be 0."""
  fmt = FormatNF4()
  abs_max = 5.0
  # Create a tensor containing exactly the "ideal" points of the format
  t = fmt.nf4_table * abs_max
  res = fmt.fake_quantize(t)
  
  # Use allclose instead of equal to ignore infinitesimal floating-point inaccuracies
  assert torch.allclose(res, t, atol=1e-6), "The codebook does not perfectly map native values!"

def test_nf4_zero():
  """Zero is a critical value and has its own dedicated index (7) in NF4."""
  fmt = FormatNF4()
  # NF4 scales the entire tensor based on the largest absolute element
  t = torch.tensor([-2.5, 0.0, 2.5])
  res = fmt.fake_quantize(t)
  
  assert res[1].item() == 0.0, "NF4 missed the absolute zero value!"

def test_nf4_4bit_constraint():
  """Ensure that the encode never generates indices outside the 4-bit range."""
  fmt = FormatNF4()
  t = torch.randn(10000) * 999.0
  q_packet = fmt.encode(t)
  
  min_index = q_packet.data.min().item()
  max_index = q_packet.data.max().item()
  
  assert min_index >= 0, "Negative index generated! The 4-bit lower limit was breached."
  assert max_index <= 15, "Exceeded value 15! The 4-bit upper limit was breached."
  assert q_packet.data.dtype == torch.uint8, "Memory type is not uint8!"

def test_nf4_asymmetric_scale():
  """NF4 uses pure symmetric scaling (abs_max). Verifying the behavior."""
  fmt = FormatNF4()
  t = torch.tensor([10.0, -100.0, 50.0])
  q_packet = fmt.encode(t)
  
  assert q_packet.abs_max == 100.0, "Asymmetry broke the abs_max calculation!"