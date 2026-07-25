import torch
from src.formats.int_fmt import FormatINT8

def test_exact_zero_mapping():
  fmt = FormatINT8()
  t = torch.tensor([-15.0, 0.0, 15.0])
  res = fmt.fake_quantize(t)
  assert res[1].item() == 0.0, "Zero point shifted during quantization!"

def test_uniform_step_size():
  fmt = FormatINT8()
  # The step size delta must be strictly equal across the entire representable range.
  t = torch.linspace(-10.0, 10.0, steps=1000)
  res = fmt.fake_quantize(t)
  unique_vals = torch.unique(res)
  diffs = torch.diff(unique_vals)
  max_diff_variance = torch.max(diffs) - torch.min(diffs)
  # Float precision math might leave tiny artifacts, so we check under an epsilon
  assert max_diff_variance < 1e-5, "Grid spacing is not uniform!"

def test_idempotence():
  fmt = FormatINT8()
  # Quantizing an already quantized tensor shouldn't degrade it further. 
  # Q(Q(x)) == Q(x)
  t = torch.randn(500) * 100
  res1 = fmt.fake_quantize(t)
  res2 = fmt.fake_quantize(res1)
  assert torch.equal(res1, res2), "Quantization is not idempotent!"

def test_clipping_bounds():
  fmt = FormatINT8()
  # INT8 has exactly 256 states. We throw massive outliers to see if they clip safely.
  t = torch.tensor([-99999.0, 1.0, 99999.0])
  res = fmt.fake_quantize(t)
  # The outliers should hit the maximum representable bucket, but not overflow.
  assert res[0].item() < res[1].item(), "Negative overflow failed!"
  assert res[2].item() > res[1].item(), "Positive overflow failed!"