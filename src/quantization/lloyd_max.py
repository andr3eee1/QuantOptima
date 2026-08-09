import torch
 
class LloydMaxQuantizer:
  """
  Implements the Lloyd-Max algorithm (1D k-means) to find the theoretically 
  optimal quantization codebook for a given tensor distribution.
  """
  
  def __init__(self, bits: int, max_iter: int = 50, tol: float = 1e-5):
    self.bits = bits
    self.n_clusters = 2 ** bits
    self.max_iter = max_iter
    self.tol = tol

  def _initialize_centroids(self, tensor: torch.Tensor) -> torch.Tensor:
    # We randomly sample 'n_clusters' elements directly from the tensor 
    # to serve as our starting codebook values.
    indices = torch.randperm(tensor.numel(), device = tensor.device)[:self.n_clusters]
    
    return tensor[indices].clone()

  def _assign_clusters(self, tensor: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # We use broadcasting to create an [N, K] matrix of distances.
    distances = torch.abs(tensor.unsqueeze(1) - centroids.unsqueeze(0))
    
    # Return the index of the closest centroid for every single weight.
    return torch.argmin(distances, dim = 1)

  def _update_centroids(self, tensor: torch.Tensor, cluster_indices: torch.Tensor, old_centroids: torch.Tensor) -> torch.Tensor:
    # bincount calculates the sum of all weights assigned to each cluster
    sums = torch.bincount(cluster_indices, weights = tensor, minlength = self.n_clusters)
    
    # bincount without weights calculates how many elements are in each cluster
    counts = torch.bincount(cluster_indices, minlength = self.n_clusters)
    
    new_centroids = torch.zeros_like(old_centroids)
    
    for i in range(self.n_clusters):
      if counts[i] > 0:
        new_centroids[i] = sums[i] / counts[i]
      else:
        # If a cluster is completely empty, we keep the old centroid to prevent a division by zero (NaN).
        new_centroids[i] = old_centroids[i]
        
    return new_centroids

  def fit(self, tensor: torch.Tensor) -> torch.Tensor:
    flat_tensor = tensor.view(-1)
    centroids = self._initialize_centroids(flat_tensor)
    
    for iteration in range(self.max_iter):
      indices = self._assign_clusters(flat_tensor, centroids)
      new_centroids = self._update_centroids(flat_tensor, indices, centroids)
      
      # Calculate the maximum distance any centroid moved during this iteration.
      shift = torch.max(torch.abs(new_centroids - centroids)).item()
      
      if shift < self.tol:
        break
        
      centroids = new_centroids
      
    return centroids

  def fake_quantize(self, fp32_tensor: torch.Tensor) -> torch.Tensor:
    original_shape = fp32_tensor.shape
    flat_tensor = fp32_tensor.view(-1)
    
    # Run Lloyd-Max to find the perfect codebook
    centroids = self.fit(flat_tensor)
    
    # Map every original weight to its optimal codebook index
    indices = self._assign_clusters(flat_tensor, centroids)
    
    # Construct the new quantized array by indexing the codebook
    quantized_flat = centroids[indices]
    
    # Reshape it back into the original 2D/3D/4D matrix structure
    return quantized_flat.view(original_shape)