import torch
import torch.nn.functional as F


def pad_to_shape(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    pad_sizes = []
    for i in range(tensor.ndim - 1, -1, -1):
        pad = target_shape[i] - tensor.shape[i]
        pad_sizes.extend([0, pad])
    return F.pad(tensor, pad_sizes, mode='constant', value=0)

def pad_and_concat(tensors : list[torch.Tensor]) -> torch.Tensor:
    # Compute target shape (max along each dimension)
    target_shape = torch.tensor([t.shape for t in tensors]).max(dim=0).values.tolist()
    padded = [pad_to_shape(t, target_shape) for t in tensors]
    return torch.cat(padded, dim=0)