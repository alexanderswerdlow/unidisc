import torch
import math

def get_interleaved_indices(modality):
    modality_mask = modality.bool()

    # Pad input_mask with zeros at both ends along the sequence dimension
    pad_input_mask = torch.nn.functional.pad(modality_mask, (1, 1), mode='constant', value=0)  # Shape: [B, N+2]

    # Compute the difference along the sequence dimension to find transitions
    diff = pad_input_mask[:, 1:].float() - pad_input_mask[:, :-1].float()  # Shape: [B, N+1]

    # Find start/end positions
    starts = (diff == 1).nonzero(as_tuple=False)  # Shape: [num_blocks, 2], columns: [batch_idx, position]
    ends = (diff == -1).nonzero(as_tuple=False)   # Shape: [num_blocks, 2], columns: [batch_idx, position]

    # Extract batch indices and positions
    batch_indices = starts[:, 0]     # Batch indices
    start_positions = starts[:, 1]  # Start positions in [0, N+1]
    end_positions = ends[:, 1]      # End positions in [0, N+1]

    return batch_indices, start_positions, end_positions

def get_contiguous_blocks(sample_ids):
    # modality: [B, N], integer tensor
    # Compute where the value changes along the sequence dimension
    diff = sample_ids[:, 1:] != sample_ids[:, :-1]  # Shape: [B, N-1]
    diff = torch.nn.functional.pad(diff, (1, 0), mode='constant', value=True)  # Pad at the beginning

    # Find start positions where the value changes (including the first position)
    starts = diff.nonzero(as_tuple=False)  # Shape: [num_blocks, 2], columns: [batch_idx, position]

    # Compute end positions by shifting diff to the left and padding at the end
    diff_end = torch.nn.functional.pad(diff[:, 1:], (0, 1), mode='constant', value=True)  # Shape: [B, N]
    ends = diff_end.nonzero(as_tuple=False)  # Shape: [num_blocks, 2], columns: [batch_idx, position]

    # Extract batch indices and positions
    batch_indices = starts[:, 0]        # Batch indices
    start_positions = starts[:, 1]      # Start positions in [0, N)
    end_positions = ends[:, 1] + 1         # End positions in [0, N)

    valid_mask = sample_ids[batch_indices, start_positions] >= 0

    return batch_indices[valid_mask], start_positions[valid_mask], end_positions[valid_mask]

def get_contiguous_blocks_per_sample(modality, sample_ids):
    # modality: [B, N], integer tensor
    # Compute where the value changes along the sequence dimension
    # Detect changes in either modality or sample_ids
    diff_modality = modality[:, 1:] != modality[:, :-1]  # Shape: [B, N-1]
    diff_sample_ids = sample_ids[:, 1:] != sample_ids[:, :-1]  # Shape: [B, N-1]
    diff = diff_modality | diff_sample_ids  # Changes in either signal count as transitions
    diff = torch.nn.functional.pad(diff, (1, 0), mode='constant', value=True)  # Pad at the beginning

    # Find start positions where either value changes (including the first position)
    starts = diff.nonzero(as_tuple=False)  # Shape: [num_blocks, 2], columns: [batch_idx, position]

    # Compute end positions by shifting diff to the left and padding at the end
    diff_end = torch.nn.functional.pad(diff[:, 1:], (0, 1), mode='constant', value=True)  # Shape: [B, N]
    ends = diff_end.nonzero(as_tuple=False)  # Shape: [num_blocks, 2], columns: [batch_idx, position]

    # Extract batch indices and positions
    batch_indices = starts[:, 0]        # Batch indices
    start_positions = starts[:, 1]      # Start positions in [0, N)
    end_positions = ends[:, 1] + 1         # End positions in [0, N)

    valid_mask = sample_ids[batch_indices, start_positions] >= 0

    return batch_indices[valid_mask], start_positions[valid_mask], end_positions[valid_mask]


def tensor_dim_slice(tensor, dim, dim_slice):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (dim_slice, )]

def packshape(shape, dim : int = -1, mask : int = 0b00000001, dtype = torch.uint8, pack = True):
    dim = dim if dim >= 0 else dim + len(shape)
    bits, nibble = (8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
    # bits = torch.iinfo(dtype).bits # does not JIT compile
    assert nibble <= bits and bits % nibble == 0
    nibbles = bits // nibble
    shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)), ) + shape[1 + dim:]) if pack else (shape[:dim] + (shape[dim] * nibbles, ) + shape[1 + dim:])
    return shape, nibbles, nibble

def packbits(tensor, dim : int = -1, mask : int = 0b00000001, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape, nibbles, nibble = packshape(tensor.shape, dim = dim, mask = mask, dtype = dtype, pack = True)
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    assert tensor.shape[dim] % nibbles == 0
    shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
    torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim = 1 + dim, out = out)
    return out

def unpackbits(tensor, dim : int = -1, mask : int = 0b00000001, shape = None, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim = dim, mask = mask, dtype = tensor.dtype, pack = False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    assert shape[dim] % nibbles == 0
    shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
    return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)