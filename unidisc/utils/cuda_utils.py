from torchtnt.utils.distributed import all_gather_tensors, get_global_rank
import torch.distributed as dist
from decoupled_utils import is_torch_xla_available, use_dist, rprint
import torch

def _get_min_max_indices(input_list):
    min_index = -1
    max_index = -1
    min_value = float("inf")
    max_value = float("-inf")
    for rank, curr_value in enumerate(input_list):
        if curr_value < min_value:
            min_value = curr_value
            min_index = rank
        if curr_value > max_value:
            max_value = curr_value
            max_index = rank

    return min_index, max_index


def sync_times(device):
    if not use_dist():
        return
    # Use torch.cuda.Event to measure time across multiple nodes
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record the start time
    start_event.record()

    # Perform a synchronization to ensure all nodes start timing at roughly the same point
    dist.barrier()

    # Record the end time
    end_event.record()

    # Wait for the end event to be completed
    end_event.synchronize()

    # Calculate the elapsed time on this node
    elapsed_time = start_event.elapsed_time(end_event)

    # Gather elapsed times from all nodes
    elapsed_time_tensor = torch.tensor([elapsed_time], device=device)
    all_elapsed_times = all_gather_tensors(elapsed_time_tensor)

    # Convert tensor list to a list of times
    elapsed_times_list = [tensor.item() for tensor in all_elapsed_times]

    # Determine the fastest and slowest ranks
    fastest_rank, slowest_rank = _get_min_max_indices(elapsed_times_list)
    time_on_fastest_rank = elapsed_times_list[fastest_rank]
    time_on_slowest_rank = elapsed_times_list[slowest_rank]
    time_difference = time_on_slowest_rank - time_on_fastest_rank

    # Print the time difference
    rprint(
        f"Time difference between fastest rank ({fastest_rank}: {time_on_fastest_rank} ms) and slowest rank ({slowest_rank}: {time_on_slowest_rank} ms) is {time_difference} milliseconds."
    )