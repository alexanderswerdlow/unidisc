import time
from collections import deque
from typing import (TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional,
                    TypeVar, Union)

import torch
import wandb
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from decoupled_utils import is_main_process, rank_zero_fn, try_except
from typing_extensions import override

_THROUGHPUT_METRICS = Dict[str, Union[int, float]]


# The API design of this class follows `torchmetrics.Metric` but it doesn't need to be an actual Metric because there's
# no need for synchronization or reduction as it doesn't use Tensors at all.
class Throughput:
    """Computes throughput.

    +------------------------+-------------------------------------------------------------------------------------+
    | Key                    | Value                                                                               |
    +========================+=====================================================================================+
    | batches_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of batches |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of samples |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | items_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of items   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | flpps_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of flops   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/batches_per_sec | batches_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/samples_per_sec | samples_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/items_per_sec   | items_per_sec divided by world size. This may include padding depending on the data |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/flops_per_sec   | flops_per_sec divided by world size.                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/mfu             | device/flops_per_sec divided by world size.                                         |
    +--------------------------+-----------------------------------------------------------------------------------+
    | time                   | Total elapsed time                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | batches                | Total batches seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples                | Total samples seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | lengths                | Total items seen                                                                    |
    +--------------------------+-----------------------------------------------------------------------------------+

    Example::

        throughput = Throughput()
        t0 = time()
        for i in range(1000):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, samples=i)
            if i % 10 == 0:
                print(throughput.compute())

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
          takes a single ``available_flops`` value.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
          samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.

    """

    def __init__(
        self, world_size: int = 1, window_size: int = 100, separator: str = "/", available_flops=None
    ) -> None:
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size
        self.available_flops = available_flops

        # throughput is computed over a window of values. at least 2 is enforced since it looks at the difference
        # between the first and last elements
        assert window_size > 1
        # custom class instead of `deque(maxlen=)` because it's easy for users to mess up their timer/counters and log
        # values that do not increase monotonically. this class will raise an error if that happens.
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._batches: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._steps: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        if available_flops is not None:
            self._flops: Deque[int] = deque(maxlen=window_size)

    def update(
        self,
        *,
        time: float,
        steps: int,
        batches: int,
        samples: int,
        lengths: Optional[int] = None,
        flops: Optional[int] = None,
    ) -> None:
        """Update throughput metrics.

        Args:
            time: Total elapsed time in seconds. It should monotonically increase by the iteration time with each
                call.
            batches: Total batches seen per device. It should monotonically increase with each call.
            samples: Total samples seen per device. It should monotonically increase by the batch size with each call.
            lengths: Total length of the samples seen. It should monotonically increase by the lengths of a batch with
                each call.
            increment_step: Flag to increment the step counter.

        """
        self._time.append(time)
        self._steps.append(steps)
        if samples < batches:
            raise ValueError(f"Expected samples ({samples}) to be greater or equal than batches ({batches})")
        self._batches.append(batches)
        self._samples.append(samples)
        self._lengths.append(lengths)
        if self.available_flops is not None:
            self._flops.append(flops)

    def compute(self) -> _THROUGHPUT_METRICS:
        """Compute throughput metrics."""
        metrics = {
            "time": self._time[-1],
            "steps": self._steps[-1],
            "batches": self._batches[-1] * self.world_size,
            "samples": self._samples[-1] * self.world_size,
            
        }
        if self._lengths:
            metrics["lengths"] = self._lengths[-1]

        # a different but valid design choice would be to still compute all these metrics even if the window of values
        # has not been filled
        if len(self._time) == self._time.maxlen:
            elapsed_time = self._time[-1] - self._time[0]
            elapsed_batches = self._batches[-1] - self._batches[0]
            elapsed_samples = self._samples[-1] - self._samples[0]
            elapsed_steps = self._steps[-1] - self._steps[0]
            # we are safe from ZeroDivisionError thanks to `_MonotonicWindow`
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            dev_steps_per_sec = elapsed_steps / elapsed_time
            metrics.update({
                f"device{self.separator}batches_per_sec": dev_batches_per_sec,
                f"device{self.separator}samples_per_sec": dev_samples_per_sec,
                f"device{self.separator}steps_per_sec": dev_steps_per_sec,
            })
            metrics.update({
                "batches_per_sec": dev_batches_per_sec * self.world_size,
                "samples_per_sec": dev_samples_per_sec * self.world_size,
                "steps_per_sec": dev_steps_per_sec * self.world_size,
            })

            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = self._lengths[-1] - self._lengths[0]
                dev_items_per_sec = elapsed_lengths / elapsed_time
                metrics[f"device{self.separator}items_per_sec"] = dev_items_per_sec
                metrics["items_per_sec"] = dev_items_per_sec * self.world_size

        if self.available_flops is not None:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            dev_flops_per_sec = (elapsed_flops / elapsed_time) if elapsed_time > 0 else 0
            flops_per_sec = dev_flops_per_sec * self.world_size
            metrics["flops_per_sec"] = flops_per_sec
            metrics[f"device{self.separator}flops_per_sec"] = dev_flops_per_sec
            metrics[f"device{self.separator}mfu"] = dev_flops_per_sec / self.available_flops

        return metrics

    def reset(self) -> None:
        self._time.clear()
        self._batches.clear()
        self._samples.clear()
        self._lengths.clear()
        self._steps.clear()



T = TypeVar("T", bound=float)


class _MonotonicWindow(List[T]):
    """Custom fixed size list that only supports right-append and ensures that all values increase monotonically."""

    def __init__(self, maxlen: int) -> None:
        super().__init__()
        self.maxlen = maxlen

    @property
    def last(self) -> Optional[T]:
        if len(self) > 0:
            return self[-1]
        return None

    @override
    def append(self, x: T) -> None:
        last = self.last
        if last is not None and last >= x:
            pass
            # print(f"Expected the value to increase, last: {last}, current: {x}")
        list.append(self, x)
        # truncate excess
        if len(self) > self.maxlen:
            del self[0]

    @override
    def __setitem__(self, key: Any, value: Any) -> None:
        # assigning is not implemented since we don't use it. it could be by checking all previous values
        raise NotImplementedError("__setitem__ is not supported")


class ThroughputMonitor(Callback):
    def __init__(
        self, batch_size_fn: Callable[[Any], int], length_fn: Optional[Callable[[Any], int]] = None, world_size: int = 1, log_every_n_steps=10, flops_per_sample=None, device = None, dtype = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn
        self._throughputs: dict = {}
        self._t0s: dict = {}
        self.inference_max_batch_size = 0
        self.stage = 'train'
        self.log_every_n_steps = log_every_n_steps
        self.flops_per_sample = flops_per_sample
        self.last_samples = 0

        if flops_per_sample is not None:
            self.available_flops = get_available_flops(device, dtype)

        throughput = Throughput(world_size=world_size, available_flops=self.available_flops, **self.kwargs)
        self._throughputs[self.stage] = throughput

        self._throughputs[self.stage].reset()
        self._t0s[self.stage] = time.perf_counter()

    @rank_zero_fn
    @try_except(write_error_to_file=True)
    @torch.inference_mode()
    def on_train_step_end(
        self, state: State, unit: TTrainUnit
    ) -> None:
        
        global_step = unit.global_step
        if global_step % self.log_every_n_steps != 0:
            return
        
        if is_torch_cuda_available():
            torch.cuda.synchronize()

        stage = self.stage
        throughput = self._throughputs[stage]
        elapsed = time.perf_counter() - self._t0s[stage]
        tokens_per_sample = unit.num_tokens_per_sample

        if self.batch_size_fn is not None:
            batch_size = self.batch_size_fn(state.batch) * unit.gradient_accumulation_steps
        else:
            batch_size = unit.step_batch_size

        if self.length_fn is not None:
            batch = state.batch
            _length = self.length_fn(batch)
        else:
            _length = tokens_per_sample * batch_size

        if self.available_flops is not None:
            _flops = self.flops_per_sample * ((global_step * batch_size) - self.last_samples)
            self.last_samples = global_step * batch_size
        else:
            _flops = None
        
        throughput.update(
            time=elapsed,
            steps=global_step,
            batches=global_step * unit.gradient_accumulation_steps,
            samples=global_step * batch_size,
            lengths=global_step * _length,
            flops=_flops
        )

        throughput = self._throughputs[stage]
        metrics = throughput.compute()
        metrics = {f"{stage}_metrics/{k}": v for k, v in metrics.items()}
        if is_main_process():
            wandb.log(dict(**metrics, **{"trainer/global_step": global_step}))


_CUDA_FLOPS: Dict[str, Dict[Union[str, torch.dtype], float]] = {
    # Hopper
    # source: https://resources.nvidia.com/en-us-tensor-core
    "h100 nvl": {
        torch.float64: 67e12,
        torch.float32: 133.8e12,
        "tfloat32": 989.4e12,
        torch.bfloat16: 1978.8e12,
        torch.float16: 1978.8e12,
        torch.int8: 3957.8e12,
    },
    "h100 sxm": {
        torch.float64: 33.5e12,
        torch.float32: 66.9e12,
        "tfloat32": 494.7e12,
        torch.bfloat16: 989.4e12,
        torch.float16: 989.4e12,
        torch.int8: 1978.9e12,
    },
    "h100 pcie": {
        torch.float64: 25.6e12,
        torch.float32: 51.2e12,
        "tfloat32": 378e12,
        torch.bfloat16: 756e12,
        torch.float16: 756e12,
        torch.int8: 1513e12,
    },
    # Ada
    # source: https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    "rtx 4090": {
        torch.float32: 82.6e12,
        "tfloat32": 82.6e12,
        torch.bfloat16: 82.6e12,
        torch.float16: 82.6e12,
        torch.int8: 660.6e12,
        "int4": 1321.2e12,
    },
    "rtx 4080": {
        torch.float32: 48.7e12,
        "tfloat32": 48.7e12,
        torch.bfloat16: 48.7e12,
        torch.float16: 48.7e12,
        torch.int8: 389.9e12,
        "int4": 779.8e12,
    },
    "l4": {
        torch.float32: 30.3e12,
        "tfloat32": 60e12,
        torch.bfloat16: 121e12,
        torch.float16: 121e12,
        torch.int8: 242e12,
        "int4": 484e12,
    },
    "l40": {
        torch.float32: 90.5e12,
        "tfloat32": 90.5e12,
        torch.bfloat16: 181e12,
        torch.float16: 181e12,
        torch.int8: 362e12,
        "int4": 724e12,
    },
    # Ampere
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        torch.float64: 9.7e12,
        torch.float32: 19.5e12,
        "tfloat32": 156e12,
        torch.bfloat16: 312e12,
        torch.float16: 312e12,
        torch.int8: 624e12,
    },
    "a6000": {
        torch.float32: 38.7e12,
        "tfloat32": 77.4e12,
        torch.bfloat16: 38.7e12,
        torch.float16: 38.7e12,
        torch.int8: 309.7e12,
        "int4": 619.3e12,
    },
    "6000ada": {
        torch.float32: 91.1e12,
        "tfloat32": 182.1e12,
        torch.bfloat16: 91.1e12,
        torch.float16: 91.1e12,
        torch.int8: 728.5e12,
        "int4": 1457.0e12,
    },
    "a5000": {
        torch.float32: 27.8e12,
        torch.bfloat16: 27.8e12,
        torch.float16: 27.8e12,
    },
    "a5500": {
        torch.float32: 34.1e12,
        torch.bfloat16: 34.1e12,
        torch.float16: 34.1e12,
    },
    "a40": {
        torch.float32: 37.4e12,
        "tfloat32": 74.8e12,
        torch.bfloat16: 37.4e12,
        torch.float16: 37.4e12,
        torch.int8: 299.3e12,
        "int4": 598.7e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        torch.float32: 31.2e12,
        "tfloat32": 62.5e12,
        torch.bfloat16: 125e12,
        torch.float16: 125e12,
        torch.int8: 250e12,
        "int4": 500e12,
    },
    "rtx 3090 ti": {
        torch.float32: 40e12,
        "tfloat32": 40e12,
        torch.bfloat16: 40e12,
        torch.float16: 40e12,
        torch.int8: 320e12,
        "int4": 640e12,
    },
    "rtx 3090": {
        torch.float32: 35.6e12,
        "tfloat32": 35.6e12,
        torch.bfloat16: 35.6e12,
        torch.float16: 35.6e12,
        torch.int8: 284e12,
        "int4": 568e12,
    },
    "rtx 3080 ti": {
        torch.float32: 34.1e12,
        "tfloat32": 34.1e12,
        torch.bfloat16: 34.1e12,
        torch.float16: 34.1e12,
        torch.int8: 272.8e12,
        "int4": 546.6e12,
    },
    "rtx 3080": {
        torch.float32: 29.8e12,
        "tfloat32": 29.8e12,
        torch.bfloat16: 29.8e12,
        torch.float16: 29.8e12,
        torch.int8: 238e12,
        "int4": 476e12,
    },
    "rtx 3070": {
        torch.float32: 20.3e12,
        "tfloat32": 20.3e12,
        torch.bfloat16: 20.3e12,
        torch.float16: 20.3e12,
        torch.int8: 162.6e12,
        "int4": 325.2e12,
    },
    # Turing
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        torch.float32: 8.1e12,
        torch.float16: 65e12,
        torch.int8: 130e12,
        "int4": 260e12,
    },
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {
        torch.float32: 11.2e12,
        torch.float16: 89.2e12,
    },
    "rtx 2080 super": {
        torch.float32: 11.2e12,
        torch.float16: 22.3e12,
        torch.int8: 178.4e12,
        "int4": 356.8e12,
    },
    "rtx 2080 ti": {
        torch.float32: 14.2e12,
        torch.float16: 28.5e12,
        torch.int8: 227.7e12,
        "int4": 455.4e12,
    },
    "rtx 2080": {
        torch.float32: 10.6e12,
        torch.float16: 21.2e12,
        torch.int8: 169.6e12,
        "int4": 339.1e12,
    },
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "rtx 2070 super": {
        torch.float32: 9.1e12,
        torch.float16: 18.1e12,
        torch.int8: 145e12,
        "int4": 290e12,
    },
    "titan rtx": {
        torch.float32: 16.3e12,
        torch.float16: 32.6e12,
        torch.int8: 261e12,
        "int4": 522e12,
    },
    # Volta
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100 sxm": {
        torch.float64: 7.8e12,
        torch.float32: 15.7e12,
        torch.float16: 125e12,
    },
    "v100 pcie": {
        torch.float64: 7e12,
        torch.float32: 14e12,
        torch.float16: 112e12,
    },
    "v100s pcie": {
        torch.float64: 8.2e12,
        torch.float32: 16.4e12,
        torch.float16: 130e12,
    },
    "l40s": {
        torch.float32: 91.6e12,
        torch.bfloat16: 362.05e12,
        torch.float16: 362.05e12,
    },
}

_TPU_FLOPS = {
    # flop count for each TPU generation is the same for all precisions
    # since bfloat16 precision is always used for performing matrix operations
    # for more info: https://cloud.google.com/tpu/docs/bfloat16#choosing_bfloat16
    # source: https://arxiv.org/pdf/1907.10701.pdf
    "v2": 45e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3
    "v3": 123e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v4
    "v4": 275e12,
    # source: https://cloud.google.com/tpu/docs/v5e-training
    "v5litepod": 197e12,
}

from decoupled_utils import is_torch_cuda_available, rprint, synchronize_device

def _is_ampere_or_later(device: Optional[torch.device] = None) -> bool:
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8  # Ampere and later leverage tensor cores, where this setting becomes useful


def get_available_flops(device: torch.device, dtype: Union[torch.dtype, str]) -> Optional[int]:
    """Returns the available theoretical FLOPs.

    This is an optimistic upper limit that could only be achievable if only thick matmuls were run in a benchmark
    environment.

    """
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        chip = device_name.lower()
        if "h100" in chip:
            if "hbm3" in chip:
                chip = "h100 sxm"
            elif "nvl" in chip:
                chip = "h100 nvl"
            elif "pcie" in chip or "hbm2e" in chip:
                chip = "h100 pcie"
        elif "l40s" in chip:
            chip = "l40s"
        elif "l4" in chip:
            chip = "l40" if "tesla" in chip else "l4"
        elif "geforce rtx" in chip:
            number = chip.split(" ")[3]
            extra = ""
            if "super" in chip:
                extra = " super"
            elif "ti" in chip:
                extra = " ti"
            chip = f"rtx {number}{extra}"
        elif "a6000" in chip:
            chip = "a6000"
        elif "6000 ada" in chip:
            chip = "6000ada"
        elif "a5500" in chip:
            chip = "a5500"
        elif "a5000" in chip:
            chip = "a5000"
        elif "a100" in chip:
            chip = "a100"
        elif "a40" in chip:
            chip = "a40"
        elif "a10g" in chip:
            chip = "a10g"
        elif "t4" in chip:
            chip = "t4"
        elif "quadro rtx 5000" in chip:
            chip = "quadro rtx 5000"
        elif "titan rtx" in chip:
            chip = "titan rtx"
        elif "v100-sxm" in chip:
            chip = "v100 sxm"
        elif "v100-pcie" in chip:
            chip = "v100 pcie"
        elif "v100s-pcie" in chip:
            chip = "v100s pcie"
        else:
            # the flops list is not exhaustive, return with a warning
            rprint(f"FLOPs not found for {device_name!r}")
            return 0
        if chip not in _CUDA_FLOPS:
            # parsing is implemented but we don't have the stats
            rprint(f"FLOPs not found for {device_name!r}, chip is {chip!r}")
            return 0
        dtype_to_flops = _CUDA_FLOPS[chip]
        if dtype is torch.float32:
            if _is_ampere_or_later() and torch.get_float32_matmul_precision() != "highest":
                dtype = "tfloat32"
        if dtype not in dtype_to_flops:
            # for example, T4 doesn't support bfloat16. it might also be that we are missing this dtype from the list
            rprint(f"{device_name!r} does not support {dtype}")
            return 0
        return int(dtype_to_flops[dtype])
    
    if device.type == "xla":
        from torch_xla._internal import tpu
        tpu_env = tpu.get_tpu_env()
        # not all TPU generations define the "TYPE" envar. example: TYPE="V4", ACCELERATOR_TYPE="v4-8"
        device_name = tpu_env.get("TYPE") or tpu_env["ACCELERATOR_TYPE"].split("-")[0]
        chip = device_name.lower()
        assert isinstance(device_name, str)
        if chip not in _TPU_FLOPS:
            rprint(f"FLOPs not found for TPU {device_name!r} with {dtype}")
            return 0
        return int(_TPU_FLOPS[chip])


# https://github.com/karpathy/llm.c/blob/master/llmc/mfu.h
class PerfData:
    def __init__(self, TF_32, BF_16_32, FP_16_32, FP_16_16, FP_8_32, FP_8_16, CLOCK, CORES):
        self.TF_32 = TF_32
        self.BF_16_32 = BF_16_32
        self.FP_16_32 = FP_16_32
        self.FP_16_16 = FP_16_16
        self.FP_8_32 = FP_8_32
        self.FP_8_16 = FP_8_16
        self.CLOCK = CLOCK
        self.CORES = CORES

class GPUEntry:
    def __init__(self, name, perf_data, new_cores, new_mhz):
        self.name = name
        self.perf_data = perf_data
        self.new_cores = new_cores
        self.new_mhz = new_mhz

VOLTA = PerfData(125.0, -1.0, 125.0, -1.0, -1.0, -1.0, 1530.0, 640.0)
AMPERE_DATACENTER = PerfData(156.0, 312.0, 312.0, 312.0, -1.0, -1.0, 1410.0, 432.0)
AMPERE_CONSUMER = PerfData(40.0, 80.0, 80.0, 160.0, -1.0, -1.0, 1860.0, 336.0)
HOPPER = PerfData(378.0, 756.0, 756.0, 756.0, 1513.0, 1513.0, 1620.0, 456.0)
ADA = PerfData(82.6, 165.2, 165.2, 330.3, 330.3, 660.6, 2520.0, 512.0)

gpu_db = [
    GPUEntry("Tesla V100-SXM2-16GB", VOLTA, 640, 1530),
    GPUEntry("Tesla V100-PCIE-32GB", VOLTA, 640, 1530),
    GPUEntry("NVIDIA A100-PCIE-40GB", AMPERE_DATACENTER, 432, 1410),
    GPUEntry("NVIDIA A100-PCIE-80GB", AMPERE_DATACENTER, 432, 1410),
    GPUEntry("NVIDIA A100-SXM4-40GB", AMPERE_DATACENTER, 432, 1410),
    GPUEntry("NVIDIA A100-SXM4-80GB", AMPERE_DATACENTER, 432, 1410),
    GPUEntry("NVIDIA RTX A2000", AMPERE_CONSUMER, 104, 1200),
    GPUEntry("NVIDIA RTX A4000", AMPERE_CONSUMER, 192, 1560),
    GPUEntry("NVIDIA RTX A4500", AMPERE_CONSUMER, 224, 1650),
    GPUEntry("NVIDIA RTX A5000", AMPERE_CONSUMER, 256, 1695),
    GPUEntry("NVIDIA RTX A5500", AMPERE_CONSUMER, 320, 1770),
    GPUEntry("NVIDIA RTX A6000", AMPERE_CONSUMER, 336, 1800),
    GPUEntry("NVIDIA GeForce RTX 3090 Ti", AMPERE_CONSUMER, 336, 1860),
    GPUEntry("NVIDIA GeForce RTX 3090", AMPERE_CONSUMER, 328, 1695),
    GPUEntry("NVIDIA GeForce RTX 3080 Ti", AMPERE_CONSUMER, 320, 1665),
    GPUEntry("NVIDIA GeForce RTX 3080", AMPERE_CONSUMER, 272, 1710),
    GPUEntry("NVIDIA GeForce RTX 3070 Ti", AMPERE_CONSUMER, 192, 1770),
    GPUEntry("NVIDIA GeForce RTX 3070", AMPERE_CONSUMER, 184, 1725),
    GPUEntry("NVIDIA GeForce RTX 3060 Ti", AMPERE_CONSUMER, 152, 1665),
    GPUEntry("NVIDIA GeForce RTX 3060", AMPERE_CONSUMER, 112, 1777),
    GPUEntry("NVIDIA RTX A2000 ADA", ADA, 88, 2130),
    GPUEntry("NVIDIA RTX A4000 ADA", ADA, 192, 2175),
    GPUEntry("NVIDIA RTX A4500 ADA", ADA, 224, 2580),
    GPUEntry("NVIDIA RTX A5000 ADA", ADA, 400, 2550),
    GPUEntry("NVIDIA RTX A5880 ADA", ADA, 440, 2460),
    GPUEntry("NVIDIA RTX A6000 ADA", ADA, 568, 2505),
    GPUEntry("NVIDIA GeForce RTX 4090", ADA, 512, 2520),
    GPUEntry("NVIDIA GeForce RTX 4080 SUPER", ADA, 320, 2550),
    GPUEntry("NVIDIA GeForce RTX 4080", ADA, 304, 2505),
    GPUEntry("NVIDIA GeForce RTX 4070 Ti SUPER", ADA, 264, 2610),
    GPUEntry("NVIDIA GeForce RTX 4070 Ti", ADA, 240, 2610),
    GPUEntry("NVIDIA GeForce RTX 4070 SUPER", ADA, 224, 2475),
    GPUEntry("NVIDIA GeForce RTX 4070", ADA, 184, 2475),
    GPUEntry("NVIDIA GeForce RTX 4060 Ti", ADA, 136, 2535),
    GPUEntry("NVIDIA GeForce RTX 4060", ADA, 96, 2460),
    GPUEntry("NVIDIA H100 PCIe", HOPPER, 456, 1620),
    GPUEntry("NVIDIA H100 80GB HBM3", HOPPER, 528, 1830)
]

MFUH_PRECISION_FP32 = 0
MFUH_PRECISION_FP16 = 1
MFUH_PRECISION_BF16 = 2

def get_flops_promised(device, precision_mode):
    """
    This function is used to estimate the Model Flops Utilization (MFU)
    basically we have to figure out how many flops the GPU can do per second.
    Note that this is not a simple endeavor and may well go wrong! The details are tricky.
    The returned value is in units of 1e12.
    """
    if precision_mode not in [MFUH_PRECISION_FP32, MFUH_PRECISION_FP16, MFUH_PRECISION_BF16]:
        print(f"Invalid precision mode: {precision_mode}")
        return -1.0

    for entry in gpu_db:
        if entry.name == device:
            perf_data = entry.perf_data

            value = -1.0
            if precision_mode == MFUH_PRECISION_BF16:
                value = perf_data.BF_16_32
            if precision_mode == MFUH_PRECISION_FP32:
                value = perf_data.TF_32
            if precision_mode == MFUH_PRECISION_FP16:
                value = perf_data.FP_16_32

            if value < 0.0:
                print(f"No data for GPU {device} and precision mode {precision_mode}")
                return -1.0

            new_cores = entry.new_cores
            new_mhz = entry.new_mhz
            adjusted = value * (new_cores / perf_data.CORES) * (new_mhz / perf_data.CLOCK)
            return adjusted

    return -1.0