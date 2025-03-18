# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import functools
import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from typing import (Any, Callable, ClassVar, Dict, Generator, Hashable,
                    Iterable, Iterator, List, Optional, Sequence, Tuple, Union)

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module, ModuleDict
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import (_flatten, _flatten_dict,
                                         _squeeze_if_scalar, allclose,
                                         dim_zero_cat, dim_zero_max,
                                         dim_zero_mean, dim_zero_min,
                                         dim_zero_sum)
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.imports import (_MATPLOTLIB_AVAILABLE,
                                            _TORCH_GREATER_EQUAL_2_1)
from torchmetrics.utilities.plot import (_AX_TYPE, _PLOT_OUT_TYPE,
                                         plot_single_or_multi_val)
from torchmetrics.utilities.prints import rank_zero_warn
from typing_extensions import Literal
from decoupled_utils import is_torch_xla_available

def jit_distributed_available() -> bool:
    """Determine if distributed mode is initialized."""
    return not is_torch_xla_available()

class Metric(Module, ABC):
    """Base class for all metrics present in the Metrics API.

    This class is inherited by all metrics and implements the following functionality:
    1. Handles the transfer of metric states to correct device
    2. Handles the synchronization of metric states across processes

    The three core methods of the base class are
    * ``add_state()``
    * ``forward()``
    * ``reset()``

    which should almost never be overwritten by child classes. Instead, the following methods should be overwritten
    * ``update()``
    * ``compute()``


    Args:
        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

            - compute_on_cpu: If metric state should be stored on CPU during computations. Only works for list states.
            - dist_sync_on_step: If metric state should synchronize on ``forward()``. Default is ``False``
            - process_group: The process group on which the synchronization is called. Default is the world.
            - dist_sync_fn: Function that performs the allgather option on the metric state. Default is an custom
              implementation that calls ``torch.distributed.all_gather`` internally.
            - distributed_available_fn: Function that checks if the distributed backend is available. Defaults to a
              check of ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
            - sync_on_compute: If metric state should synchronize when ``compute`` is called. Default is ``True``
            - compute_with_cache: If results from ``compute`` should be cached. Default is ``True``

    """

    __jit_ignored_attributes__: ClassVar[List[str]] = ["device"]
    __jit_unused_properties__: ClassVar[List[str]] = [
        "is_differentiable",
        "higher_is_better",
        "plot_lower_bound",
        "plot_upper_bound",
        "plot_legend_name",
        "metric_state",
        "_update_called",
    ]
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None
    full_state_update: Optional[bool] = None

    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None
    plot_legend_name: Optional[str] = None

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"torchmetrics.metric.{self.__class__.__name__}")
        # magic patch for `RuntimeError: DataLoader worker (pid(s) 104) exited unexpectedly`
        self._TORCH_GREATER_EQUAL_2_1 = bool(_TORCH_GREATER_EQUAL_2_1)
        self._device = torch.device("cpu")
        self._dtype = torch.get_default_dtype()

        self.compute_on_cpu = kwargs.pop("compute_on_cpu", False)
        if not isinstance(self.compute_on_cpu, bool):
            raise ValueError(
                f"Expected keyword argument `compute_on_cpu` to be an `bool` but got {self.compute_on_cpu}"
            )

        self.dist_sync_on_step = kwargs.pop("dist_sync_on_step", False)
        if not isinstance(self.dist_sync_on_step, bool):
            raise ValueError(
                f"Expected keyword argument `dist_sync_on_step` to be an `bool` but got {self.dist_sync_on_step}"
            )

        self.process_group = kwargs.pop("process_group", None)

        self.dist_sync_fn = kwargs.pop("dist_sync_fn", None)
        if self.dist_sync_fn is not None and not callable(self.dist_sync_fn):
            raise ValueError(
                f"Expected keyword argument `dist_sync_fn` to be an callable function but got {self.dist_sync_fn}"
            )

        self.distributed_available_fn = kwargs.pop("distributed_available_fn", None) or jit_distributed_available

        self.sync_on_compute = kwargs.pop("sync_on_compute", True)
        if not isinstance(self.sync_on_compute, bool):
            raise ValueError(
                f"Expected keyword argument `sync_on_compute` to be a `bool` but got {self.sync_on_compute}"
            )
        self.compute_with_cache = kwargs.pop("compute_with_cache", True)
        if not isinstance(self.compute_with_cache, bool):
            raise ValueError(
                f"Expected keyword argument `compute_with_cache` to be a `bool` but got {self.compute_with_cache}"
            )

        if kwargs:
            kwargs_ = [f"`{a}`" for a in sorted(kwargs)]
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs_)}")

        # initialize
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore[method-assign]
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore[method-assign]
        self._computed = None
        self._forward_cache = None
        self._update_count = 0
        self._to_sync = self.sync_on_compute
        self._should_unsync = True
        self._enable_grad = False
        self._dtype_convert = False

        # initialize state
        self._defaults: Dict[str, Union[List, Tensor]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        # state management
        self._is_synced = False
        self._cache: Optional[Dict[str, Union[List[Tensor], Tensor]]] = None

    @property
    def _update_called(self) -> bool:
        rank_zero_warn(
            "This property will be removed in 2.0.0. Use `Metric.updated_called` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.update_called

    @property
    def update_called(self) -> bool:
        """Returns `True` if `update` or `forward` has been called initialization or last `reset`."""
        return self._update_count > 0

    @property
    def update_count(self) -> int:
        """Get the number of times `update` and/or `forward` has been called since initialization or last `reset`."""
        return self._update_count

    @property
    def metric_state(self) -> Dict[str, Union[List[Tensor], Tensor]]:
        """Get the current state of the metric."""
        return {attr: getattr(self, attr) for attr in self._defaults}

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """Add metric state variable. Only used by subclasses.

        Metric state variables are either `:class:`~torch.Tensor` or an empty list, which can be appended to by the
        metric. Each state variable must have a unique name associated with it. State variables are accessible as
        attributes of the metric i.e, if ``name`` is ``"my_state"`` then its value can be accessed from an instance
        ``metric`` as ``metric.my_state``. Metric states behave like buffers and parameters of :class:`~torch.nn.Module`
        as they are also updated when ``.to()`` is called. Unlike parameters and buffers, metric states are not by
        default saved in the modules :attr:`~torch.nn.Module.state_dict`.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a :class:`~torch.Tensor` or an empty list.
                The state will be reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state across multiple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will use ``torch.sum``,
                ``torch.mean``, ``torch.cat``, ``torch.min`` and ``torch.max``` respectively, each with argument
                ``dim=0``. Note that the ``"cat"`` reduction only makes sense if the state is a list, and not
                a tensor. The user can also pass a custom function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.
                Default is ``False``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is :class:`~torch.Tensor`, the synced value will be a stacked :class:`~torch.Tensor`
              across the process dimension if the metric state was a :class:`~torch.Tensor`. The original
              :class:`~torch.Tensor` metric state retains dimension and hence the synchronized output will be of shape
              ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        Note:
            The values inserted into a list state are deleted whenever :meth:`~Metric.reset` is called. This allows
            device memory to be automatically reallocated, but may produce unexpected effects when referencing list
            states. To retain such values after :meth:`~Metric.reset` is called, you must first copy them to another
            object.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.
            ValueError:
                If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``"min"``,
                ``"max"`` or ``None``.

        """
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', 'min', 'max', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Aggregate and evaluate batch input directly.

        Serves the dual purpose of both computing the metric on the current batch of inputs but also add the batch
        statistics to the overall accumululating metric state. Input arguments are the exact same as corresponding
        ``update`` method. The returned output is the exact same as the output of ``compute``.

        Args:
            args: Any arguments as required by the metric ``update`` method.
            kwargs: Any keyword arguments as required by the metric ``update`` method.

        Returns:
            The output of the ``compute`` method evaluated on the current batch.

        Raises:
            TorchMetricsUserError:
                If the metric is already synced and ``forward`` is called again.

        """
        # check if states are already synced
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_cache = self._forward_full_state_update(*args, **kwargs)
        else:
            self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)

        return self._forward_cache

    def _forward_full_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward computation using two calls to `update`.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        This is the most safe method to use for any metric but also the slower version of the two forward
        implementations.

        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = self._copy_state_dict()

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward computation using single call to `update`.

        This can be done when the global metric state is a sinple reduction of batch states. This can be unsafe for
        certain metric cases but is also the fastest way to both accumulate globally and compute locally.

        """
        # store global state and reset to default
        global_state = self._copy_state_dict()
        _update_count = self._update_count
        self.reset()

        # local synchronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # reduce batch and global state
        self._update_count = _update_count + 1
        with torch.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _reduce_states(self, incoming_state: Dict[str, Any]) -> None:
        """Add an incoming metric state to the current state of the metric.

        Args:
            incoming_state: a dict containing a metric state similar metric itself

        """
        for attr in self._defaults:
            local_state = getattr(self, attr)
            global_state = incoming_state[attr]
            reduce_fn = self._reductions[attr]
            if reduce_fn == dim_zero_sum:
                reduced = global_state + local_state
            elif reduce_fn == dim_zero_mean:
                reduced = ((self._update_count - 1) * global_state + local_state).float() / self._update_count
            elif reduce_fn == dim_zero_max:
                reduced = torch.max(global_state, local_state)
            elif reduce_fn == dim_zero_min:
                reduced = torch.min(global_state, local_state)
            elif reduce_fn == dim_zero_cat:
                if isinstance(global_state, Tensor):
                    reduced = torch.cat([global_state, local_state])
                else:
                    reduced = global_state + local_state
            elif reduce_fn is None and isinstance(global_state, Tensor):
                reduced = torch.stack([global_state, local_state])
            elif reduce_fn is None and isinstance(global_state, list):
                reduced = _flatten([global_state, local_state])
            elif reduce_fn and callable(reduce_fn):
                reduced = reduce_fn(torch.stack([global_state, local_state]))
            else:
                raise TypeError(f"Unsupported reduce_fn: {reduce_fn}")
            setattr(self, attr, reduced)

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

            # cornor case in distributed settings where a rank have not received any data, create empty to concatenate
            if (
                self._TORCH_GREATER_EQUAL_2_1
                and reduction_fn == dim_zero_cat
                and isinstance(input_dict[attr], list)
                and len(input_dict[attr]) == 0
            ):
                input_dict[attr] = [torch.tensor([], device=self.device, dtype=self.dtype)]

        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)

            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            with torch.set_grad_enabled(self._enable_grad):
                try:
                    update(*args, **kwargs)
                except RuntimeError as err:
                    if "Expected all tensors to be on" in str(err):
                        raise RuntimeError(
                            "Encountered different devices in metric calculation (see stacktrace for details)."
                            " This could be due to the metric class not being on the same device as input."
                            f" Instead of `metric={self.__class__.__name__}(...)` try to do"
                            f" `metric={self.__class__.__name__}(...).to(device)` where"
                            " device corresponds to the device of the input."
                        ) from err
                    raise err

            if self.compute_on_cpu:
                self._move_list_states_to_cpu()

        return wrapped_func

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults:
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def sync(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> None:
        """Sync function for manually controlling when metrics states should be synced across processes.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            distributed_available: Function to determine if we are running inside a distributed setting

        Raises:
            TorchMetricsUserError:
                If the metric is already synced and ``sync`` is called again.

        """
        if self._is_synced and should_sync:
            raise TorchMetricsUserError("The Metric has already been synced.")

        if distributed_available is None and self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        is_distributed = distributed_available() if callable(distributed_available) else None

        if not should_sync or not is_distributed:
            return

        if dist_sync_fn is None:
            dist_sync_fn = gather_all_tensors

        # cache prior to syncing
        self._cache = self._copy_state_dict()

        # sync
        self._sync_dist(dist_sync_fn, process_group=process_group)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        """Unsync function for manually controlling when metrics states should be reverted back to their local states.

        Args:
            should_unsync: Whether to perform unsync

        """
        if not should_unsync:
            return

        if not self._is_synced:
            raise TorchMetricsUserError("The Metric has already been un-synced.")

        if self._cache is None:
            raise TorchMetricsUserError("The internal cache should exist to unsync the Metric.")

        # if we synced, restore to cache so that we can continue to accumulate un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> Generator:
        """Context manager to synchronize states.

        This context manager is used in distributed setting and makes sure that the local cache states are restored
        after yielding the synchronized state.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting

        """
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if not self.update_called:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            # compute relies on the sync context manager to gather the states across processes and apply reduction
            # if synchronization happened, the current rank accumulated states will be restored to keep
            # accumulation going if ``should_unsync=True``,
            with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
            ):
                value = _squeeze_if_scalar(compute(*args, **kwargs))
                # clone tensor to avoid in-place operations after compute, altering already computed results
                value = apply_to_collection(value, Tensor, lambda x: x.clone())

            if self.compute_with_cache:
                self._computed = value

            return value

        return wrapped_func

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value.

        This method will automatically synchronize state variables when running in distributed backend.

        """

    def plot(self, *_: Any, **__: Any) -> Any:
        """Override this method plot the metric value."""
        raise NotImplementedError

    def _plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor], Dict[str, Tensor], Sequence[Dict[str, Tensor]]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        """
        val = val if val is not None else self.compute()
        fig, ax = plot_single_or_multi_val(
            val,
            ax=ax,
            higher_is_better=self.higher_is_better,
            name=self.__class__.__name__,
            lower_bound=self.plot_lower_bound,
            upper_bound=self.plot_upper_bound,
            legend_name=self.plot_legend_name,
        )
        return fig, ax

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                getattr(self, attr).clear()  # delete/free list items

        # reset internal states
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Make a copy of the metric."""
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the current state, including all metric states, for the metric.

        Used for loading and saving a metric.

        """
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute", "_update_signature"]}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the metric, based on a input state.

        Used for loading and saving a metric.

        """
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore[method-assign]
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore[method-assign]

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwrite default method to prevent specific attributes from being set by user."""
        if name in (
            "higher_is_better",
            "is_differentiable",
            "full_state_update",
            "plot_lower_bound",
            "plot_upper_bound",
            "plot_legend_name",
        ):
            raise RuntimeError(f"Can't change const `{name}`.")
        super().__setattr__(name, value)

    @property
    def device(self) -> "torch.device":
        """Return the device of the metric."""
        return self._device

    @property
    def dtype(self) -> "torch.dtype":
        """Return the default dtype of the metric."""
        return self._dtype

    def type(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.

        """
        return self

    def float(self) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.

        """
        return self

    def double(self) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.

        """
        return self

    def half(self) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.

        """
        return self

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as string or dtype object

        """
        self._dtype_convert = True
        out = super().type(dst_type)
        out._dtype_convert = False
        return out

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        """Overwrite `_apply` function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is garded and will only happen through the special `set_dtype` method.

        Args:
            fn: the function to apply
            exclude_state: list of state variables to exclude from applying the function, that then needs to be handled
                by the metric class itself.

        """
        this = super()._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if key in exclude_state:
                continue

            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    f"Expected metric state to be either a Tensor or a list of Tensor, but encountered {current_val}"
                )

        # make sure to update the device attribute
        # if the dummy tensor moves device by fn function we should also update the attribute
        _dummy_tensor = fn(torch.zeros(1, device=self.device))
        self._device = _dummy_tensor.device
        self._dtype = _dummy_tensor.dtype

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this

    def persistent(self, mode: bool = False) -> None:
        """Change post-init if metric states should be saved to its state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(  # type: ignore[override]  # todo
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """Get the current state of metric as an dictionary.

        Args:
            destination: Optional dictionary, that if provided, the state of module will be updated into the dict and
                the same object is returned. Otherwise, an ``OrderedDict`` will be created and returned.
            prefix: optional string, a prefix added to parameter and buffer names to compose the keys in state_dict.
            keep_vars: by default the :class:`~torch.Tensor` returned in the state dict are detached from autograd.
                If set to ``True``, detaching will not be performed.

        """
        destination: Dict[str, Union[torch.Tensor, List, Any]] = super().state_dict(
            destination=destination,  # type: ignore[arg-type]
            prefix=prefix,
            keep_vars=keep_vars,
        )
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if not self._persistent[key]:
                continue
            current_val = getattr(self, key)
            if not keep_vars:
                if isinstance(current_val, Tensor):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [cur_v.detach() if isinstance(cur_v, Tensor) else cur_v for cur_v in current_val]
            destination[prefix + key] = deepcopy(current_val)
        return destination

    def _copy_state_dict(self) -> Dict[str, Union[Tensor, List[Any]]]:
        """Copy the current state values."""
        cache: Dict[str, Union[Tensor, List[Any]]] = {}
        for attr in self._defaults:
            current_value = getattr(self, attr)

            if isinstance(current_value, Tensor):
                cache[attr] = current_value.detach().clone().to(current_value.device)
            else:
                cache[attr] = [  # safely copy (non-graph leaf) Tensor elements
                    _.detach().clone().to(_.device) if isinstance(_, Tensor) else deepcopy(_) for _ in current_value
                ]

        return cache

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Load metric states from state_dict."""
        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs such that they match the update signature of the metric."""
        # filter all parameters based on update signature except those of
        # types `VAR_POSITIONAL` for `* args` and `VAR_KEYWORD` for `** kwargs`
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params and _sign_params[k].kind not in _params)
        }

        exists_var_keyword = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in _sign_params.values())
        # if no kwargs filtered, return all kwargs as default
        if not filtered_kwargs and not exists_var_keyword:
            # no kwargs in update signature -> don't return any kwargs
            return {}
        if exists_var_keyword:
            # kwargs found in update signature -> return all kwargs to be sure to not omit any.
            # filtering logic is likely implemented within the update call.
            return kwargs
        return filtered_kwargs

    def __hash__(self) -> int:
        """Return an unique hash of the metric.

        The hash depends on both the class itself but also the current metric state, which therefore enforces that two
        instances of the same metrics never have the same hash even if they have been updated on the same data.

        """
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the addition operator."""
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical and operator."""
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __eq__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":  # type: ignore[override]
        """Construct compositional metric using the equal operator."""
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the floor division operator."""
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the greater than or equal operator."""
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the greater than operator."""
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the less than or equal operator."""
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the less than operator."""
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the matrix multiplication operator."""
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the remainder operator."""
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the multiplication operator."""
        return CompositionalMetric(torch.mul, self, other)

    def __ne__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":  # type: ignore[override]
        """Construct compositional metric using the not equal operator."""
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical or operator."""
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the exponential/power operator."""
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the addition operator."""
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical and operator."""
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: "CompositionalMetric") -> "Metric":
        """Construct compositional metric using the floor division operator."""
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the matrix multiplication operator."""
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the remainder operator."""
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the multiplication operator."""
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical or operator."""
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the exponential/power operator."""
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the subtraction operator."""
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the true divide operator."""
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical xor operator."""
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the subtraction operator."""
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the true divide operator."""
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: Union["Metric", builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical xor operator."""
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self) -> "CompositionalMetric":
        """Construct compositional metric using the absolute operator."""
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self) -> "CompositionalMetric":
        """Construct compositional metric using the not operator."""
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self) -> "CompositionalMetric":
        """Construct compositional metric using the not operator."""
        return self.__inv__()

    def __neg__(self) -> "CompositionalMetric":
        """Construct compositional metric using absolute negative operator."""
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> "CompositionalMetric":
        """Construct compositional metric using absolute operator."""
        return CompositionalMetric(torch.abs, self, None)

    def __getitem__(self, idx: int) -> "CompositionalMetric":
        """Construct compositional metric using the get item operator."""
        return CompositionalMetric(lambda x: x[idx], self, None)

    def __getnewargs__(self) -> Tuple:
        """Needed method for construction of new metrics __new__ method."""
        return tuple(
            Metric.__str__(self),
        )

    __iter__ = None


def _neg(x: Tensor) -> Tensor:
    return -torch.abs(x)


class BaseAggregator(Metric):
    """Base class for aggregation metrics.

    Args:
        fn: string specifying the reduction function
        default_value: default tensor value to use for the metric state
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        state_name: name of the metric state
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    """

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        fn: Union[Callable, str],
        default_value: Union[Tensor, List],
        nan_strategy: Union[str, float] = "error",
        state_name: str = "value",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy}"
                f" but got {nan_strategy}."
            )

        self.nan_strategy = nan_strategy
        self.add_state(state_name, default=default_value, dist_reduce_fx=fn)
        self.state_name = state_name

    def _cast_and_nan_check_input(
        self, x: Union[float, Tensor], weight: Optional[Union[float, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """Convert input ``x`` to a tensor and check for Nans."""
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)

        nans = torch.isnan(x)
        if weight is not None:
            nans_weight = torch.isnan(weight)
        else:
            nans_weight = torch.zeros_like(nans).bool()
            weight = torch.ones_like(x)
        if nans.any() or nans_weight.any():
            if self.nan_strategy == "error":
                raise RuntimeError("Encountered `nan` values in tensor")
            if self.nan_strategy in ("ignore", "warn"):
                if self.nan_strategy == "warn":
                    rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
                x = x[~(nans | nans_weight)]
                weight = weight[~(nans | nans_weight)]
            else:
                if not isinstance(self.nan_strategy, float):
                    raise ValueError(f"`nan_strategy` shall be float but you pass {self.nan_strategy}")
                x[nans | nans_weight] = self.nan_strategy
                weight[nans | nans_weight] = self.nan_strategy

        return x.to(self.dtype), weight.to(self.dtype)

    def update(self, value: Union[float, Tensor]) -> None:
        """Overwrite in child class."""

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return getattr(self, self.state_name)
    
class MeanMetric(BaseAggregator):
    """Aggregate a stream of value into their mean value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.
    - ``weight`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float value with
      arbitrary shape ``(...,)``. Needs to be broadcastable with the shape of ``value`` tensor.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated (weighted) mean over all inputs received

    Args:
       nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics.aggregation import MeanMetric
        >>> metric = MeanMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)

    """

    mean_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state("weight", default=torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor] = 1.0) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to `1.0` corresponding to simple
                harmonic average.

        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)

        # OLD:
        # value, weight = self._cast_and_nan_check_input(value, weight)

        # NEW:
        value, weight = value.to(self.dtype), weight.to(self.dtype)
        # value, weight = torch.where(torch.isnan(value), torch.tensor(0.0, dtype=self.dtype, device=self.device), value), torch.where(torch.isnan(weight), torch.tensor(0.0, dtype=self.dtype, device=self.device), weight)
        
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric([i, i+1]))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MetricCollection(ModuleDict):
    """MetricCollection class can be used to chain metrics that have the same call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.

            * arguments: similar to passing in as a list, metrics passed in as arguments will use their metric
              class name as key for the output dict.

            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

        prefix: a string to append in front of the keys of the output dict

        postfix: a string to append after the keys of the output dict

        compute_groups:
            By default the MetricCollection will try to reduce the computations needed for the metrics in the collection
            by checking if they belong to the same **compute group**. All metrics in a compute group share the same
            metric state and are therefore only different in their compute step e.g. accuracy, precision and recall
            can all be computed from the true positives/negatives and false positives/negatives. By default,
            this argument is ``True`` which enables this feature. Set this argument to `False` for disabling
            this behaviour. Can also be set to a list of lists of metrics for setting the compute groups yourself.

    .. note::
        The compute groups feature can significantly speedup the calculation of metrics under the right conditions.
        First, the feature is only available when calling the ``update`` method and not when calling ``forward`` method
        due to the internal logic of ``forward`` preventing this. Secondly, since we compute groups share metric
        states by reference, calling ``.items()``, ``.values()`` etc. on the metric collection will break this
        reference and a copy of states are instead returned in this case (reference will be reestablished on the next
        call to ``update``).

    .. note::
        Metric collections can be nested at initialization (see last example) but the output of the collection will
        still be a single flatten dictionary combining the prefix and postfix arguments from the nested collection.

    Raises:
        ValueError:
            If one of the elements of ``metrics`` is not an instance of ``pl.metrics.Metric``.
        ValueError:
            If two elements in ``metrics`` have the same ``name``.
        ValueError:
            If ``metrics`` is not a ``list``, ``tuple`` or a ``dict``.
        ValueError:
            If ``metrics`` is ``dict`` and additional_metrics are passed in.
        ValueError:
            If ``prefix`` is set and it is not a string.
        ValueError:
            If ``postfix`` is set and it is not a string.

    Example::
        In the most basic case, the metrics can be passed in as a list or tuple. The keys of the output dict will be
        the same as the class name of the metric:

        >>> from torch import tensor
        >>> from pprint import pprint
        >>> from torchmetrics import MetricCollection
        >>> from torchmetrics.regression import MeanSquaredError
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
        >>> target = tensor([0, 2, 0, 2, 0, 1, 0, 2])
        >>> preds = tensor([2, 1, 2, 0, 1, 2, 2, 2])
        >>> metrics = MetricCollection([MulticlassAccuracy(num_classes=3, average='micro'),
        ...                             MulticlassPrecision(num_classes=3, average='macro'),
        ...                             MulticlassRecall(num_classes=3, average='macro')])
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example::
        Alternatively, metrics can be passed in as arguments. The keys of the output dict will be the same as the
        class name of the metric:

        >>> metrics = MetricCollection(MulticlassAccuracy(num_classes=3, average='micro'),
        ...                            MulticlassPrecision(num_classes=3, average='macro'),
        ...                            MulticlassRecall(num_classes=3, average='macro'))
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example::
        If multiple of the same metric class (with different parameters) should be chained together, metrics can be
        passed in as a dict and the output dict will have the same keys as the input dict:

        >>> metrics = MetricCollection({'micro_recall': MulticlassRecall(num_classes=3, average='micro'),
        ...                             'macro_recall': MulticlassRecall(num_classes=3, average='macro')})
        >>> same_metric = metrics.clone()
        >>> pprint(metrics(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}
        >>> pprint(same_metric(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}

    Example::
        Metric collections can also be nested up to a single time. The output of the collection will still be a single
        dict with the prefix and postfix arguments from the nested collection:

        >>> metrics = MetricCollection([
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='macro'),
        ...         MulticlassPrecision(num_classes=3, average='macro')
        ...     ], postfix='_macro'),
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='micro'),
        ...         MulticlassPrecision(num_classes=3, average='micro')
        ...     ], postfix='_micro'),
        ... ], prefix='valmetrics/')
        >>> pprint(metrics(preds, target))  # doctest: +NORMALIZE_WHITESPACE
        {'valmetrics/MulticlassAccuracy_macro': tensor(0.1111),
         'valmetrics/MulticlassAccuracy_micro': tensor(0.1250),
         'valmetrics/MulticlassPrecision_macro': tensor(0.0667),
         'valmetrics/MulticlassPrecision_micro': tensor(0.1250)}

    Example::
        The `compute_groups` argument allow you to specify which metrics should share metric state. By default, this
        will automatically be derived but can also be set manually.

        >>> metrics = MetricCollection(
        ...     MulticlassRecall(num_classes=3, average='macro'),
        ...     MulticlassPrecision(num_classes=3, average='macro'),
        ...     MeanSquaredError(),
        ...     compute_groups=[['MulticlassRecall', 'MulticlassPrecision'], ['MeanSquaredError']]
        ... )
        >>> metrics.update(preds, target)
        >>> pprint(metrics.compute())
        {'MeanSquaredError': tensor(2.3750), 'MulticlassPrecision': tensor(0.0667), 'MulticlassRecall': tensor(0.1111)}
        >>> pprint(metrics.compute_groups)
        {0: ['MulticlassRecall', 'MulticlassPrecision'], 1: ['MeanSquaredError']}

    """

    _modules: Dict[str, Metric]  # type: ignore[assignment]
    _groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        super().__init__()

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        print(f"Metrics compute_groups: {compute_groups}")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *additional_metrics)

    @property
    def metric_state(self) -> Dict[str, Dict[str, Any]]:
        """Get the current state of the metric."""
        return {k: m.metric_state for k, m in self.items(keep_base=False, copy_state=False)}

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Call forward for each metric sequentially.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.

        """
        return self._compute_and_reduce("forward", *args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Call update for each metric sequentially.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.

        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            # Delete the cache of all metrics to invalidate the cache and therefore recent compute calls, forcing new
            # compute calls to recompute
            for k in self.keys(keep_base=True):
                mi = getattr(self, str(k))
                mi._computed = None
            for cg in self._groups.values():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            if self._state_is_copy:
                # If we have deep copied state in between updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False
        else:  # the first update always do per metric to form compute groups
            for m in self.values(copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterate over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one. The complexity of the method is approximately
        ``O(number_of_metrics_in_collection ** 2)``, as all metrics need to be compared to all other metrics.

        """
        num_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups) != num_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups) == num_groups:
                break
            num_groups = len(self._groups)

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults:
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):
                return False

            if isinstance(state1, Tensor) and isinstance(state2, Tensor):
                return state1.shape == state2.shape and allclose(state1, state2)

            if isinstance(state1, list) and isinstance(state2, list):
                return all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2))

        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference

        """
        if not self._state_is_copy:
            for cg in self._groups.values():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        # Determine if we just should set a reference or a full copy
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                    mi._update_count = deepcopy(m0._update_count) if copy else m0._update_count
        self._state_is_copy = copy

    def compute(self) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        return self._compute_and_reduce("compute")

    def _compute_and_reduce(
        self, method_name: Literal["compute", "forward"], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute result from collection and reduce into a single dictionary.

        Args:
            method_name: The method to call on each metric in the collection.
                Should be either `compute` or `forward`.
            args: Positional arguments to pass to each metric (if method_name is `forward`)
            kwargs: Keyword arguments to pass to each metric (if method_name is `forward`)

        Raises:
            ValueError:
                If method_name is not `compute` or `forward`.

        """
        result = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute()
            elif method_name == "forward":
                res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(f"method_name should be either 'compute' or 'forward', but got {method_name}")
            result[k] = res

        _, duplicates = _flatten_dict(result)

        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    # if duplicates of keys we need to add unique prefix to each key
                    if duplicates:
                        stripped_k = k.replace(getattr(m, "prefix", ""), "")
                        stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                        key = f"{stripped_k}_{key}"
                    if getattr(m, "_from_collection", None) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", None) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}

    def reset(self) -> None:
        """Call reset for each metric sequentially."""
        for m in self.values(copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            # reset state reference
            self._compute_groups_create_state_ref()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection":
        """Make a copy of the metric collection.

        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict.

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Change if metric states should be saved to its state_dict after initialization."""
        for m in self.values(copy_state=False):
            m.persistent(mode)

    def add_metrics(
        self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]], *additional_metrics: Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                sel = metrics if isinstance(m, Metric) else remain
                sel.append(m)

            if remain:
                rank_zero_warn(
                    f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored."
                )
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[k] = v
        else:
            raise ValueError(
                "Unknown input to MetricCollection. Expected, `Metric`, `MetricCollection` or `dict`/`sequence` of the"
                f" previous, but got {metrics}"
            )

        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group

        """
        if isinstance(self._enable_compute_groups, list):
            self._groups = dict(enumerate(self._enable_compute_groups))
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                            f" Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}"
                        )
            self._groups_checked = True
        else:
            # Initialize all metrics as their own compute group
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> Dict[int, List[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    def _to_renamed_ordered_dict(self) -> OrderedDict:
        od = OrderedDict()
        for k, v in self._modules.items():
            od[self._set_name(k)] = v
        return od

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys of the MetricDict."""
        return iter(self.keys())

    # TODO: redefine this as native python dict
    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.

        Args:
            keep_base: Whether to add prefix/postfix on the items collection.

        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_ordered_dict().keys()

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[Tuple[str, Metric]]:
        r"""Return an iterable of the ModuleDict key/value pairs.

        Args:
            keep_base: Whether to add prefix/postfix on the collection.
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Metric]:
        """Return an iterable of the ModuleDict values.

        Args:
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Metric:
        """Retrieve a single metric from the collection.

        Args:
            key: name of metric to retrieve
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        if self.prefix:
            key = key.removeprefix(self.prefix)
        if self.postfix:
            key = key.removesuffix(self.postfix)
        return self._modules[key]

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        """Return the representation of the metric collection including all metrics in the collection."""
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "MetricCollection":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string.

        """
        for m in self.values(copy_state=False):
            m.set_dtype(dst_type)
        return self

    def plot(
        self,
        val: Optional[Union[Dict, Sequence[Dict]]] = None,
        ax: Optional[Union[_AX_TYPE, Sequence[_AX_TYPE]]] = None,
        together: bool = False,
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """Plot a single or multiple values from the metric.

        The plot method has two modes of operation. If argument `together` is set to `False` (default), the `.plot`
        method of each metric will be called individually and the result will be list of figures. If `together` is set
        to `True`, the values of all metrics will instead be plotted in the same figure.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: Either a single instance of matplotlib axis object or an sequence of matplotlib axis objects. If
                provided, will add the plots to the provided axis objects. If not provided, will create a new. If
                argument `together` is set to `True`, a single object is expected. If `together` is set to `False`,
                the number of axis objects needs to be the same length as the number of metrics in the collection.
            together: If `True`, will plot all metrics in the same axis. If `False`, will plot each metric in a separate

        Returns:
            Either install tuple of Figure and Axes object or an sequence of tuples with Figure and Axes object for each
            metric in the collection.

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
            ValueError:
                If `together` is not an bool
            ValueError:
                If `ax` is not an instance of matplotlib axis object or a sequence of matplotlib axis objects

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics import MetricCollection
            >>> from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
            >>> metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(), BinaryRecall()])
            >>> metrics.update(torch.rand(10), torch.randint(2, (10,)))
            >>> fig_ax_ = metrics.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics import MetricCollection
            >>> from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
            >>> metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(), BinaryRecall()])
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metrics(torch.rand(10), torch.randint(2, (10,))))
            >>> fig_, ax_ = metrics.plot(values, together=True)

        """
        if not isinstance(together, bool):
            raise ValueError(f"Expected argument `together` to be a boolean, but got {type(together)}")
        if ax is not None:
            if together and not isinstance(ax, _AX_TYPE):
                raise ValueError(
                    f"Expected argument `ax` to be a matplotlib axis object, but got {type(ax)} when `together=True`"
                )
            if not together and not (
                isinstance(ax, Sequence) and all(isinstance(a, _AX_TYPE) for a in ax) and len(ax) == len(self)
            ):
                raise ValueError(
                    f"Expected argument `ax` to be a sequence of matplotlib axis objects with the same length as the "
                    f"number of metrics in the collection, but got {type(ax)} with len {len(ax)} when `together=False`"
                )
        val = val or self.compute()
        if together:
            return plot_single_or_multi_val(val, ax=ax)
        fig_axs = []
        for i, (k, m) in enumerate(self.items(keep_base=False, copy_state=False)):
            if isinstance(val, dict):
                f, a = m.plot(val[k], ax=ax[i] if ax is not None else ax)
            elif isinstance(val, Sequence):
                f, a = m.plot([v[k] for v in val], ax=ax[i] if ax is not None else ax)
            fig_axs.append((f, a))
        return fig_axs
