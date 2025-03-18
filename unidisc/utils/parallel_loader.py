import itertools
import threading
import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
from decoupled_utils import gprint


class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
    self.loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self.queue = kq.Queue(maxsize=device_prefetch_size)
    self.close_queue_count = itertools.count()
    gprint("PerDeviceQueue initialized")


class PerDeviceLoader(object):

  def __init__(self, loader, device):
    self._loader = loader
    self._device = device
    self._mark_step_batch_count = loader.batches_per_execution - 1
    self._batches_yielded = 0
    gprint("PerDeviceLoader initialized")

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def __len__(self):
    return self._loader.per_device_samples()

  def next(self):
    gprint("Getting next item")
    if xp.get_tracer_marked_step():
      gprint("Marking step traced")
      xp.set_tracer_marked_step(False)
      self._batches_yielded += 1
    else:
      if self._mark_step_batch_count <= self._batches_yielded:
        gprint(f"before Marking step, {self._batches_yielded}, {self._mark_step_batch_count}")
        self._batches_yielded = 0
        xm.mark_step()
        gprint("Marking step")
      else:
        self._batches_yielded += 1
        gprint("Not marking step, batches yielded: ", self._batches_yielded)

    gprint("Getting next item")
    item = self._loader.next_item(self._device)
    gprint("Item retrieved")
    if item is None:
      gprint("Item is None, marking step", item)
      xm.mark_step()
      gprint("Marked step, exiting since item is None")
      raise StopIteration
    return item


class ParallelLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    devices (`torch.device`...): The list of devices where the data has to be
      sent. The i-th sample returned by the `loader` will be sent to `devices[i
      % len(devices)]`.
    batchdim (int, optional): The dimension which is holding the batch size.
      Default: 0
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 8
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 4
    host_to_device_transfer_threads (int, optional): The number of threads that
      work in parallel to transfer data from loader queue to device queue.
      Default: 1
    input_sharding (ShardingSpec, optional): Sharding spec to apply to
      compatible input tensors after loading.
      Default: None
  """

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               batches_per_execution=1,
               loader_prefetch_size=12,
               device_prefetch_size=4,
               host_to_device_transfer_threads=4,
               input_sharding=None):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._batches_per_execution = batches_per_execution
    self._done = False
    self._queues = dict()
    self._input_sharding = input_sharding
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue in self._queues.values():
      for i in range(host_to_device_transfer_threads):
        thread = threading.Thread(
            target=self._worker,
            args=(
                dqueue,
                host_to_device_transfer_threads,
            ))
        thread.daemon = True
        thread.start()

    gprint("ParallelLoader finished")

  def per_device_loader(self, device):
    """Retrieves the loader iterator object for the given device.

    Args:
      device (`torch.device`): The device whole loader is being requested.

    Returns:
      The loader iterator object for the `device`. This is not a
      `torch.utils.data.DataLoader` interface, but a Python iterator which
      returns the same tensor data structure as returned by the wrapped
      `torch.utils.data.DataLoader`, but residing on XLA devices.
    """
    return PerDeviceLoader(self, torch.device(device))

  def per_device_samples(self):
    return len(self._loader) // len(self._devices)

  def next_item(self, device):
    dqueue = self._queues[device]
    gprint("Getting item from queue")
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in self._queues.values():
      dqueue.queue.close()
      dqueue.loader_queue.close()

  @property
  def batches_per_execution(self):
    return self._batches_per_execution

  def _loader_worker(self):
    queues = list(self._queues.values())
    data_iter = enumerate(self._loader)
    batch = []
    while not self._done:
      try:
        gprint("Getting next item")
        _, data = next(data_iter)
        gprint("Item retrieved inside loader worker")
      except StopIteration:
        gprint("StopIteration")
        break

      gprint("Appending item to batch, type: ", type(data))
      batch.append(data)
      if len(batch) == len(self._devices):
        gprint("Batch full, sending to queues")
        for queue_no, device_batch in enumerate(batch):
          queues[queue_no].loader_queue.put(device_batch)
        batch = []

      gprint(f"Current batch length: {len(batch)}")
    gprint("Loader worker done")
    for dqueue in queues:
      dqueue.loader_queue.close_write()
    gprint("Loader worker closed")

  def _get_batch(self, dqueue):
    batch = []
    while dqueue.queue.max_size() > len(batch):
      gprint("Getting item from loader queue")
      item = dqueue.loader_queue.get()
      gprint(f"Item retrieved")
      if item is None:
        gprint("Item is None, breaking", item)
        break
      batch.append(item)
    gprint(f"Batch retrieved: length {len(batch)}")
    return batch

  def _worker(self, dqueue, host_to_device_transfer_threads):
    device = torch.device(dqueue.device)
    gprint("Worker initialized")
    while True:
      gprint("Getting batch")
      batch = self._get_batch(dqueue)
      gprint("Batch retrieved")
      if not batch:
        gprint("Batch empty, breaking, ", batch)
        break
        
      gprint("Sending batch to device")
      batch = xm.send_cpu_data_to_device(batch, device, self._input_sharding)
      gprint("Batch sent to device")
      for data in batch:
        gprint("Putting data in queue")
        if data is None:
          print("Data is None! ", data)
        dqueue.queue.put(data)
        gprint("Data put in queue")
    gprint("Closing queue")
    close_queue_count = next(dqueue.close_queue_count)
    gprint(f"Close queue count: {close_queue_count}")
    if close_queue_count == host_to_device_transfer_threads - 1:
      gprint("Closing queue")
      dqueue.queue.close_write()
      gprint("Queue closed")

    gprint("Worker done!!")


class MpDeviceLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  This class should only be using with multi-processing data parallelism.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    device (`torch.device`...): The device where the data has to be sent.
    kwargs: Named arguments for the `ParallelLoader` constructor.
  """

  def __init__(self, loader, device, **kwargs):
    self._loader = loader
    self._device = device
    self._parallel_loader_kwargs = kwargs

  def __iter__(self):
    parallel_loader = ParallelLoader(self._loader, [self._device], **self._parallel_loader_kwargs)
    gprint("ParallelLoader initialized")
    return parallel_loader.per_device_loader(self._device)

  def __len__(self):
    return len(self._loader)
