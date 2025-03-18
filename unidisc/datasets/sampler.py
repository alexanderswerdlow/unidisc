import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from decoupled_utils import gprint, rprint, dprint, tensor_hash

def _get_len(dataset):
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return 1000000000
    else:
        return len(dataset)

class WeightedDatasetSampler(Sampler):
    def __init__(self, combined_dataset, generator=None, batch_size=100000):
        self.dataset_names = combined_dataset.dataset_names
        self.datasets = combined_dataset.datasets
        self.weights = combined_dataset.weights
        self.generator = generator
        self.batch_size = batch_size
        
        assert len(self.datasets) == len(self.weights), "Each dataset must have a corresponding weight"
        
        # Samples per epoch is the least common multiple of dataset sizes
        self.lcm_size = np.lcm.reduce([_get_len(d) for d in self.datasets])
        if self.lcm_size < 1:
            self.lcm_size = max([_get_len(d) for d in self.datasets]) * 1000
        
        total_dataset_sizes_sum = sum(_get_len(d) for d in self.datasets)
        self.weights = [
            weight if weight >= 0 else _get_len(dataset) / total_dataset_sizes_sum
            for weight, dataset in zip(self.weights, self.datasets)
        ]
        self.counts = {name: int(round(weight / sum(self.weights) * self.lcm_size)) for weight, name in zip(self.weights, self.dataset_names)}
        rprint(f"LCM Size: {self.lcm_size}, Weighted Sampler Counts: {self.counts}, Weights: {self.weights}")
        self.dataset_idx_to_name = {idx: name for idx, name in enumerate(self.dataset_names)}
        self._reset_state()
        self.raise_stop_iteration = False

    def _generate_multinomial_batch(self):
        normalized_weights = [weight / sum(self._state['available_weights']) for weight in self._state['available_weights']]
        _batched_indices = torch.multinomial(torch.tensor(normalized_weights), self.batch_size, replacement=True, generator=self.generator)
        for i, weight in enumerate(self.weights):
            if weight == 0 and i in _batched_indices:
                print(f"WARNING: sampling item with zero weight")

        map_to_original = torch.tensor(self._state['available_datasets']).to(_batched_indices)
        self._state['batched_indices'] = map_to_original[_batched_indices].tolist()
        self._state['batch_pointer'] = 0

    def state_dict(self):
        if self.generator is not None:
            gprint(f"Generator state at saving: {tensor_hash(self.generator.get_state())}")

        gprint(f"Counts: {sum(self._state['current_counts'].values())}, Batch pointer: {self._state['batch_pointer']}")
        return self._state
    
    def load_state_dict(self, state_dict):
        self._state['generator'] = state_dict['generator']
        self._state['batched_indices'] = state_dict['batched_indices']
        self._state['batch_pointer'] = state_dict['batch_pointer']

        if set(state_dict['current_counts'].keys()) == set(self.dataset_names):
            self._state['current_counts'] = state_dict['current_counts']
            self._state['dataset_iters'] = state_dict['dataset_iters']
        else:
            rprint(f"Dataset names mismatch, updating state_dict")
            self._state['current_counts'].update(state_dict['current_counts'])
            if self._state['dataset_iters'] is None:
                self._state['dataset_iters'] = {name: (torch.randperm(_get_len(self.datasets[idx]), generator=self._state['generator']), 0) for idx, name in enumerate(self.dataset_names) if name not in state_dict['current_counts']}
            gprint(f"new len of dataset iters: {len(self._state['dataset_iters'])}")
            self._state['dataset_iters'].update(state_dict['dataset_iters'])
            gprint(f"final len of dataset iters: {len(self._state['dataset_iters'])}")

        gprint(f"Weights: {self._state['available_weights']}")
        
        gprint(f"Updated state_dict: {self._state['current_counts']}")
        gprint(f"Finished loading sampler state_dict")
        if self.generator is not None:
            gprint(f"Generator state at loading: {tensor_hash(self._state['generator'].get_state())}")

        gprint(f"Counts: {sum(self._state['current_counts'].values())}, Batch pointer: {self._state['batch_pointer']}")

    def restart(self):
        if self._state['dataset_iters'] is None:
            rprint(f"Resetting dataset iter. We have: {self.dataset_names}")
            self._state['dataset_iters'] = {name: (torch.randperm(_get_len(self.datasets[idx]), generator=self._state['generator']), 0) for idx, name in enumerate(self.dataset_names)}

    def exhausted(self):
        self._reset_state()
        if self.raise_stop_iteration:
            dprint(f"Sampler exhausted")
            raise StopIteration
        else:
            self.restart()

    def check_is_not_exhausted(self):
        return any(self._state['current_counts'][name] < self.counts[name] for name in self.dataset_names)

    def __iter__(self):
        self.restart()
        while self.check_is_not_exhausted() or self.raise_stop_iteration is False:
            if len(self._state['available_datasets']) == 0 or (self.raise_stop_iteration is False and self.check_is_not_exhausted() is False):
                self.exhausted()
            
            if self._state['batched_indices'] is None or self._state['batch_pointer'] >= self.batch_size:
                self._generate_multinomial_batch()
            
            try:
                dataset_name = self.dataset_idx_to_name[self._state['batched_indices'][self._state['batch_pointer']]]
            except Exception as e:
                gprint(f"Error in dataset_name: {e}, batch pointer: {self._state['batch_pointer']}, batched indices: {self._state['batched_indices']}, dataset idx to name: {self.dataset_idx_to_name}")
            self._state['batch_pointer'] += 1
            
            tensor, idx = self._state['dataset_iters'][dataset_name]
            if idx >= len(tensor):
                rprint(f"Resetting dataset iter for {dataset_name}")
                tensor = torch.randperm(_get_len(self.datasets[self.dataset_names.index(dataset_name)]), generator=self._state['generator'])
                idx = 0
                self._state['dataset_iters'][dataset_name] = (tensor, idx)
            
            self._state['dataset_iters'][dataset_name] = (tensor, idx + 1)
            self._state['current_counts'][dataset_name] += 1

            dataset_idx = self.dataset_names.index(dataset_name)
            if self._state['current_counts'][dataset_name] >= self.counts[dataset_name]:
                index = self._state['available_datasets'].index(dataset_idx)
                self._state['available_datasets'].pop(index)
                self._state['available_weights'].pop(index)
                if len(self._state['available_datasets']) > 0:
                    rprint(f"{dataset_name} has no samples left, resetting dataset sampler")
                    self._generate_multinomial_batch()
            
            # print(f"Yielding {dataset_idx}, {tensor[idx].item()}")
            yield dataset_idx, tensor[idx].item()

        self.exhausted()

    def _reset_state(self):
        self._state = {
            'batched_indices': None,
            'batch_pointer': 0,
            'current_counts': {name: 0 for name in self.dataset_names},
            'available_datasets': list(range(len(self.dataset_names))),
            'available_weights': [weight / sum(self.weights) for weight in self.weights],
            'dataset_iters': None,
            'generator': self.generator,
        }

    def __len__(self):
        return sum(self.counts.values())

class DummyDataset(Dataset):
    def __init__(self, dataset_name, size):
        self.dataset_name = dataset_name
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.dataset_name, idx)  # Just return the index for testing