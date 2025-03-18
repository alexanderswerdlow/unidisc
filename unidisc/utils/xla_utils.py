from decoupled_utils import is_torch_xla_available, rprint, gprint
import torch
from functools import partial

# Wrap the base model with an outer FSDP wrapper
def shard_output(output, mesh):
    import torch_xla.distributed.spmd as xs
    from transformers.modeling_outputs import CausalLMOutputWithPast

    real_output = None
    if isinstance(output, torch.Tensor):
        real_output = output
    elif isinstance(output, tuple):
        real_output = output[0]
    elif isinstance(output, CausalLMOutputWithPast):
        real_output = output.logits

    xs.mark_sharding(real_output, mesh, (('dcn', 'fsdp'), None, None))


from typing import Set, Type
import torch.nn as nn

def _module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: Set[Type[nn.Module]],
    min_num_params: int,
) -> bool:
    """
    This auto wrap policy wraps every module that is an instance of any type in
    ``module_classes`` as its own FSDP instance. The root module given by
    ``module`` is always wrapped as an FSDP instance regardless. Since the
    wrapping proceeds bottom up, each FSDP instance manages the parameters in
    its subtree excluding any already managed by a child FSDP instance.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.
        module_classes (Set[Type[nn.Module]]): Set of module classes that are
            wrapped as FSDP instances.

    Returns:
        ``True`` if ``recurse=True``, and whether ``module`` should be wrapped
        if ``recurse=False``.
    """

    print(f"Found {module.__class__.__name__} with {nonwrapped_numel} parameters; we have min_num_params={min_num_params}")

    if recurse and nonwrapped_numel >= min_num_params:
        print(f"Recursing down {module.__class__.__name__}")
        return True  # always recurse
    
    if nonwrapped_numel >= min_num_params and isinstance(module, tuple(module_classes)):
        print(f"Wrapping {module.__class__.__name__}")
    return isinstance(module, tuple(module_classes)) and nonwrapped_numel >= min_num_params

def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[nn.Module]],
    min_num_params: int = int(1e6)
) -> bool:
    """
    See :func:`_module_wrap_policy`, where ``transformer_layer_cls`` is the
    same as ``module_classes``. Note that shared parameters must be wrapped in
    the same FSDP instance, so this auto wrap policy can help wrap shared
    embeddings into the same FSDP instance for transformer models.
    """
    return _module_wrap_policy(module, recurse, unwrapped_params, transformer_layer_cls, min_num_params)

# Taken from HF Transformers
def wrap_xla_fsdp(config, model):
    import torch_xla.distributed.spmd as spmd
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    import torch_xla

    is_fsdp_xla_v2_enabled = True
    try:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
        from torch_xla.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy
        )

        if is_fsdp_xla_v2_enabled:
            from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                SpmdFullyShardedDataParallel as FSDPv2,
            )
    except ImportError:
        raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
    
    # See: https://github.com/pytorch-tpu/transformers/blob/alanwaketan/flash_attention/examples/pytorch/language-modeling/run_clm.py
    auto_wrap_policy = None
    auto_wrapper_callable = None
    fsdp_transformer_layer_cls_to_wrap = ["DDiTBlock", "ChameleonDecoderLayer", "OpenELMDecoderLayer"]

    if getattr(config.trainer, "fsdp_size_based_auto_wrap", False):
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(1e6),
        )
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        from transformers.trainer_pt_utils import get_module_class_from_name
        transformer_cls_to_wrap = set()
        found_valid_layer = False
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is not None:
                transformer_cls_to_wrap.add(transformer_cls)
                rprint(f"Found valid layer: {layer_class}")
                found_valid_layer = True

        if not found_valid_layer:
            raise Exception("Could not find the transformer layer class to wrap in the model.")

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )

    gradient_checkpointing = config.trainer.use_gradient_checkpointing
    if gradient_checkpointing:
        def auto_wrapper_callable(m, *args, **kwargs):
            target_cls = FSDP if not is_fsdp_xla_v2_enabled else FSDPv2
            return target_cls(checkpoint_module(m), *args, **kwargs)
    
    patch_xla_linear = getattr(config.trainer, "patch_xla_linear", False)
    if patch_xla_linear:
        rprint("WARNING!!!! Patching XLA Linear")
        from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
        model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

    # Patch `xm.optimizer_step` should not reduce gradients in this case,
    # as FSDP does not need gradient reduction over sharded parameters.
    patch_xla_optimizer_step = getattr(config.trainer, "patch_xla_optimizer_step", True)
    if patch_xla_optimizer_step:
        def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            loss = optimizer.step(**optimizer_args)
            if barrier:
                xm.mark_step()
            return loss
        xm.optimizer_step = patched_optimizer_step
    
    custom_spmd_wrap = getattr(config.trainer, "custom_spmd_wrap", False)
    custom_chameleon_spmd_wrap = getattr(config.trainer, "custom_chameleon_spmd_wrap", False)
    auto_spmd = getattr(config.trainer, "auto_spmd", False)

    if auto_spmd:
        pass
    elif custom_spmd_wrap:
        # Replace the meta tensor parameter with the initialized XLA tensor
        # Shard each parameter in the model based on the sharding strategy provided.
        gprint("Custom SPMD wrap enabled")
        spmd_mesh = xs.get_global_mesh()
        spmd_fsdp_sharding = True
        spmd_2d_sharding = 0
        for name, param in model.named_parameters():
            if spmd_fsdp_sharding:
                print('> [FSDP] Sharding tensor', name, param.shape, param.dtype)
                # We don't care about layernorm's weights, and
                # LLaMA doesn't use biases.
                if len(param.shape) == 1:
                    continue
                assert len(param.shape) == 2

                # Shard the largest dimension
                if param.shape[0] > param.shape[1]:
                    partition_spec = ('fsdp', None)
                else:
                    partition_spec = (None, 'fsdp')
                xs.mark_sharding(param, spmd_mesh, partition_spec)
            elif spmd_2d_sharding > 0:
                # Apply 2D sharding:
                print('> [2D] Sharding tensor', name, param.shape)

                # We don't care about layernorm's weights, and
                # LLaMA doesn't use biases.
                if len(param.shape) == 1:
                    continue

                if 'embed_tokens' in name:
                    xs.mark_sharding(param, spmd_mesh, ('model', 'fsdp'))
                elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    xs.mark_sharding(param, spmd_mesh, ('fsdp', 'model'))
                elif 'o_proj' in name:
                    xs.mark_sharding(param, spmd_mesh, ('model', 'fsdp'))
                elif 'gate_proj' in name or 'up_proj' in name:
                    xs.mark_sharding(param, spmd_mesh, ('model', 'fsdp'))
                elif 'down_proj' in name:
                    xs.mark_sharding(param, spmd_mesh, ('fsdp', 'model'))
                elif 'lm_head' in name:
                    xs.mark_sharding(param, spmd_mesh, ('model', 'fsdp'))

            print(f'{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

        for i in range(len(model.model.blocks)):
            spmd.xla_sharding.apply_backward_optimization_barrier(model.blocks[i])
    
    elif custom_chameleon_spmd_wrap:
        # Replace the meta tensor parameter with the initialized XLA tensor
        # Shard each parameter in the model based on the sharding strategy provided.
        gprint("Custom Chameleon SPMD wrap enabled")
        spmd_mesh = xs.get_global_mesh()
        spmd_fsdp_sharding = True
        spmd_2d_sharding = 0
        for name, param in model.named_parameters():
            # We don't care about layernorm's weights, and
            # LLaMA doesn't use biases.
            if len(param.shape) == 1:
                gprint(f"Skipping shard of {name} with {param.numel()} elements because it is 1D, shape: {param.shape}")
                continue

            if param.requires_grad is False and getattr(config.trainer, "no_shard_grad_false", False):
                gprint(f"Skipping shard of {name} with {param.numel()} elements because requires_grad is False, shape: {param.shape}")
                continue

            if param.numel() < int(1e6) and getattr(config.trainer, "no_shard_small", False):
                gprint(f"Skipping shard of {name} with {param.numel()} elements, shape: {param.shape}")
                continue

            assert len(param.shape) == 2

            print('> [FSDP] Sharding tensor', name, param.shape, param.dtype)

            # Shard the largest dimension
            if param.shape[0] > param.shape[1]:
                partition_spec = ('fsdp', None)
            else:
                partition_spec = (None, 'fsdp')

            xs.mark_sharding(param, spmd_mesh, partition_spec)
            print(f'{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

        gprint(f"Model: {model.__class__.__name__}")
        for block in (model.base_model.model.model.layers if config.model.use_lora else model.model.layers):
            gprint(f"Applying barrier to {block.__class__.__name__}")
            spmd.xla_sharding.apply_backward_optimization_barrier(block)

    else:
        gprint(f"Using FSDPv2, {xs.get_global_mesh()}")
        model = FSDPv2(
            model,
            shard_output=shard_output,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
        )

        for name, param in model.named_parameters():
            if param.requires_grad is False or param.numel() < int(1e6):
                xs.clear_sharding(param)
                xs.mark_sharding(param, xs.get_global_mesh(), tuple([None] * len(param.shape)))

            if torch_xla._XLAC._get_xla_sharding_spec(param) != "":
                gprint(f'Sharding {name} {param.shape} requires_grad={param.requires_grad} numel={param.numel()} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

    return model


def tpu_spmd_dataloader(dataloader, device):
    if is_torch_xla_available():
        import torch_xla.distributed.spmd as xs
        import torch_xla
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        sharding_spec = xs.ShardingSpec(xs.get_global_mesh(), (('dcn', 'fsdp'), None))
        if isinstance(dataloader, MpDeviceLoader):
            rprint("Modifying existing MpDeviceLoader")
            dataloader._parallel_loader_kwargs["input_sharding"] = sharding_spec
        else:
            rprint("Creating MpDeviceLoader")
            rprint(f"Drop Last: {dataloader.drop_last}")
            loader = MpDeviceLoader(
                dataloader,
                device=torch_xla.device(),
                input_sharding=sharding_spec,
            )
            loader.dataset = dataloader.dataset
            dataloader = loader
        return dataloader
    else:
        return dataloader

