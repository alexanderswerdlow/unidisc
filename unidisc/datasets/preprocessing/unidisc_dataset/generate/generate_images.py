from __future__ import annotations

import io
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import typer
from tqdm import tqdm

from decoupled_utils import breakpoint_on_error, check_gpu_memory_usage, rprint
from unidisc.datasets.prompts.llm_prompting import get_llm

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

hostname = socket.gethostname()
hosting_node = "main-node-0-0"
prompt_folder = Path(f"diffusion/prompts/inputs")
root_output_folder = Path(f"diffusion/prompts/generated_images/v4")

def get_list_from_file(data_path):
    output_data = None

    if not Path(data_path).exists():
        data_path = prompt_folder / data_path

    if str(data_path).endswith(".txt"):
        with open(data_path, "r", encoding="utf-8") as file:
            output_data = file.readlines()

    elif str(data_path).endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as file:
            output_data = json.load(file)

    if output_data:
        output_data = ["".join(char for char in line if ord(char) < 128).replace("\n", "").strip() for line in output_data]
        output_data = [line for line in output_data if line]
    else:
        print("Warning: No data loaded from the file.")

    output_data = [x.strip() for x in output_data]

    return output_data


def get_to_process(timestamp, data_path, return_raw_data=False):
    data = get_list_from_file(data_path)
    return list(range(len(data)))


def get_pipe(compile=True, model="stabilityai/stable-diffusion-3-medium-diffusers", quantize_text_encoder=False, **kwargs):
    from diffusers import (LuminaText2ImgPipeline, PixArtAlphaPipeline,
                           PixArtSigmaPipeline, StableDiffusion3Pipeline,
                           Transformer2DModel)

    torch.set_float32_matmul_precision("high")

    if compile:
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

    if "PixArt-Sigma" in model:
        model_cls = PixArtSigmaPipeline
        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",  # "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            subfolder="transformer",
            use_safetensors=True,
            **kwargs,
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", transformer=transformer, use_safetensors=True, **kwargs
        )
    elif "stable-diffusion-3" in model and quantize_text_encoder:
        from transformers import BitsAndBytesConfig, T5EncoderModel

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder = T5EncoderModel.from_pretrained(model, subfolder="text_encoder_3", quantization_config=quantization_config, device_map="auto")
        pipe = StableDiffusion3Pipeline.from_pretrained(model, text_encoder_3=text_encoder, device_map="balanced", **kwargs)
    else:
        if "stable-diffusion-3" in model:
            model_cls = StableDiffusion3Pipeline
        elif "PixArt-alpha" in model:
            model_cls = PixArtAlphaPipeline
        else:
            model_cls = LuminaText2ImgPipeline

        if "PixArt" in model:
            kwargs["use_safetensors"] = True
            kwargs["device_map"] = "balanced"

        rprint(f"Loading model: {model}")
        pipe = model_cls.from_pretrained(model, **kwargs)

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # pipe.enable_xformers_memory_efficient_attention()

    if compile:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

        rprint(f"Compiled model: {model}")

    return pipe


def get_indices_from_server(slurm_job_id, chunk_size, total_indices, output_dir, expected_samples_per_index):
    import time
    rprint(f"Getting indices from server: {slurm_job_id}, {chunk_size}, {total_indices}, {output_dir}, {expected_samples_per_index}")
    retries = 10
    for attempt in range(retries):
        try:
            response = requests.post(
                f"http://{hosting_node}:5000/get_indices",
                json={"slurm_job_id": slurm_job_id, "chunk_size": chunk_size, "total_indices": total_indices, "output_dir": str(output_dir), "expected_samples_per_index": expected_samples_per_index},
                timeout=1200
            )
            rprint(f"Response: {response}")
            if response.status_code == 200:
                return response.json().get("indices", [])
        except requests.RequestException as e:
            rprint(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(120)
            else:
                raise Exception(f"Failed to get indices from server: {slurm_job_id}, {chunk_size}, {total_indices}, {output_dir}, {expected_samples_per_index}")


def train(data_path, indices, batch_size=None, compile=True, model="stabilityai/stable-diffusion-3-medium-diffusers", resolution=512, augment_prompts=True, **kwargs):
    rprint(f"Unused kwargs: {kwargs}, augment_prompts: {augment_prompts}")
    rprint(f"Before getting device")
    result = torch.cuda.get_device_name()
    rprint(f"Initializing on {hostname}, Got {len(indices)} indices.")
    rprint(f"GPU name: {result}")
    check_gpu_memory_usage()

    model_kwargs = dict(torch_dtype=torch.bfloat16, compile=compile, model=model)

    if "A5500" in result or "A5000" in result:
        if compile:
            gpu_batch_size = 4
        else:
            gpu_batch_size = 8
    elif "A100" in result or "6000 Ada" in result or "A6000" in result:
        gpu_batch_size = 12
    elif "V100" in result:
        model_kwargs.update(torch_dtype=torch.float16)
        gpu_batch_size = 6
    else:
        model_kwargs.update(torch_dtype=torch.float16)
        gpu_batch_size = 4

    if ("080" in result or "TITAN X" in result) and "stable-diffusion-3" in model:
        rprint("Disabling text encoder 3 and compile...")
        model_kwargs.update(text_encoder_3=None, tokenizer_3=None)
        compile = False
        model_kwargs["compile"] = False

    if "3090" in result:
        compile = False
        model_kwargs["compile"] = False

    if batch_size is None:
        batch_size = gpu_batch_size

    rprint(f"Using batch size: {batch_size}")
    pipe = get_pipe(**model_kwargs)

    if ("080" in result or "TITAN X" in result) and ("stable-diffusion-3" in model or "PixArt" in model):
        rprint("Enabling model offload...")
        pipe.enable_model_cpu_offload()

    llm_model_type = "gpt-4o-mini"
    llm = get_llm(hosting_node=hosting_node, llm_model_type=llm_model_type)

    selected_lines = get_list_from_file(data_path)

    output_folder = root_output_folder / data_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name()
    new_samples_per_index = 10

    for i in tqdm(range(0, (len(indices) // batch_size) + 1), unit="it", unit_scale=batch_size, desc="Processing batches"):
        initial_prompts = []
        for j in range(batch_size):
            if i * batch_size + j < len(indices):
                initial_prompts.append(
                    (indices[i * batch_size + j], 0, "", selected_lines[indices[i * batch_size + j]], selected_lines[indices[i * batch_size + j]])
                )

        if len(initial_prompts) == 0:
            rprint("No prompts to process...skipping")
            continue

        augmented_prompts = []
        successful_initial_prompts = []
        for prompt_index, _, _, original_prompt, _ in initial_prompts:
            if augment_prompts:
                try:
                    rprint(f'Given original prompt: "{original_prompt}", (index: {prompt_index})')
                    generated_prompts, llm_model_name = llm(prompt=original_prompt, new_samples_per_index=new_samples_per_index)
                except Exception as e:
                    rprint(f"Error generating prompts for {original_prompt}: {e}")
                    continue

                rprint(f"Generated prompts: {generated_prompts}")

                if len(generated_prompts) < new_samples_per_index - 1:  # Allow one less
                    rprint(f"Only {len(generated_prompts)} prompts generated for {original_prompt}")
                    continue

                augmented_prompts.extend(
                    [(prompt_index, k + 1, llm_model_name, original_prompt, aug_prompt) for k, aug_prompt in enumerate(generated_prompts)]
                )
            successful_initial_prompts.append((prompt_index, 0, "", original_prompt, original_prompt))

        successful_initial_prompts = [
            prompt for prompt in successful_initial_prompts
            if not (output_folder / f"{prompt[0]}_0.json").exists()
        ]

        if len(successful_initial_prompts) < len(initial_prompts):
            rprint(f"Filtered out {len(initial_prompts) - len(successful_initial_prompts)} already processed initial prompts")

        all_prompts = successful_initial_prompts + augmented_prompts
        rprint(f"Total prompts to generate: {len(all_prompts)}")

        total_generated = 0
        for k in range(0, len(all_prompts), batch_size):
            batch_all_prompts = all_prompts[k : k + batch_size]
            pipe_kwargs = dict(negative_prompt=[""] * len(batch_all_prompts), height=resolution, width=resolution)

            if "lumina" in model:
                pipe_kwargs.pop("height")
                pipe_kwargs.pop("width")

            start_time = time.time()
            images = pipe(list(map(lambda x: x[-1], batch_all_prompts)), **pipe_kwargs).images
            end_time = time.time()
            rprint(f"Image generation time: {end_time - start_time:.2f} seconds")

            for j, image in enumerate(images):
                total_generated += 1
                prompt_idx, augmentation_idx, llm_model_name, original_prompt, augmented_prompt = batch_all_prompts[j]
                generation_id = f"{prompt_idx}_{augmentation_idx}"
                output_image_path = output_folder / f"{generation_id}.jpg"

                while output_image_path.exists():
                    augmentation_idx += 1
                    generation_id = f"{prompt_idx}_{augmentation_idx}"
                    output_image_path = output_folder / f"{generation_id}.jpg"

                image.save(output_image_path)
                metadata = {
                    "prompt_index": prompt_idx,
                    "augmentation_idx": augmentation_idx,
                    "original_prompt": original_prompt,
                    "augmented_prompt": augmented_prompt,
                    "is_augmented": original_prompt != augmented_prompt,
                    "model_name": model,
                    "llm_model_name": llm_model_name,
                    "height": pipe_kwargs.get("height"),
                    "width": pipe_kwargs.get("width"),
                    "input_file": data_path.stem,
                    "hostname": hostname,
                    "image_path": str(output_image_path),
                    "gpu_name": gpu_name,
                    "generation_timestamp": datetime.now().isoformat(),
                }

                metadata_file = output_image_path.with_suffix(".json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

        rprint(f"Generated {total_generated} prompts out of total {len(all_prompts)} prompts")

    exit()


def tail_log_file(log_file_path, glob_str=None):
    import subprocess
    import time

    max_retries = 60
    retry_interval = 4
    for _ in range(max_retries):
        try:
            if (glob_str is None and Path(log_file_path).exists()) or len(list(Path(log_file_path).rglob(glob_str))) > 0:
                try:
                    if glob_str is None:
                        print(f"Tailing {log_file_path}")
                        proc = subprocess.Popen(["tail", "-f", "-n", "+1", f"{log_file_path}"], stdout=subprocess.PIPE)
                    else:
                        print(["tail", "-f", "-n", "+1", f"{log_file_path}/{glob_str}"])
                        proc = subprocess.Popen(["sh", "-c", f"tail -f -n +1 {log_file_path}/{glob_str}"], stdout=subprocess.PIPE)
                    for line in iter(proc.stdout.readline, b""):
                        print(line.decode("utf-8"), end="")
                except:
                    proc.terminate()
        except:
            print(f"Tried to glob: {log_file_path}, {glob_str}")
        finally:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")


# TODO: Set this if desired
cluster_node_gpus = {
    "main-node-0-0": "titanx",
}

def get_excluded_nodes(*args):
    return [x for x in cluster_node_gpus.keys() if any(s in cluster_node_gpus[x] for s in args)]


def run_slurm(data_path, num_chunks, num_workers, current_datetime, partition, chunk_size, extra_args, tail_log=False):
    print(f"Running slurm job with {num_chunks} chunks and {num_workers} workers...")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from simple_slurm import Slurm

    hostname = socket.gethostname()

    kwargs = dict()
    # TODO: Only needed if you wish to exclude specific bad nodes.
    if "main-node" in hostname:
        exclude = set(get_excluded_nodes())
        exclude.add("main-node-0-0")
        kwargs["exclude"] = ",".join(exclude)
        print(f"Excluding nodes: {kwargs['exclude']}")

    log_folder = Path("outputs/generate_images")
    log_folder.mkdir(parents=True, exist_ok=True)
    slurm = Slurm(
        "--requeue",
        job_name=f"generate_parallel_{data_path.stem}",
        cpus_per_task=8,
        mem="24g",
        export="ALL",
        gres=["gpu:1"],
        output=f"{str(log_folder)}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out",
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if "kate" in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=f"0-{num_chunks-1}%{num_workers}",
        partition=partition,
        comment="generate",
        **kwargs,
    )
    job_id = slurm.sbatch(
        f"python {Path(__file__).relative_to(os.getcwd())} {data_path} --is_slurm_task --slurm_task_datetme={current_datetime} --slurm_task_index=$SLURM_ARRAY_TASK_ID --chunk_size={chunk_size} {' '.join(extra_args)}"
    )
    print(f"Submitted job {job_id} with {num_chunks} tasks and {num_workers} workers...")
    if tail_log:
        tail_log_file(Path(f"outputs/generate_images"), f"{job_id}*")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    data_path: Path,
    num_workers: int = 1,
    use_slurm: bool = False,
    is_slurm_task: bool = False,
    slurm_task_datetme: str = None,
    slurm_task_index: int = None,
    max_chunk_size: int = 20000,
    num_chunks: Optional[int] = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    invalidate_cache: bool = False,
    partition: str = "all",
    chunk_size: Optional[int] = None,
    tail_log: bool = False,
):

    rprint(f"Running with data_path: {data_path}, args: {ctx.args}")
    default_values = dict(compile=False, batch_size=None, model="stabilityai/stable-diffusion-3-medium-diffusers", resolution=512, expected_samples_per_index=100, augment_prompts=True)
    for arg in ctx.args:
        if arg.removeprefix("--").split("=")[0] in default_values:
            default_values[arg.removeprefix("--").split("=")[0]] = arg.split("=")[1]
        else:
            assert False, f"Unknown argument: {arg}"

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((Path.home() / ".cache" / "torchinductor").resolve())
    current_datetime = datetime.now()
    datetime_up_to_hour = current_datetime.strftime("%Y_%m_%d_%H_00_00") if use_slurm else current_datetime.strftime("%Y_%m_%d_00_00_00")
    _timestamp = slurm_task_datetme if is_slurm_task else datetime_up_to_hour
    if invalidate_cache or use_slurm:
        _timestamp = current_datetime.strftime("%Y_%m_%d_%H_%M_00")

    dataset = get_to_process(_timestamp, data_path)

    if is_slurm_task:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if not slurm_job_id:
            raise RuntimeError("SLURM_JOB_ID environment variable not set")
        indices = get_indices_from_server(slurm_job_id, chunk_size, len(dataset), Path(root_output_folder) / data_path.stem, default_values["expected_samples_per_index"])
        if indices is None or len(indices) == 0:
            rprint(f"No images to process. Exiting...")
            exit()
        rprint(f"Running slurm task {slurm_task_index} with {len(indices)} images...")
        train(data_path, indices, **default_values)
        exit()

    submission_list = list(range(len(dataset)))
    if len(submission_list) == 0:
        rprint("No images to process. Exiting...")
        exit()

    if shuffle:
        import random

        random.seed(shuffle_seed)
        random.shuffle(submission_list)

    if chunk_size is None:
        chunk_size = min(len(submission_list) // num_workers, max_chunk_size)  # Adjust this based on the number of workers

    chunks = [submission_list[i : i + chunk_size] for i in range(0, len(submission_list), chunk_size)]
    assert sum([len(chunk) for chunk in chunks]) == len(submission_list)
    if len(chunks) > 999:
        rprint(f"Too many chunks ({len(chunks)}), truncating to 999...")
        chunks = chunks[:999]

    num_chunks = num_chunks if num_chunks is not None else len(chunks)

    if use_slurm:
        run_slurm(data_path, num_chunks, num_workers, datetime_up_to_hour, partition, chunk_size, tail_log=tail_log, extra_args=ctx.args)
        exit()
    else:
        import random
        random.shuffle(submission_list)

    with breakpoint_on_error():
        train(data_path, submission_list, **default_values)


if __name__ == "__main__":
    app()
