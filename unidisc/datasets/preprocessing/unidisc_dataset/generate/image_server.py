import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from decoupled_utils import rprint

app = Flask(__name__)

job_allocations = {}
completed_indices = set()
first_request = True

allocations_file = Path("static/allocations.json")
memory = joblib.Memory("static/.cache", verbose=0)
cache_duration = timedelta(minutes=60)

@memory.cache
def load_completed_indices(output_dir, expected_samples_per_index):
    rprint(f"Loading completed indices from {output_dir}")
    completed_files = output_dir.glob("*.json")
    index_count = defaultdict(int)
    for f in completed_files:
        index = int(f.stem.split("_")[0])
        index_count[index] += 1
    rprint(f"Finsihed loading completed indices: {index_count}")
    return {index for index, count in index_count.items() if count >= (expected_samples_per_index - 2)}


def get_running_jobs():
    try:
        result = subprocess.run(["squeue", "-h", "-o", "%A"], capture_output=True, text=True)
        job_ids = result.stdout.split()
        return set(job_ids)
    except subprocess.CalledProcessError as e:
        rprint(f"Failed to run squeue: {e}")
        return set()


def save_allocations():
    allocations_file.parent.mkdir(parents=True, exist_ok=True)
    with allocations_file.open("w") as f:
        json.dump(
            {
                k: {"indices": list(v["indices"]), "timestamp": v["timestamp"].isoformat(), "dataset_key": v["dataset_key"]}
                for k, v in job_allocations.items()
            },
            f,
        )


def load_allocations():
    if allocations_file.exists():
        with allocations_file.open("r") as f:
            data = json.load(f)
            return {
                k: {"indices": set(v["indices"]), "timestamp": datetime.fromisoformat(v["timestamp"]), "dataset_key": v["dataset_key"]}
                for k, v in data.items()
            }
    return {}


job_allocations = load_allocations()


@app.route("/get_indices", methods=["POST"])
def get_indices():
    global last_cache_time

    slurm_job_id = request.json.get("slurm_job_id", None)
    chunk_size = request.json.get("chunk_size", None)
    total_indices = request.json.get("total_indices", None)
    output_dir_path = request.json.get("output_dir", None)
    expected_samples_per_index = int(request.json.get("expected_samples_per_index", 100))

    if not slurm_job_id or not output_dir_path or not chunk_size or not total_indices:
        return jsonify({"error": "SLURM job ID and output directory are required"}), 400

    output_dir = Path(output_dir_path)
    dataset_key = output_dir.stem
    current_time = datetime.now()

    if "last_cache_time" not in globals() or current_time - last_cache_time > cache_duration:
        load_completed_indices.clear()
        last_cache_time = current_time

    completed_indices = set()
    running_jobs = get_running_jobs()

    n_hours = 12
    threshold_time = datetime.now() - timedelta(hours=n_hours)
    for job_id in list(job_allocations.keys()):
        if job_id not in running_jobs or job_allocations[job_id]["timestamp"] < threshold_time:
            rprint(f"Deleting job {job_id} from allocations")
            del job_allocations[job_id]

    if slurm_job_id in job_allocations:
        allocated_indices = job_allocations[slurm_job_id]["indices"]
        num_available_indices = None
        rprint(f"Job {slurm_job_id} already allocated indices: {allocated_indices}")
    else:
        all_reserved_indices = {idx for indices in job_allocations.values() for idx in indices["indices"]}
        available_indices = set(range(0, total_indices)) - completed_indices - all_reserved_indices

        if not available_indices:
            rprint(f"No indices available for job {slurm_job_id}, completed len: {len(completed_indices)}, running jobs: {len(running_jobs)}, current allocations: {len(job_allocations)}, expected samples per index: {expected_samples_per_index}, output dir: {output_dir}")
            return jsonify({"error": "No indices available"}), 404
        
        available_indices = list(available_indices)
        random.shuffle(available_indices)
        allocated_indices = set(available_indices[:chunk_size])
        job_allocations[slurm_job_id] = {"indices": allocated_indices, "timestamp": datetime.now(), "dataset_key": dataset_key}
        save_allocations()
        num_available_indices = len(available_indices)

    rprint(
        f"Dataset {dataset_key}, total len: {total_indices}, chunk size: {chunk_size}, completed len: {len(completed_indices)}, available len: {num_available_indices}, running jobs: {len(running_jobs)}, current allocations: {len(job_allocations)}"
    )
    rprint(f"Job {slurm_job_id} allocated indices: {allocated_indices}")

    return jsonify({"indices": list(allocated_indices)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)