import subprocess
import socket
import os
REQUEUE_COMMAND = "scontrol requeue JobId={job_id}; scontrol update JobId={job_id} ExcNodeList={exclude_list}"

def get_hostname() -> str:
    return socket.gethostname()

def is_cuda_available() -> bool:
    try:  # test if CUDA is available
        import torch

        test_tensor = torch.rand(5, 3, device="cuda")
        test_tensor.requires_grad = True
        (test_tensor**2).sum().backward()
        return True

    except RuntimeError as e:
        print(f"Cuda test failed: {e}")

    return False


def get_current_exclude_list(job_id: str) -> list:
    try:  # get the current exclude list
        exclude_list = []
        output = subprocess.run(["scontrol", "show", "job", job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

        for line in output.splitlines():
            if "ExcNodeList=" in line:
                for host in line.split("ExcNodeList=")[1].split()[0].split(","):

                    if host != "(null)":
                        exclude_list.append(host)

    except subprocess.CalledProcessError as e:
        print(f"Failed to retrieve exclude list for job {job_id}: {e}")

    return exclude_list


def requeue_job_excluding_node(job_id: str, bad_node: str) -> bool:
    try:  # stop and requeue the job excluding the bad node
        exclude_list = get_current_exclude_list(job_id)
        exclude_list.append(bad_node)
        exclude_list = ",".join(exclude_list)
        subprocess.run(REQUEUE_COMMAND.format(job_id=job_id, exclude_list=exclude_list), shell=True, check=True)
        print(f"Requeued job {job_id}, excluding {bad_node}.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to requeue job {job_id}: {e}")

    return False

def check_requeue() -> bool:
    job_id = os.getenv("SLURM_JOB_ID", None)
    if not job_id:
        return False

    if not (cuda_available := is_cuda_available()):
        print("Attempting to requeue job.")
        requeue_job_excluding_node(job_id, get_hostname())
    else:
        print("CUDA is available. Proceeding with the job.")

    return cuda_available

if __name__ == "__main__":
    check_requeue()
