# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from .config import BaseQueueConf

class DummyLogger:
    def __init__(self):
        pass

    def debug(self, message):
        print(f"DEBUG: {message}")

    def info(self, message):
        print(f"INFO: {message}")

    def warning(self, message):
        print(f"WARNING: {message}")

    def error(self, message):
        print(f"ERROR: {message}")

    def critical(self, message):
        print(f"CRITICAL: {message}")

log = DummyLogger()

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
                        proc = subprocess.Popen(['tail', '-f', "-n", "+1", f"{log_file_path}"], stdout=subprocess.PIPE)
                    else:
                        print(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"])
                        proc = subprocess.Popen(['sh', '-c', f'tail -f -n +1 {log_file_path}/{glob_str}'], stdout=subprocess.PIPE)
                    for line in iter(proc.stdout.readline, b''):
                        print(line.decode('utf-8'), end='')
                except:
                    proc.terminate()
        except:
            print(f"Tried to glob: {log_file_path}, {glob_str}")
        finally:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")

class BaseSubmititLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(self, **params: Any) -> None:
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.sweep_configs: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
        sweep_keys: Optional[List[str]] = None,
    ) -> JobReturn:
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()
        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            # Populate new job variables
            job.id = submitit.JobEnvironment().job_id  # type: ignore
            sweep_config.hydra.job.num = job_num
            sweep_config.hydra.job.sweep_keys = sweep_keys

        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Resubmit the current callable at its current state with the same initial arguments."""
        # lazy import to ensure plugin discovery remains fast
        import submitit

        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int, sweep_keys=None
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout", "python", "python_suffix"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                    [] if sweep_keys is None else list(sweep_keys),
                )
            )

        jobs = executor.map_array(self, *zip(*job_params))
        # tail_log_file(str(Path(jobs[0].paths.stdout).parent.parent), "**/*.out")
        return [j.results()[0] for j in jobs]


class LocalLauncher(BaseSubmititLauncher):
    _EXECUTOR = "local"


class SlurmLauncher(BaseSubmititLauncher):
    _EXECUTOR = "slurm"
