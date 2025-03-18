import logging
from pathlib import Path
from typing import Optional

import os
import time
import atexit

class DebugLogger:
    def __init__(self, log_name=None, file_name=None, log_dir="/dev/shm", buffer_size=100, flush_interval=1.0, add_user_prefix=True):
        """
        Initializes the logger.

        :param identifier: Optional; Unique identifier for the log file (e.g., PID or custom string).
                           If None, uses the current process ID.
        :param log_dir: Directory where log files are stored.
        :param buffer_size: Number of messages to buffer before writing to file.
        :param flush_interval: Time interval (in seconds) to flush logs if buffer isn't full.
        """

        if file_name is None:
            file_name = f"pid_{os.getpid()}"

        self.log_name = log_name
        self.log_dir = Path(log_dir)
        if add_user_prefix:
            self.log_dir = self.log_dir / os.getenv("USER")

        self.log_dir = self.log_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        prefix_timestamp = time.strftime("%Y%m%d_%H%M")
        suffix_timestamp = f"{int(time.time())}_{int(time.time_ns() % 1_000_000_000)}"
        self.log_file_path = os.path.join(self.log_dir, f"{prefix_timestamp}_{file_name}_{suffix_timestamp}.out")
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.last_flush_time = time.time()
        self.file = open(self.log_file_path, 'a', buffering=1)  # Line-buffered
        atexit.register(self.close)

    def log(self, *args, sep=' ', end='\n', flush=False, **kwargs):
        """
        Adds a log message to the buffer.

        Supports arbitrary positional and keyword arguments like the built-in print() function.

        :param args: Arbitrary positional arguments to be logged.
        :param sep: String inserted between values, default a space.
        :param end: String appended after the last value, default a newline.
        :param kwargs: Additional keyword arguments (ignored but accepted for compatibility).
        """
        message = sep.join(str(arg) for arg in args) + end
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp = f"{timestamp}, {self.log_name}" if self.log_name is not None else timestamp
        formatted_message = f"[{timestamp}] {message}".rstrip('\n')
        self.buffer.append(formatted_message)

        if len(self.buffer) >= self.buffer_size:
            self.flush()
        elif (time.time() - self.last_flush_time) >= self.flush_interval:
            self.flush()

    def flush(self):
        """
        Writes buffered log messages to the file and clears the buffer.
        """
        if len(self.buffer) > 0:
            try:
                self.file.write('\n'.join(self.buffer) + '\n')
                self.file.flush()
                os.fsync(self.file.fileno())  # Ensure it's written to disk
                self.buffer.clear()
                self.last_flush_time = time.time()
            except Exception as e:
                print(f"Failed to write logs: {e}")

    def close(self):
        """
        Flushes any remaining logs and closes the file.
        """
        self.flush()
        if not self.file.closed:
            self.file.close()

logger: Optional[logging.Logger] = None
file_only_logger: Optional[logging.Logger] = None  # New global variable
memory_logger: Optional[DebugLogger] = None

def set_logger(name: str, log_file_path: Optional[str] = None):
    global logger, file_only_logger, memory_logger
    if logger is not None and logger.hasHandlers():
        logger.handlers.clear()

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_format = "[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file_path:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        file_only_logger = logging.getLogger(name + f"_")
        file_only_logger.handlers = []
        file_only_logger.setLevel(logging.DEBUG)
        file_only_logger.addHandler(file_handler)
        file_only_logger.propagate = False

    if memory_logger is not None:
        memory_logger.close()
    memory_logger = DebugLogger(log_name=name, file_name=Path(log_file_path).stem if log_file_path is not None else None)

def get_logger():
    return logger

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method

def get_logger_(main_process_only: bool) -> logging.Logger:
    global logger
    from decoupled_utils import get_rank, is_main_process
    if is_main_process() or not main_process_only:
        if logger is not None:
            return logger
        else:
            return set_logger(__name__ + f"_rank_{get_rank()}")
    else:
        return Dummy()
    
def combine_args(*args):
    return " ".join((str(arg) for arg in args))

def _always_debug_log(*args, **kwargs) -> logging.Logger:
    from decoupled_utils import is_main_process
    if not is_main_process() and file_only_logger is not None:
        file_only_logger.debug(combine_args(*args), **kwargs)

    log_memory(*args, **kwargs)

def log_debug(*args, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(combine_args(*args), **kwargs)
    get_logger_(main_process_only=main_process_only).debug(combine_args(*args), **kwargs)


def log_info(*args, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(combine_args(*args), **kwargs)
    get_logger_(main_process_only=main_process_only).info(combine_args(*args), **kwargs)


def log_error(*args, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(combine_args(*args), **kwargs)
    get_logger_(main_process_only=main_process_only).error(combine_args(*args), **kwargs)


def log_warn(*args, main_process_only: bool = True, **kwargs):
    kwargs.pop("end", None)
    if main_process_only: _always_debug_log(combine_args(*args), **kwargs)
    get_logger_(main_process_only=main_process_only).warning(combine_args(*args), **kwargs)

def log_memory(*args, **kwargs):
    kwargs.pop("end", None)
    if memory_logger is not None:
        memory_logger.log(combine_args(*args), **kwargs)