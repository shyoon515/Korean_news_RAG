import os
import logging
from datetime import datetime
from pathlib import Path

import subprocess
import time
import requests
import signal

def setup_logger(name, log_dir=Path(__file__).resolve().parent.parent / "logs", subdirectory=''):
    """Set up Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    datetime_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_log_dir = os.path.join(log_dir, subdirectory, datetime_folder)
    os.makedirs(full_log_dir, exist_ok=True)

    log_filename = f"{name}.log"
    
    file_handler = logging.FileHandler(os.path.join(full_log_dir, log_filename))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

class VLLMServerManager:
    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 8000,
        max_model_len: int = 4096,
        gpu_mem_util: float = 0.8,
        dtype: str = "float16",
        env: dict | None = None,
        logger = None
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.max_model_len = max_model_len
        self.gpu_mem_util = gpu_mem_util
        self.dtype = dtype
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.logger = logger

    def start(self, timeout_s: int = 120):
        if self.logger:
            self.logger.info(f"[VLLMServerManager] Starting vLLM server for model: {self.model} on {self.host}:{self.port}")

        if self.proc is not None and self.proc.poll() is None:
            if self.logger:
                self.logger.info(f"[VLLMServerManager] vLLM server is already running.")
            return  # already running

        cmd = [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_mem_util),
            "--max-model-len", str(self.max_model_len),
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self.env,
        )

        # wait until /v1/models responds
        if self.logger:
            self.logger.info(f"[VLLMServerManager] Waiting for vLLM server to be ready...")
        start_t = time.time()
        while time.time() - start_t < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError("vLLM server exited early while starting.")
            try:
                r = requests.get(f"{self.base_url}/models", timeout=2)
                if r.status_code == 200:
                    if self.logger:
                        self.logger.info(f"[VLLMServerManager] vLLM server is ready.")
                    return
            except Exception:
                pass
            time.sleep(0.5)

        if self.logger:
            self.logger.error(f"[VLLMServerManager] vLLM server did not become ready in time.")
        raise TimeoutError("vLLM server did not become ready in time.")

    def stop(self, kill_after_s: int = 10):
        if self.logger:
            self.logger.info(f"[VLLMServerManager] Stopping vLLM server for model: {self.model}")
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self.proc = None
            return

        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=kill_after_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None
