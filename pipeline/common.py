import os
import logging
from datetime import datetime

import torch

# GPU Setup
def get_free_gpu():
    """In Linux kernel, type export CUDA_VISIBLE_DEVICES='[list of gpu numbers]'"""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return None

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if cuda_visible_devices:
        gpu_ids = [int(x) for x in cuda_visible_devices.split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {gpu_ids}")
    
    if not gpu_ids:
        print("No GPUs available.")
        return None

    free_memory = []
    for i, gpu_id in enumerate(gpu_ids):
        try:
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_memory.append(free_mem)
            print(f"GPU {gpu_id}: {free_mem/1024**3:.2f} GB free / {total_memory/1024**3:.2f} GB total")
        except Exception as e:
            print(f"GPU {gpu_id}: Unable to query device properties - {str(e)}")
            free_memory.append(0)

    if not free_memory or max(free_memory) == 0:
        print("Unable to query GPU memory or all GPUs are full.")
        return None

    selected_index = free_memory.index(max(free_memory))
    selected_gpu = gpu_ids[selected_index]
    print(f"Selected GPU: {selected_gpu}")
    
    return selected_index

def setup_logger(name, log_dir, subdirectory=''):
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