import os
import subprocess

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
class ComputeManager:
    def __init__(self):
        """
        Initializes the ComputeManager and identifies the GPU with the minimum memory usage.
        If no GPU is available, falls back to CPU mode (no CUDA_VISIBLE_DEVICES set).
        """
        self.device_id = self.get_min_memory_gpu()
        self.is_cpu = self.device_id is None

    def get_available_gpus(self):
        """Returns a list of available GPU indices as strings."""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,nounits,noheader'])
            return [gpu.strip() for gpu in output.decode('utf-8').strip().split('\n')]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def get_min_memory_gpu(self):
        """Identifies the GPU with the minimum used memory."""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
            gpu_memory_list = [int(memory) for memory in output.decode('utf-8').strip().split('\n')]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        if not gpu_memory_list:
            return None

        min_memory = min(gpu_memory_list)
        min_memory_gpu = gpu_memory_list.index(min_memory)

        return str(min_memory_gpu)

    def configure_gpu(self):
        """Configure the GPU for computing. Falls back to CPU if no GPU is available."""
        if self.device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
            print(f"Using GPU {self.device_id}.")
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print("No GPU available. Falling back to CPU.")

    def set_gpu(self, device_id: str):
        """
        Manually set the GPU device to use.

        Args:
            device_id (str): The device ID (e.g., '0' for the first GPU) to be used for computation.
        """
        available_gpus = self.get_available_gpus()
        if device_id not in available_gpus:
            raise ValueError(f"Invalid GPU id: {device_id}. Available GPUs: {', '.join(available_gpus)}")

        self.device_id = device_id
        self.is_cpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device_id
        print(f"Manually selected GPU {self.device_id} for computation.")

def test_compute_manager():
    cm = ComputeManager()
    print("Available GPUs:", cm.get_available_gpus())
    cm.configure_gpu()
    print("Running in CPU mode?", cm.is_cpu)

if __name__ == "__main__":
    test_compute_manager()
