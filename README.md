# GPU Compute Manager

🧠 Lightweight GPU device manager for Python projects.

## Features
- 🔍 Automatically selects the GPU with the lowest memory usage
- 🧼 Falls back to CPU if no GPU is detected
- ⚙️ Allows manual GPU selection
- 🧪 Test function for quick validation

## Installation
```bash
pip install .
```

## Basic Usage
```python
from compute_manager import ComputeManager

cm = ComputeManager()
cm.configure_gpu()

if cm.is_cpu:
    print("Running on CPU")
else:
    print(f"Running on GPU {cm.device_id}")
```

## Advanced Usage

### Manually Set GPU
```python
cm = ComputeManager()
cm.set_gpu("1")  # use GPU with index 1
cm.configure_gpu()
```

### List All Available GPUs
```python
available = cm.get_available_gpus()
print("Available GPUs:", available)
```

### Use in Training Workflow
```python
cm = ComputeManager()
cm.configure_gpu()

# Example with PyTorch
import torch

device = torch.device("cuda" if not cm.is_cpu else "cpu")
tensor = torch.rand(1000).to(device)
```

## CLI Test
You can run the built-in test by executing:
```bash
python compute_manager.py
```

## License
MIT License
