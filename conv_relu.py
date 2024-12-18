import sys
import psutil
import onnxruntime as ort
import numpy as np
import time

# Set the random seed
np.random.seed(0)

session_options = ort.SessionOptions()
session_options.enable_profiling = True  # Enable profiling
# Load the ONNX model
session = ort.InferenceSession("simple_conv_relu.onnx",session_options)

inputs = session.get_inputs()
warmup_input = np.random.randn(1, 3, 224, 224).astype('f')
warmup_dict = {inputs[0].name: warmup_input}
session.run(None, warmup_dict)
# Generate random input data
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
# Monitor CPU and memory usage

def monitor_process(pid, duration=5):
    process = psutil.Process(pid)
    cpu_usage = []
    memory_usage = []

    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_usage.append(process.cpu_percent(interval=0.1))
        memory_usage.append(process.memory_info().rss / (1024 ** 2))  # Memory in MB

    avg_cpu = sum(cpu_usage) / len(cpu_usage)
    avg_memory = sum(memory_usage) / len(memory_usage)
    return avg_cpu, avg_memory

# Warm-up runs (optional, to stabilize performance)
for _ in range(10):
    session.run(None, {"input": input_data})

# Measure inference time
nth = 1000
total_time_ns= 0
start_ns = time.perf_counter_ns()

# Monitor CPU and Memory during inference
pid = psutil.Process().pid
avg_cpu, avg_memory = monitor_process(pid, duration=5)

for _ in range(nth):
    np_input = np.random.randn(1, 3, 224, 224).astype('f')
    input_ = {inputs[0].name: np_input}

    end_ns = time.perf_counter_ns()
    outputs = session.run(None, input_)
    total_time_ns += end_ns - start_ns

avg_time_ns = total_time_ns / nth
avg_time_ms = avg_time_ns / 1e6

print(f'# Python{sys.version_info.major}.{sys.version_info.minor} numpy: {np.__version__}\n'
      f'# Onnxruntime:[{ort.__version__}] Average inference time: {avg_time_ms:.3f} ms')
print(f"# CPU Usage: {avg_cpu:.2f}%")
print(f"# Memory Usage: {avg_memory:.2f} MB")


# Average latency per inference: 27.9542 ms
# Average latency per inference: 37.0142 ms

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.17.3] Average inference time: 5118.047 ms
# CPU Usage: 0.43%
# Memory Usage: 46.72 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.14.1] Average inference time: 5095.924 ms
# CPU Usage: 0.53%
# Memory Usage: 45.37 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.17.3] Average inference time: 5386.151 ms
# CPU Usage: 0.36%
# Memory Usage: 47.57 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.14.1] Average inference time: 17809.714 ms
# CPU Usage: 0.60%
# Memory Usage: 64.52 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.17.3] Average inference time: 17911.368 ms
# CPU Usage: 0.57%
# Memory Usage: 61.03 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.14.1] Average inference time: 17814.377 ms
# CPU Usage: 0.71%
# Memory Usage: 54.53 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.15.1] Average inference time: 17968.825 ms
# CPU Usage: 0.40%
# Memory Usage: 70.91 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.16.1] Average inference time: 17729.820 ms
# CPU Usage: 0.81%
# Memory Usage: 61.30 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.17.3] Average inference time: 17740.959 ms
# CPU Usage: 0.60%
# Memory Usage: 57.36 MB

#-----
# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.16.1] Average inference time: 6382.194 ms
# CPU Usage: 0.73%
# Memory Usage: 55.69 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.17.3] Average inference time: 6366.145 ms
# CPU Usage: 0.64%
# Memory Usage: 56.17 MB


# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.15.1] Average inference time: 6398.729 ms
# CPU Usage: 0.52%
# Memory Usage: 56.28 MB

# Python3.8 numpy: 1.24.4
# Onnxruntime:[1.14.1] Average inference time: 6358.798 ms
# CPU Usage: 0.91%
# Memory Usage: 53.60 MB



