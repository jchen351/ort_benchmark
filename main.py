import sys
import time
import onnxruntime
import numpy as np
import psutil
# Set the random seed
np.random.seed(0)

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

onnx_model_path = 'poc.onnx'

# Load the ONNX model with the CPUExecutionProvider
sess_options = onnxruntime.SessionOptions()
sess_options.enable_profiling = True

ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'],sess_options=sess_options)
ort_session.get_modelmeta()
inputs = ort_session.get_inputs()
ort_session.get_session_options()

nth = 1000

# Warm-up inference to cache optimizations
warmup_input = np.random.randn(1, 3, 224, 224).astype('f')
warmup_dict = {inputs[0].name: warmup_input}
ort_session.run(None, warmup_dict)
# Monitor CPU and Memory during inference
pid = psutil.Process().pid
avg_cpu, avg_memory = monitor_process(pid, duration=5)
# Warm-up runs (optional, to stabilize performance)
for _ in range(1):
    ort_session.run(None, warmup_dict)

# Measure inference time excluding input creation
total_time_ns = 0
for _ in range(nth):
    np_input = np.random.randn(1, 3, 224, 224).astype('f')
    input_ = {inputs[0].name: np_input}

    start_ns = time.perf_counter_ns()
    ort_session.run(None, input_)
    end_ns = time.perf_counter_ns()

    total_time_ns += end_ns - start_ns

avg_time_ns = total_time_ns / nth
avg_time_ms = avg_time_ns / 1e6

print(f'# Python{sys.version_info.major}.{sys.version_info.minor}\n'
      f'# Onnxruntime:[{onnxruntime.__version__}] Average inference time: {avg_time_ms:.3f} ms')
print(f"# CPU Usage: {avg_cpu:.2f}%")
print(f"# Memory Usage: {avg_memory:.2f} MB")

'''
# Python3.8
# Onnxruntime:[1.14.1] Average inference time: 3.748 ms
# CPU Usage: 0.68%
# Memory Usage: 59.34 MB

# Onnxruntime:[1.15.1] Average inference time: 4.160 ms
# CPU Usage: 0.58%
# Memory Usage: 63.38 MB

# Onnxruntime:[1.16.1] Average inference time: 4.178 ms
# CPU Usage: 0.56%
# Memory Usage: 64.16 MB

# Onnxruntime:[1.17.3] Average inference time: 4.225 ms
# CPU Usage: 0.54%
# Memory Usage: 68.86 MB


'''