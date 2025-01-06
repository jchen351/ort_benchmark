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

Grouped by operator 1.14.1
----------------------------------------------------------------
Total(μs)       Time%   Kernel(μs)      Kernel% Calls   AvgKernel(μs)   Fence(μs)       Operator
   2103309      57.08       2103227     57.08   19038            110.5          82      Conv
    425084      11.54        425035     11.54    9018             47.1          49      FusedConv
    371402      10.08        371385     10.08    4008             92.7          17      Selu
    290047       7.87        290012      7.87    9018             32.2          35      Celu
    180409       4.90        180393      4.90    4008             45.0          16      MaxPool
    138644       3.76        138585      3.76    8016             17.3          59      Concat
     91412       2.48         91402      2.48    2004             45.6          10      Elu
     63492       1.72         63477      1.72    2004             31.7          15      Softmax
     11954       0.32         11954      0.32    1002             11.9           0      GlobalAveragePool
      8919       0.24          8919      0.24    1002              8.9           0      LeakyRelu


Grouped by operator 1.15.1
----------------------------------------------------------------
Total(μs)       Time%   Kernel(μs)      Kernel% Calls   AvgKernel(μs)   Fence(μs)       Operator
   2084293      51.11       2084207     51.12   18036            115.6          86      Conv
    732816      17.97        732788     17.97    4008            182.8          28      Selu
    487019      11.94        486936     11.94   10020             48.6          83      FusedConv
    304087       7.46        304039      7.46    9018             33.7          48      Celu
    177066       4.34        177045      4.34    4008             44.2          21      MaxPool
    137296       3.37        137211      3.37    8016             17.1          85      Concat
     85715       2.10         85702      2.10    2004             42.8          13      Elu
     57731       1.42         57717      1.42    2004             28.8          14      Softmax
     11671       0.29         11664      0.29    1002             11.6           7      GlobalAveragePool


Grouped by operator 1.17.3
----------------------------------------------------------------
Total(μs)       Time%   Kernel(μs)      Kernel% Calls   AvgKernel(μs)   Fence(μs)       Operator
   2024386      49.84       2024309     49.84   18036            112.2          77      Conv
    724966      17.85        724945     17.85    4008            180.9          21      Selu
    459986      11.33        459923     11.32   10020             45.9          63      FusedConv
    397710       9.79        397675      9.79    9018             44.1          35      Celu
    182745       4.50        182736      4.50    4008             45.6           9      MaxPool
    137003       3.37        136959      3.37    8016             17.1          44      Concat
     85255       2.10         85234      2.10    2004             42.5          21      Elu
     37964       0.93         37946      0.93    2004             18.9          18      Softmax
     11532       0.28         11525      0.28    1002             11.5           7      GlobalAveragePool


Grouped by operator 1.7.3
----------------------------------------------------------------
Total(μs)       Time%   Kernel(μs)      Kernel% Calls   AvgKernel(μs)   Fence(μs)       Operator
   2084892      51.06       2084795     51.06   18036            115.6          97      Conv
    735250      18.01        735235     18.01    4008            183.4          15      Selu
    464653      11.38        464592     11.38   10020             46.4          61      FusedConv
    319560       7.83        319535      7.83    9018             35.4          25      Celu
    202089       4.95        202070      4.95    4008             50.4          19      MaxPool
    139045       3.41        138995      3.40    8016             17.3          50      Concat
     87017       2.13         87010      2.13    2004             43.4           7      Elu
     38968       0.95         38950      0.95    2004             19.4          18      Softmax
     11618       0.28         11612      0.28    1002             11.6           6      GlobalAveragePool


'''