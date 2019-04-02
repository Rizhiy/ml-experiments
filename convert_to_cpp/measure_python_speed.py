import time

import torch
from convert_to_onnx import SoftMaxModel
from torchvision.models import resnet18
from tqdm import tqdm

cuda_batch_size = 32
cpu_batch_size = 32

num_cuda_images = 8192
num_cpu_images = 256

model = SoftMaxModel(resnet18(pretrained=True))

# Measure CPU speed
start_time = time.time()
for _ in tqdm(range(0, num_cpu_images, cpu_batch_size), desc="Measuring CPU speed"):
    batch = torch.randn(cpu_batch_size, 3, 244, 244)
    _ = model(batch)
print(f"CPU FPS = {num_cpu_images / (time.time() - start_time):.3f}")

# Measure GPU speed
model = model.cuda()
start_time = time.time()
for _ in tqdm(range(0, num_cuda_images, cuda_batch_size), desc="Measuring GPU speed"):
    batch = torch.randn(cuda_batch_size, 3, 244, 244, device='cuda')
    _ = model(batch)
print(f"GPU FPS = {num_cuda_images / (time.time() - start_time):.3f}")
