import torch
import torchvision.models as models
import time

# ── 1. Check CUDA ──────────────────────────────────────────────
print("=== CUDA Info ===")
print(f"CUDA available     : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA not available. Check your PyTorch installation.")
    exit()

print(f"Device name        : {torch.cuda.get_device_name(0)}")
print(f"VRAM total         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch version    : {torch.__version__}")
print()

device = torch.device("cuda")

# ── 2. Load VGG16 onto GPU ─────────────────────────────────────
print("=== Loading VGG16 ===")
model = models.vgg16(weights="IMAGENET1K_V1")
model.eval()
model = model.to(device)
print(f"Model on device    : {next(model.parameters()).device}")
print()

# ── 3. Speed test — CPU vs GPU ─────────────────────────────────
print("=== Speed Test (100 forward passes, batch of 16) ===")
dummy_input = torch.randn(16, 3, 224, 224)  # fake batch of 16 images

# CPU
model_cpu = model.to("cpu")
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model_cpu(dummy_input)
cpu_time = time.time() - start
print(f"CPU time           : {cpu_time:.2f}s")

# GPU
model_gpu = model.to(device)
dummy_input_gpu = dummy_input.to(device)

# Warmup (first GPU call is always slow)
with torch.no_grad():
    _ = model_gpu(dummy_input_gpu)
torch.cuda.synchronize()

start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model_gpu(dummy_input_gpu)
torch.cuda.synchronize()  # wait for GPU to finish
gpu_time = time.time() - start
print(f"GPU time           : {gpu_time:.2f}s")
print(f"Speedup            : {cpu_time / gpu_time:.1f}x faster on GPU")
print()
print("✅ CUDA is working correctly!")