import torch
import torchvision
import torchaudio
import numpy as np
import PIL

print(f"Torch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

print(f"numpy Version: {np.__version__}")
print(f"PIL Version: {PIL.__version__}")