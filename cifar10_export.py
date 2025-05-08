import os
import numpy as np
from torchvision import datasets, transforms
import torch
# Configuration
EXPORT_DIR = './cifar_data'  # Match the path in your CUDA code
BATCH_SIZE = 512
os.makedirs(EXPORT_DIR, exist_ok=True)

# No normalization (CUDA code scales to [0,1] manually)
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle for reproducibility

# Export as binary files (mimic original CIFAR-10 format)
for batch_idx, (images, labels) in enumerate(loader):
    filename = f"{EXPORT_DIR}/data_batch_{batch_idx + 1}.bin"  # data_batch_1.bin, etc.
    
    # Rescale to [0, 255] and convert to uint8
    images_uint8 = (images * 255).numpy().astype('uint8')
    labels_uint8 = labels.numpy().astype('uint8')
    
    # Write to binary file: [label, R, G, B, R, G, B, ...] per image
    with open(filename, 'wb') as f:
        for i in range(BATCH_SIZE):
            f.write(bytes([labels_uint8[i]]))  # Label (1 byte)
            f.write(images_uint8[i].tobytes())  # Image data (3072 bytes)
    
    if batch_idx == 4:  # Create 5 files (like original CIFAR-10)
        break

print(f"Exported binary batches to: {EXPORT_DIR}")