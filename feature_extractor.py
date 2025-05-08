import os
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms, models

# === Configuration ===
EXPORT_DIR = './cifar_features'
BATCH_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(EXPORT_DIR, exist_ok=True)

# === Load Pretrained CNN (ResNet18 as Feature Extractor) ===
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remove final FC layer (output: 512-dim)
model = model.to(DEVICE).eval()

# === Preprocessing (match what ResNet expects) ===
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats
                         std=[0.229, 0.224, 0.225])
])

# === Load CIFAR-10 ===
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Extract and Save Features ===
for batch_idx, (images, labels) in enumerate(loader):
    images = images.to(DEVICE)
    with torch.no_grad():
        features = model(images)  # [B, 512]
    
    features = features.cpu().numpy().astype('float32')
    labels = labels.cpu().numpy().astype('uint8')

    # Write to binary: [label][f1][f2]...[f512] per sample
    filename = f"{EXPORT_DIR}/feat_batch_{batch_idx + 1}.bin"
    with open(filename, 'wb') as f:
        for i in range(len(labels)):
            f.write(bytes([labels[i]]))        # 1 byte label
            f.write(features[i].tobytes())     # 512 × float32 = 2048 bytes

    print(f"Wrote {filename}")
    if batch_idx == 4:  # Export 5 batches (~2500 samples)
        break

print("✅ Feature export complete!")
