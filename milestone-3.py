# Generated from: milestone-3.ipynb
# Converted at: 2026-03-15T16:38:38.149Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import math
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(32, 1, 64, 64)

x = x.view(32, -1)  # flatten
print(x.shape)      # (32, 4096)

fc = nn.Linear(4096, 128)


# 1 second of fake audio at 16kHz
sample_rate = 16000
waveform = torch.randn(1, sample_rate)

# Mel Spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=64
)

mel_spec = mel_transform(waveform)

print("Mel Spectrogram shape:", mel_spec.shape)

x = torch.randn(1, 1, 64, 64)

# Define layers
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Forward pass
out = conv(x)
print("After Conv2d:", out.shape)

out = pool(out)
print("After MaxPool2d:", out.shape)

layer = nn.Linear(128, 64)

total_params = sum(p.numel() for p in layer.parameters())
print("Total parameters:", total_params)

data = torch.randn(1050, 10)
dataset = TensorDataset(data)

loader = DataLoader(dataset, batch_size=32, drop_last=False)

print("Number of batches:", len(loader))

# Manual calculation
print("Manual calculation:", math.ceil(1050 / 32))

logits = torch.tensor([[2.5, 1.0, 0.1]])
target = torch.tensor([0])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print("Loss:", round(loss.item(), 3))

w = torch.tensor([0.5], requires_grad=True)

# Manually set gradient
w.grad = torch.tensor([0.2])

# SGD update
learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad

print("Updated weight:", w.item())

x = torch.randn(1, 1, 64, 64)

# Conv layer with kernel_size=5, stride=1, padding=2
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)

out = conv(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)

correct = 170
total = 200

accuracy = correct / total

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)