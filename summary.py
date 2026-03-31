import torch
from torchinfo import summary
import os

# 1. Define the architecture (Must match cnn.py exactly)
import torch.nn as nn

class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _ConvBnRelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

class WeatherCNN(nn.Module):
    def __init__(self, in_channels=42):
        super().__init__()
        self.stem   = _ConvBnRelu(in_channels, 64)
        self.stage1 = _ResBlock(64,  128, stride=2)
        self.stage2 = _ResBlock(128, 256, stride=2)
        self.stage3 = _ResBlock(256, 256, stride=2)
        self.stage4 = _ResBlock(256, 512, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.cont_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 6),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return self.cont_head(x), self.bin_head(x).squeeze(1)

# 2. Setup Paths
MODEL_PATH = "/cluster/tufts/c26sp1cs0137/ashen05/best_model_20260324_210752.pt"

# 3. Load and Summarize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeatherCNN(in_channels=42).to(device)

if os.path.exists(MODEL_PATH):
    print(f"Loading weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Weight file not found, showing architecture only.")

print("\n" + "="*80)
print("WEATHER CNN ARCHITECTURE SUMMARY")
print("="*80)

# Input size: (Batch, Channels, Height, Width)
summary(model, input_size=(1, 42, 352, 352), device=device)