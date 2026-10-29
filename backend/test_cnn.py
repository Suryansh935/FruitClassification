
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_DIR = Path(r"F:\dataset\test")
MODEL_PATH = Path("models/custom_cnn.pth")
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

if dataset.classes != class_names:
    raise ValueError(
        f"Class mismatch. Model: {class_names}; "
        f"Test dataset: {dataset.classes}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = CustomCNN(len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0
class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for label, prediction in zip(labels, predictions):
            index = label.item()
            class_total[index] += 1
            class_correct[index] += int(
                prediction.item() == index
            )

print(f"\nTest accuracy: {100 * correct / total:.2f}%")
print(f"Correct predictions: {correct}/{total}")

print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    accuracy = (
        100 * class_correct[i] / class_total[i]
        if class_total[i] else 0
    )
    print(
        f"{name}: {accuracy:.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )
