
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

DATASET_DIR = Path(r"F:\dataset\test")
MODEL_PATH = Path("models/model1_fruit.pth")
IMAGE_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
class_names = checkpoint["class_names"]

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

if dataset.classes != class_names:
    raise ValueError(
        f"Class mismatch. Model: {class_names}; "
        f"Test dataset: {dataset.classes}"
    )

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = models.mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT
)
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

correct = 0
total = 0
class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for label, prediction in zip(labels, predictions):
            index = label.item()
            class_total[index] += 1
            class_correct[index] += int(prediction.item() == index)

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
