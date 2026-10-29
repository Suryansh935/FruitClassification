
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = r"F:\dataset\test"
MODEL_PATH = os.path.join("models", "freshness_model.pth")
IMAGE_SIZE = 128
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class FreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.samples = []

        mapping = {
            "freshapples": 0,
            "freshbanana": 0,
            "freshoranges": 0,
            "rottenapples": 1,
            "rottenbanana": 1,
            "rottenoranges": 1,
        }

        for path, _ in self.dataset.samples:
            folder = os.path.basename(os.path.dirname(path)).lower()
            if folder in mapping:
                self.samples.append((path, mapping[folder]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class FreshnessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

dataset = FreshnessDataset(TEST_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = FreshnessCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_names = ["fresh", "rotten"]
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

correct = sum(p == y for p, y in zip(all_preds, all_labels))
accuracy = 100 * correct / len(all_labels)

print(f"\nTest images: {len(all_labels)}")
print(f"Test accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))
