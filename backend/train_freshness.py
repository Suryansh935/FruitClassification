
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

TRAIN_DIR = Path(r"F:\dataset\train")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map the original six classes into two freshness classes.
FRESHNESS_MAP = {
    "freshapples": "fresh",
    "freshbanana": "fresh",
    "freshoranges": "fresh",
    "rottenapples": "rotten",
    "rottenbanana": "rotten",
    "rottenoranges": "rotten",
}

CLASS_NAMES = ["fresh", "rotten"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


class FreshnessDataset(Dataset):
    def __init__(self, root_dir):
        self.base = datasets.ImageFolder(str(root_dir))
        self.samples = []

        for image_path, original_idx in self.base.samples:
            original_class = self.base.classes[original_idx]

            if original_class not in FRESHNESS_MAP:
                raise ValueError(f"Unexpected class: {original_class}")

            condition = FRESHNESS_MAP[original_class]
            label = CLASS_TO_IDX[condition]
            self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return transform(image), label


# CNN built from scratch
class FreshnessCNN(nn.Module):
    def __init__(self, num_classes=2):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    print("Device:", DEVICE)

    dataset = FreshnessDataset(TRAIN_DIR)
    print("Freshness classes:", CLASS_NAMES)
    print("Total images:", len(dataset))

    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in dataset.samples:
        counts[CLASS_NAMES[label]] += 1
    print("Images per class:", counts)

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False
    )

    model = FreshnessCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= train_size
        train_acc = 100 * train_correct / train_size
        val_loss /= val_size
        val_acc = 100 * val_correct / val_size

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "architecture": "freshness_cnn"
            }, MODEL_DIR / "freshness_model.pth")

    print(f"\nBest validation accuracy: {best_accuracy:.2f}%")
    print("Saved model:", MODEL_DIR / "freshness_model.pth")


if __name__ == "__main__":
    main()

