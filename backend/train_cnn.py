
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------- CONFIGURATION ----------------

DATASET_DIR = Path(r"F:\dataset\train")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "custom_cnn.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_RATIO = 0.15
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {DEVICE}")


# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ---------------- LOAD DATASET ----------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(dataset))

if num_classes < 2:
    raise ValueError("At least two classes are required.")

if len(dataset) < 2:
    raise ValueError("Not enough images to train.")

# Split into training and validation sets.
validation_size = int(len(dataset) * VALIDATION_RATIO)
training_size = len(dataset) - validation_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=generator
)

# Calculate class weights using TRAINING samples only.
train_labels = [
    dataset.samples[index][1]
    for index in train_dataset.indices
]

counts = Counter(train_labels)

class_weights = torch.tensor(
    [
        len(train_labels) / (num_classes * counts[i])
        for i in range(num_classes)
    ],
    dtype=torch.float32
).to(DEVICE)

print("Training class counts:", {
    class_names[i]: counts[i] for i in range(num_classes)
})
print("Class weights:", class_weights.cpu().tolist())
print("Training images:", training_size)
print("Validation images:", validation_size)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ---------------- CUSTOM CNN ARCHITECTURE ----------------

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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
        x = self.features(x)
        return self.classifier(x)


model = CustomCNN(num_classes=num_classes).to(DEVICE)

# Weighted loss gives more importance to underrepresented classes.
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Reduce learning rate when validation loss stops improving.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ---------------- TRAINING ----------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_val_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "architecture": "custom_cnn",
        }, MODEL_PATH)

        print(f"Best model saved to: {MODEL_PATH}")

print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")
print("Custom CNN training completed.")
