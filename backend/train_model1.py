
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

# ---------------- CONFIGURATION ----------------

DATASET_DIR = Path(r"F:\dataset\train")
MODEL_DIR = Path(
    r"C:\Users\Harshal Makode\Desktop\DL\FruitClassification\backend\models"
)
MODEL_PATH = MODEL_DIR / "model1_fruit.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
VALIDATION_RATIO = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {DEVICE}")

# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD DATASET ----------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

print("Detected classes:", dataset.classes)
print("Total images:", len(dataset))

if len(dataset) < 2:
    raise ValueError("Not enough images to train.")

if len(dataset.classes) < 2:
    raise ValueError("At least two classes are required.")

class_counts = {
    class_name: sum(1 for _, label in dataset.samples if label == class_index)
    for class_index, class_name in enumerate(dataset.classes)
}
print("Images per class:", class_counts)

# ---------------- TRAIN / VALIDATION SPLIT ----------------

generator = torch.Generator().manual_seed(SEED)

validation_size = int(len(dataset) * VALIDATION_RATIO)
training_size = len(dataset) - validation_size

if validation_size < 1:
    raise ValueError("Not enough images for a validation split.")

train_dataset, val_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=generator
)

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

print("Training images:", training_size)
print("Validation images:", validation_size)

# ---------------- BUILD MODEL ----------------

weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze pretrained feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# IMPORTANT: Four output classes, based on dataset folders.
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(dataset.classes))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# ---------------- TRAINING ----------------

best_val_accuracy = -1.0
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss_total = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    train_loss = train_loss_total / train_total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss_total / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    # Save the best validation checkpoint.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": dataset.classes,
            "image_size": IMAGE_SIZE,
            "architecture": "mobilenet_v3_small"
        }, MODEL_PATH)

        print(f"Best model saved to: {MODEL_PATH}")

print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")
print("Model 1 training completed.")
