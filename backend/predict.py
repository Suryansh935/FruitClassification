
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -------------------- CONFIGURATION --------------------

MODEL_DIR = Path("models")
FRUIT_MODEL_PATH = MODEL_DIR / "fruit_model.pth"
FRESHNESS_MODEL_PATH = MODEL_DIR / "freshness_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- IMAGE TRANSFORM --------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -------------------- MODEL 1: FRUIT CNN --------------------

class FruitCNN(nn.Module):
    def __init__(self, num_classes=3):
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


# -------------------- MODEL 2: FRESHNESS CNN --------------------

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


# -------------------- LOAD MODELS --------------------

def load_models():
    if not FRUIT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fruit model not found: {FRUIT_MODEL_PATH}"
        )

    if not FRESHNESS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Freshness model not found: {FRESHNESS_MODEL_PATH}"
        )

    fruit_checkpoint = torch.load(
        FRUIT_MODEL_PATH, map_location=DEVICE
    )

    freshness_checkpoint = torch.load(
        FRESHNESS_MODEL_PATH, map_location=DEVICE
    )

    fruit_classes = fruit_checkpoint["class_names"]
    freshness_classes = freshness_checkpoint["class_names"]

    fruit_model = FruitCNN(
        num_classes=len(fruit_classes)
    ).to(DEVICE)

    freshness_model = FreshnessCNN(
        num_classes=len(freshness_classes)
    ).to(DEVICE)

    fruit_model.load_state_dict(
        fruit_checkpoint["model_state_dict"]
    )

    freshness_model.load_state_dict(
        freshness_checkpoint["model_state_dict"]
    )

    fruit_model.eval()
    freshness_model.eval()

    return (
        fruit_model,
        freshness_model,
        fruit_classes,
        freshness_classes
    )


# -------------------- PREDICT IMAGE --------------------

def predict_image(image_path, models):
    fruit_model, freshness_model, fruit_classes, freshness_classes = models

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        fruit_output = fruit_model(image_tensor)
        freshness_output = freshness_model(image_tensor)

        fruit_probabilities = torch.softmax(
            fruit_output, dim=1
        )[0]

        freshness_probabilities = torch.softmax(
            freshness_output, dim=1
        )[0]

        fruit_idx = fruit_probabilities.argmax().item()
        freshness_idx = freshness_probabilities.argmax().item()

    fruit_name = fruit_classes[fruit_idx]
    freshness_name = freshness_classes[freshness_idx]

    fruit_confidence = fruit_probabilities[fruit_idx].item() * 100
    freshness_confidence = (
        freshness_probabilities[freshness_idx].item() * 100
    )

    print("\n========== FRUIT CLASSIFICATION RESULT ==========")
    print(f"Image: {image_path.name}")
    print(f"Fruit type: {fruit_name.title()}")
    print(f"Fruit confidence: {fruit_confidence:.2f}%")
    print(f"Condition: {freshness_name.title()}")
    print(f"Freshness confidence: {freshness_confidence:.2f}%")
    print("=================================================\n")


# -------------------- MAIN --------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "path\\to\\image.jpg"')
        sys.exit(1)

    try:
        models = load_models()
        predict_image(sys.argv[1], models)
    except (FileNotFoundError, KeyError, RuntimeError, OSError) as error:
        print(f"Error: {error}")
        sys.exit(1)
