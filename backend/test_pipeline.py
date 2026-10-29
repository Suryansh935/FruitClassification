
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report
from PIL import Image

from predict import load_models, transform, DEVICE


TEST_DIR = Path(r"F:\dataset\test")
BATCH_SIZE = 32

FRUIT_MAP = {
    "freshapples": "apple",
    "rottenapples": "apple",
    "freshbanana": "banana",
    "rottenbanana": "banana",
    "freshoranges": "orange",
    "rottenoranges": "orange",
}

FRESHNESS_MAP = {
    "freshapples": "fresh",
    "freshbanana": "fresh",
    "freshoranges": "fresh",
    "rottenapples": "rotten",
    "rottenbanana": "rotten",
    "rottenoranges": "rotten",
}


class PipelineTestDataset(Dataset):
    def __init__(self, root_dir):
        base = datasets.ImageFolder(str(root_dir))
        self.samples = []

        for image_path, original_label in base.samples:
            folder = base.classes[original_label].lower()

            self.samples.append((
                image_path,
                FRUIT_MAP[folder],
                FRESHNESS_MAP[folder]
            ))

        self.fruit_classes = ["apple", "banana", "orange"]
        self.freshness_classes = ["fresh", "rotten"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, fruit_label, freshness_label = self.samples[index]

        with Image.open(image_path) as image:
            image = transform(image.convert("RGB"))

        return image, fruit_label, freshness_label


def main():
    print("Device:", DEVICE)

    dataset = PipelineTestDataset(TEST_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    fruit_model, freshness_model, fruit_classes, freshness_classes = (
        load_models()
    )

    fruit_correct = 0
    freshness_correct = 0
    both_correct = 0
    total = 0

    fruit_true, fruit_pred = [], []
    freshness_true, freshness_pred = [], []

    fruit_index = {name: i for i, name in enumerate(fruit_classes)}
    freshness_index = {
        name: i for i, name in enumerate(freshness_classes)
    }

    with torch.inference_mode():
        for images, true_fruits, true_freshness in loader:
            images = images.to(DEVICE)

            predicted_fruits = fruit_model(images).argmax(dim=1).cpu()
            predicted_freshness = (
                freshness_model(images).argmax(dim=1).cpu()
            )

            true_fruit_ids = torch.tensor([
                fruit_index[name] for name in true_fruits
            ])
            true_freshness_ids = torch.tensor([
                freshness_index[name] for name in true_freshness
            ])

            fruit_matches = predicted_fruits == true_fruit_ids
            freshness_matches = (
                predicted_freshness == true_freshness_ids
            )

            fruit_correct += fruit_matches.sum().item()
            freshness_correct += freshness_matches.sum().item()
            both_correct += (
                fruit_matches & freshness_matches
            ).sum().item()
            total += len(images)

            fruit_true.extend(true_fruit_ids.tolist())
            fruit_pred.extend(predicted_fruits.tolist())
            freshness_true.extend(true_freshness_ids.tolist())
            freshness_pred.extend(predicted_freshness.tolist())

    print("\n========== PIPELINE TEST RESULTS ==========")
    print("Total test images:", total)
    print(f"Fruit accuracy: {100 * fruit_correct / total:.2f}%")
    print(f"Freshness accuracy: {100 * freshness_correct / total:.2f}%")
    print(f"Both correct: {both_correct}/{total}")
    print(f"Combined pipeline accuracy: {100 * both_correct / total:.2f}%")

    print("\nFruit classification report:")
    print(classification_report(
        fruit_true, fruit_pred,
        labels=list(range(len(fruit_classes))),
        target_names=fruit_classes,
        zero_division=0
    ))

    print("Freshness classification report:")
    print(classification_report(
        freshness_true, freshness_pred,
        labels=list(range(len(freshness_classes))),
        target_names=freshness_classes,
        zero_division=0
    ))


if __name__ == "__main__":
    main()
