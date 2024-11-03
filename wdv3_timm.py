import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


@dataclass
class LabelData:
    names: List[str]
    rating: List[int]
    general: List[int]
    character: List[int]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    max_dim = max(image.size)
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, ((max_dim - image.width) // 2, (max_dim - image.height) // 2))
    return padded_image


def load_labels_hf() -> LabelData:
    csv_path = os.path.join(os.path.dirname(__file__), 'labels.csv')
    names = []
    rating, general, character = [], [], []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            names.append(row["name"])
            category = int(row["category"])
            if category == 9:
                rating.append(idx)
            elif category == 0:
                general.append(idx)
            elif category == 4:
                character.append(idx)
    return LabelData(names=names, rating=rating, general=general, character=character)


def get_tags(probs: Tensor, labels: LabelData, gen_threshold: float, char_threshold: float) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
    probs = list(zip(labels.names, map(float, probs.cpu().numpy())))
    rating_labels = {labels.names[i]: probs[i][1] for i in labels.rating}
    gen_labels = {label: score for label, score in (probs[i] for i in labels.general) if score > gen_threshold}
    char_labels = {label: score for label, score in (probs[i] for i in labels.character) if score > char_threshold}

    caption = ", ".join(sorted(gen_labels | char_labels, key=lambda x: -(gen_labels | char_labels)[x]))
    return caption, rating_labels, char_labels, gen_labels


def process_image(model: nn.Module, transform, image_path: Path, labels: LabelData, gen_threshold: float, char_threshold: float):
    img = Image.open(image_path)
    img = pil_pad_square(pil_ensure_rgb(img))
    img_tensor = transform(img).unsqueeze(0).to(torch_device)
    img_tensor = img_tensor[:, [2, 1, 0]]  # RGB to BGR

    with torch.inference_mode():
        model = model.to(torch_device)
        output = model(img_tensor)
        output = F.sigmoid(output)

    return get_tags(output.squeeze(0), labels, gen_threshold, char_threshold)


def load_model(repo_id: str) -> nn.Module:
    model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).eval()
    model.load_state_dict(timm.models.load_state_dict_from_hf(repo_id))
    return model


def main():
    parser = argparse.ArgumentParser(description="Process images with a pretrained model and extract tags.")
    parser.add_argument("image_path", type=Path, help="Path to an image or a directory containing images")
    parser.add_argument("--model", choices=MODEL_REPO_MAP.keys(), default="vit", help="Model to use")
    parser.add_argument("--gen-threshold", type=float, default=0.35, help="General tag threshold")
    parser.add_argument("--char-threshold", type=float, default=0.75, help="Character tag threshold")

    args = parser.parse_args()

    repo_id = MODEL_REPO_MAP[args.model]
    model = load_model(repo_id)
    labels = load_labels_hf()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    if args.image_path.is_file():
        image_files = [args.image_path]
    elif args.image_path.is_dir():
        image_files = list(args.image_path.glob("*.jpg")) + list(args.image_path.glob("*.png"))
        if not image_files:
            print(f"No images found in {args.image_path}")
            return
    else:
        print("The provided path is neither a file nor a directory.")
        return

    for image_path in image_files:
        print(f"Processing {image_path}...")
        caption, ratings, characters, generals = process_image(
            model, transform, image_path, labels, args.gen_threshold, args.char_threshold
        )
        print(f"Image: {image_path}")
        print(f"Caption: {caption}")
        print("Ratings:", ratings)
        print("Character tags:", characters)
        print("General tags:", generals)


if __name__ == "__main__":
    main()
