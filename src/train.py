import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from dataset_csv import CSVImageDataset, split_dataset, make_csv_train_val_datasets

try:
    from torchvision.models import mobilenet_v3_small, resnet18
    try:
        # Newer torchvision API
        from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights
        HAS_WEIGHTS_ENUM = False
    except Exception:
        HAS_WEIGHTS_ENUM = False
except Exception as e:
    raise RuntimeError("torchvision is required for models: {}".format(e))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int = 224, aggressive_augment: bool = True):
    """
    Build training and validation transforms.

    Args:
        img_size: Target image size
        aggressive_augment: Use stronger augmentation for small datasets (recommended for <100 images per class)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if aggressive_augment:
        # Stronger augmentation for small jawbone dataset
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Jawbones can be oriented differently
            transforms.RandomRotation(degrees=15),  # Slight rotation for angle variation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),  # Sometimes photos are grayscale
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Original lighter augmentation
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


essential_keys = [
    "data_dir", "epochs", "batch_size", "lr", "weight_decay",
    "img_size", "pretrained", "freeze_backbone", "val_split",
    "num_workers", "seed"
]


def get_dataloaders(data_dir: str, img_size: int, batch_size: int, val_split: float, num_workers: int,
                    csv_file: str | None = None, images_root: str | None = None):
    train_tf, val_tf = build_transforms(img_size)

    if csv_file:
        train_ds, val_ds, class_to_idx = make_csv_train_val_datasets(
            csv_file=csv_file,
            images_root=images_root or data_dir,
            train_tf=train_tf,
            val_tf=val_tf,
            val_split=val_split,
            seed=42,
        )
    else:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        use_explicit_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)

        if use_explicit_split:
            train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
            val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
            class_to_idx = train_ds.class_to_idx
        else:
            full_ds = datasets.ImageFolder(data_dir, transform=None)
            class_to_idx = full_ds.class_to_idx
            targets = full_ds.targets if hasattr(full_ds, 'targets') else [y for (_, y) in full_ds.samples]
            # Stratified split over ImageFolder targets
            by_class = {}
            for i, y in enumerate(targets):
                by_class.setdefault(y, []).append(i)
            rng = random.Random(42)
            train_idx, val_idx = [], []
            for y, idxs in by_class.items():
                rng.shuffle(idxs)
                n = len(idxs)
                n_val = max(1, int(round(n * val_split))) if n > 1 else 1
                val_idx.extend(idxs[:n_val])
                train_idx.extend(idxs[n_val:])

            # Build per-split datasets with independent transforms
            class ImageFolderSubset(torch.utils.data.Dataset):
                def __init__(self, base, indices, transform):
                    self.base = base
                    self.indices = indices
                    self.transform = transform
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, i):
                    idx = self.indices[i]
                    path, target = self.base.samples[idx]
                    from PIL import Image
                    img = Image.open(path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    return img, target

            train_ds = ImageFolderSubset(full_ds, train_idx, train_tf)
            val_ds = ImageFolderSubset(full_ds, val_idx, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    num_classes = len(class_to_idx)
    return train_loader, val_loader, num_classes, class_to_idx


def build_model(model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    model_name = model_name.lower()
    if model_name == 'mobilenet_v3_small':
        if HAS_WEIGHTS_ENUM:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_small(weights=weights)
        else:
            model = mobilenet_v3_small(pretrained=pretrained)

        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            last_idx = None
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    last_idx = i
                    break
            if last_idx is None:
                raise RuntimeError("Could not locate final Linear layer in classifier.")
            in_features = model.classifier[last_idx].in_features
            model.classifier[last_idx] = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected MobileNetV3 classifier structure.")

        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith('classifier'):
                    p.requires_grad = False

    elif model_name == 'resnet18':
        if HAS_WEIGHTS_ENUM:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        else:
            model = resnet18(pretrained=pretrained)

        in_features = model.fc.in_features
        print("feature nums=")
        print(in_features)
        model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            print("Freeze")
            for name, p in model.named_parameters():
                if name.startswith('fc.'):
                    continue
                p.requires_grad = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train image classifier (ResNet18 or MobileNetV3-Small) on ImageFolder or CSV data")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data root')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file with columns path,label')
    parser.add_argument('--images_root', type=str, default=None, help='Root to resolve relative CSV paths (defaults to data_dir)')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v3_small'], help='Backbone model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use ImageNet weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Disable ImageNet weights')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze all but classifier')
    parser.add_argument('--val_split', type=float, default=0.0, help='Used if no explicit val folder or when using CSV')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (CUDA only)')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, num_classes, class_to_idx = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        csv_file=args.csv_file,
        images_root=args.images_root,
    )

    save_json(class_to_idx, os.path.join(args.output_dir, 'class_to_idx.json'))

    model = build_model(model_name=args.model,
                        num_classes=num_classes,
                        pretrained=args.pretrained,
                        freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.2)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    best_acc = 0.0
    history = []

    run_meta = {k: getattr(args, k) for k in vars(args)}
    run_meta.update({
        'device': str(device),
        'num_classes': num_classes,
        'start_time': datetime.now().isoformat(timespec='seconds')
    })

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # Save last checkpoint
        torch.save({'model_state': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'args': vars(args)},
                   os.path.join(args.checkpoint_dir, 'last.pt'))

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state': model.state_dict(),
                        'class_to_idx': class_to_idx,
                        'args': vars(args)},
                       os.path.join(args.checkpoint_dir, 'best.pt'))

        save_json({'meta': run_meta, 'history': history},
                  os.path.join(args.output_dir, 'last_run.json'))

    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
