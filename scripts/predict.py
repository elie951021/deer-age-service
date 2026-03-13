import argparse
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from torchvision.models import mobilenet_v3_small, resnet18
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights
        HAS_WEIGHTS_ENUM = True
    except Exception:
        HAS_WEIGHTS_ENUM = False
except Exception as e:
    raise RuntimeError("torchvision is required: {}".format(e))


def build_transform(img_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def make_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=None) if HAS_WEIGHTS_ENUM else mobilenet_v3_small(pretrained=False)
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
        return model
    elif model_name == 'resnet18':
        model = resnet18(weights=None) if HAS_WEIGHTS_ENUM else resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_model(checkpoint_path: str, num_classes: int, model_name: str | None):
    state = torch.load(checkpoint_path, map_location='cpu')
    ckpt_args = state.get('args', {})
    inferred_model = ckpt_args.get('model') if isinstance(ckpt_args, dict) else None
    model_to_use = (model_name or inferred_model or 'resnet18').lower()

    model = make_model(model_to_use, num_classes=num_classes)
    model.load_state_dict(state['model_state'], strict=True)
    model.eval()
    return model, model_to_use


def predict_image(model, tfm, image_path: str, idx_to_class: dict, device: torch.device):
    img = Image.open(image_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    cls = idx_to_class[int(pred.item())]
    return cls, float(conf.item())


def main():
    parser = argparse.ArgumentParser(description='Predict with ResNet18/MobileNetV3 checkpoint')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', type=str, help='Path to a single image')
    g.add_argument('--images_dir', type=str, help='Predict all images in a directory')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--class_map', type=str, required=True, help='JSON with class_to_idx mapping')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', type=str, default=None, choices=['resnet18', 'mobilenet_v3_small'], help='Override model arch; defaults to checkpoint args')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.class_map, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model, used_model = load_model(args.checkpoint, num_classes=len(idx_to_class), model_name=args.model)
    model = model.to(device)
    tfm = build_transform(args.img_size)

    if args.image:
        cls, conf = predict_image(model, tfm, args.image, idx_to_class, device)
        print(f"[{used_model}] {args.image} -> {cls} ({conf:.3f})")
    else:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        paths = [os.path.join(args.images_dir, p) for p in os.listdir(args.images_dir)
                 if os.path.splitext(p)[1].lower() in exts]
        if not paths:
            print("No images found in directory.")
            return
        for p in paths:
            cls, conf = predict_image(model, tfm, p, idx_to_class, device)
            print(f"[{used_model}] {p} -> {cls} ({conf:.3f})")


if __name__ == '__main__':
    main()
