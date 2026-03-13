import argparse
import csv
import os
from collections import Counter


def summarize_csv(csv_file: str, images_root: str | None, val_split: float):
    labels = []
    missing = []
    total = 0
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'path' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError("CSV must contain 'path' and 'label' headers")
        for row in reader:
            p = row['path']
            if images_root and not os.path.isabs(p):
                p = os.path.join(images_root, p)
            total += 1
            if not os.path.exists(p):
                missing.append(p)
            labels.append(row['label'])

    counts = Counter(labels)
    print(counts)
    print("CVXXXXXXX")

    print(f"CSV samples: {total}")
    if missing:
        print(f"Missing files: {len(missing)}")
        for m in missing[:10]:
            print(f" - {m}")
        if len(missing) > 10:
            print(" ...")

    print("Class counts:")
    for cls, n in sorted(counts.items()):
        n_val = max(1, int(round(n * val_split))) if n > 1 else 1
        n_train = n - n_val
        print(f" - {cls}: total={n}, train~{n_train}, val~{n_val}")

    if len(counts) < 2:
        print("[WARN] Only one class found — classification won't work.")
        print("       Add at least a second class or use regression mode.")


def summarize_imagefolder(data_dir: str, val_split: float):
    # ImageFolder structure: data_dir/class_x/*.jpg
    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir not found: {data_dir}")
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    counts = {}
    total = 0
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        files = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in exts]
        counts[cls] = len(files)
        total += len(files)

    print(f"ImageFolder samples: {total}")
    print("Class counts:")
    for cls, n in sorted(counts.items()):
        n_val = max(1, int(round(n * val_split))) if n > 1 else 1
        n_train = n - n_val
        print(f" - {cls}: total={n}, train~{n_train}, val~{n_val}")

    if len(counts) < 2:
        print("[WARN] Only one class found — classification won't work.")


def main():
    parser = argparse.ArgumentParser(description="Summarize dataset class counts and approximate splits")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv_file', type=str, help='CSV file with path,label')
    group.add_argument('--imagefolder', type=str, help='Path to folder containing class subfolders')
    parser.add_argument('--images_root', type=str, default=None, help='Root to resolve relative CSV paths')
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    if args.csv_file:
        summarize_csv(args.csv_file, args.images_root, args.val_split)
    else:
        summarize_imagefolder(args.imagefolder, args.val_split)


if __name__ == '__main__':
    main()
