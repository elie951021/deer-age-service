# DeerAge MVP — ResNet18/MobileNetV3 Training + API

This MVP trains an image classifier using transfer learning. By default it uses pretrained ResNet18, and it also supports MobileNetV3-Small. It supports ImageFolder layout or a CSV file listing image paths and labels. It provides a training loop, checkpointing, a CLI predictor, and a FastAPI prediction server.

## Dataset Layout
Use one of these:

1) Separate train/val folders
```
data/
  train/
    class_a/ ...images...
    class_b/ ...
  val/
    class_a/ ...
    class_b/ ...
```

2) Single folder (auto split by `--val_split`)
```
data/
  class_a/ ...images...
  class_b/ ...
```

3) CSV file with image paths and labels
```
data/
  images/...
  train.csv

# train.csv (headers required)
path,label
images/img1.jpg,class_a
images/img2.jpg,class_b
```

## Quickstart (Windows)

1) Create a virtual environment and activate it:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install requirements:
```powershell
pip install -r requirements.txt
```

3) Train (auto-detects classes from the dataset):
```powershell
# ImageFolder (with data/train and data/val or auto-split). Defaults to pretrained ResNet18.
py .\src\train.py --data_dir .\data --epochs 10 --batch_size 32 --lr 3e-4 --img_size 224 --freeze_backbone

# CSV (uses data/train.csv and data/images). Defaults to pretrained ResNet18.
python .\src\train.py --data_dir .\data --csv_file .\data\train.csv --images_root .\data --epochs 20 --batch_size 32 --lr 3e-4 --img_size 224 --freeze_backbone

# Switch backbone to MobileNetV3-Small
py .\src\train.py --data_dir .\data --model mobilenet_v3_small --epochs 10 --batch_size 32 --lr 3e-4 --img_size 224 --freeze_backbone
```

Artifacts:
- Best checkpoint: `checkpoints/best.pt`
- Class map: `outputs/class_to_idx.json`
- Training log/config: `outputs/last_run.json`

## Inference
Predict for a single image or a directory of images:
```powershell
py .\scripts\predict.py --image .\path\to\image.jpg --checkpoint .\checkpoints\best.pt --class_map .\outputs\class_to_idx.json
# or
py .\scripts\predict.py --images_dir .\path\to\images --checkpoint .\checkpoints\best.pt --class_map .\outputs\class_to_idx.json
```

## FastAPI Server
Run an HTTP prediction server (loads latest checkpoint and class map):
```powershell
set CHECKPOINT_PATH=checkpoints\best.pt
set CLASS_MAP_PATH=outputs\class_to_idx.json
set MODEL_NAME=resnet18
set SQLITE_DB_PATH=outputs\app.db
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- GET /health — service status
- GET /history?usermail=<email> — returns saved predictions for a user (newest first)
- POST /predict — multipart file field `file`; returns top-k predictions

Prediction results are also saved to SQLite (`predictions` table) with filename, saved image path, timestamp, top prediction, confidence, reliability, and full JSON response.

Example request with PowerShell (Invoke-RestMethod):
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -InFile .\path\to\image.jpg -ContentType 'multipart/form-data'
```

## Notes
- Default model: pretrained ResNet18; switch with `--model mobilenet_v3_small`.
- Toggle pretraining: add `--no-pretrained` to train from scratch.
- Freezing the backbone (`--freeze_backbone`) speeds up training on small datasets.
- If CUDA is available, the script uses it automatically.
- You can customize transforms and hyperparameters via CLI flags.
