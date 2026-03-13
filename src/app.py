import os
import json
import sqlite3
from pathlib import Path
from typing import Optional
from io import BytesIO
from datetime import datetime
from uuid import uuid4

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
    raise RuntimeError(f"torchvision is required: {e}")


CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', 'checkpoints/best.pt')
CLASS_MAP_PATH = os.getenv('CLASS_MAP_PATH', 'outputs/class_to_idx.json')
MODEL_NAME = os.getenv('MODEL_NAME', 'resnet18').lower()
IMG_SIZE = int(os.getenv('IMG_SIZE', '224'))
TOP_K = int(os.getenv('TOP_K', '3'))
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'upload')
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'database/app.db')

app = FastAPI(
    title="DeerAge - Jawbone Age Classification API",
    description="AI-powered deer age estimation using jawbone analysis based on tooth wear patterns",
    version="1.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1:8081"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
STATIC_DIR = SCRIPT_DIR / "static"

# Mount static files if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if Path(UPLOAD_DIR).exists():
    app.mount("/upload", StaticFiles(directory=UPLOAD_DIR), name="upload")

# Age class metadata for reliability assessment
# Scientific characteristics based on tooth replacement and wear patterns
AGE_CLASS_INFO = {
    "0.5": {
        "description": "Fawn (6 months)",
        "reliability": "high",
        "characteristics": "Fawns have 5 or less teeth present. The 3rd premolar (tooth 3) has 3 cusps. Tooth 6 has not yet erupted. In younger fawns, tooth 5 has not erupted and only 4 teeth will be visible.",
        "key_indicators": ["5 or fewer teeth", "Tooth 3 has 3 cusps", "Tooth 6 not erupted"],
        "note": None
    },
    "1.5": {
        "description": "Yearling (1.5 years)",
        "reliability": "high",
        "characteristics": "Tooth 3 (3rd premolar) has 3 cusps with heavy wear. Tooth 6 has erupted and is slightly visible just above the gum line. All 6 teeth are now present.",
        "key_indicators": ["3-cusp tooth 3 with heavy wear", "Tooth 6 just erupted", "6 teeth visible"],
        "note": None
    },
    "2.5": {
        "description": "2.5 years old",
        "reliability": "high",
        "characteristics": "Lingual crest on all molars are sharp and pointed. Tooth 3 now has 2 cusps (permanent replacement). Back cusp of tooth 6 is sharp and pointed. Enamel is wider than the dentine in teeth 4, 5, and 6.",
        "key_indicators": ["Tooth 3 now has 2 cusps", "Sharp lingual crests", "Enamel wider than dentine"],
        "note": None
    },
    "3.5": {
        "description": "3.5 years old",
        "reliability": "high",
        "characteristics": "Lingual crest on tooth 4 is blunt. The dentine is as wide or wider than the enamel in tooth 4. The back cusp on tooth 6 is forming a concavity.",
        "key_indicators": ["Blunt lingual crest on tooth 4", "Dentine equals enamel width in tooth 4", "Concavity forming on tooth 6"],
        "note": None
    },
    "4.5": {
        "description": "4.5 years old",
        "reliability": "moderate",
        "characteristics": "Lingual crest on tooth 4 is almost rounded off and lingual crest in tooth 5 is blunt. The dentine in tooth 4 is twice as wide as the enamel. The dentine in tooth 5 is wider than the enamel. The back cusp on tooth 6 slopes downward towards the cheek.",
        "key_indicators": ["Rounded lingual crest on tooth 4", "Dentine 2x wider than enamel in tooth 4", "Cusp slopes downward on tooth 6"],
        "note": None
    },
    "5+": {
        "description": "5+ years old (mature deer)",
        "reliability": "moderate",
        "characteristics": "Lingual crests show significant wear or are worn away. Dentine is wider than enamel on multiple teeth. Teeth may have 'dished out' appearance. For deer 5.5 years and older, precise aging becomes unreliable due to individual variation in tooth wear.",
        "key_indicators": ["Dentine wider than enamel", "Significant lingual crest wear", "Dished out appearance possible"],
        "note": "Visual aging becomes unreliable for deer 5+ years old. Tooth wear varies significantly by diet, habitat, and soil conditions. For precise aging of mature deer, cementum annuli analysis is strongly recommended."
    }
}


def get_reliability_level(age_class: str, confidence: float) -> str:
    """Determine overall reliability based on age class and model confidence."""
    base_reliability = AGE_CLASS_INFO.get(age_class, {}).get("reliability", "unknown")

    # Adjust based on confidence
    if confidence < 0.4:
        return "low"
    elif confidence < 0.6:
        if base_reliability == "high":
            return "moderate"
        return "low"
    elif confidence < 0.8:
        if base_reliability == "low":
            return "low"
        return base_reliability
    else:
        return base_reliability


def build_transform(img_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path: str, num_classes: int, model_name: str):
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
    elif model_name == 'resnet18':
        model = resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state'], strict=True)
    model.eval()
    return model


def init_database(db_path: str):
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usermail TEXT,
                created_at TEXT NOT NULL,
                original_filename TEXT,
                saved_filename TEXT,
                saved_image_path TEXT,
                age_estimate TEXT,
                confidence REAL,
                reliability TEXT
            )
            """
        )
        conn.commit()


def save_prediction_result(
    db_path: str,
    usermail: str,
    original_filename: str,
    saved_filename: str,
    saved_image_path: str,
    response_payload: dict,
):
    prediction = response_payload.get("prediction", {})
    created_at = datetime.utcnow().isoformat(timespec='seconds') + "Z"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at,
                usermail,
                original_filename,
                saved_filename,
                saved_image_path,
                age_estimate,
                confidence,
                reliability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                usermail,
                original_filename,
                saved_filename,
                saved_image_path,
                prediction.get("age_estimate"),
                prediction.get("confidence"),
                prediction.get("reliability")
            ),
        )
        conn.commit()


def to_upload_url(saved_image_path: Optional[str]) -> Optional[str]:
    if not saved_image_path:
        return None

    normalized = saved_image_path.replace("\\", "/")
    if normalized.startswith("upload/"):
        return f"/{normalized}"
    if normalized.startswith("/upload/"):
        return normalized

    upload_index = normalized.find("/upload/")
    if upload_index >= 0:
        return normalized[upload_index:]

    return None


def get_prediction_history_by_usermail(db_path: str, usermail: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                usermail,
                created_at,
                original_filename,
                saved_filename,
                saved_image_path,
                age_estimate,
                confidence,
                reliability
            FROM predictions
            WHERE lower(usermail) = lower(?)
            ORDER BY created_at DESC, id DESC
            """,
            (usermail,),
        ).fetchall()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "usermail": row["usermail"],
            "created_at": row["created_at"],
            "original_filename": row["original_filename"],
            "saved_filename": row["saved_filename"],
            "saved_image_path": row["saved_image_path"],
            "saved_image_url": to_upload_url(row["saved_image_path"]),
            "prediction": {
                "age_estimate": row["age_estimate"],
                "confidence": row["confidence"],
                "reliability": row["reliability"],
            },
        })

    return history


@app.on_event("startup")
def startup_event():
    global model, idx_to_class, tfm, device

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(CLASS_MAP_PATH):
        raise RuntimeError(f"Class map not found: {CLASS_MAP_PATH}")

    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CHECKPOINT_PATH, num_classes=len(idx_to_class), model_name=MODEL_NAME).to(device)
    tfm = build_transform(IMG_SIZE)
    init_database(SQLITE_DB_PATH)


@app.get('/', response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return HTMLResponse(content="<h1>DeerAge API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get('/health')
async def health():
    return {"status": "ok", "model": MODEL_NAME, "classes": list(idx_to_class.values()) if 'idx_to_class' in globals() else []}


@app.get('/age-classes')
async def get_age_classes():
    """Get information about all supported age classes and their characteristics."""
    return JSONResponse(content={
        "age_classes": AGE_CLASS_INFO,
        "methodology": "Jawbone tooth replacement and wear analysis",
        "sources": [
            "Mossy Oak Gamekeeper - Jawbone Analysis",
            "Texas A&M University Research",
            "National Deer Association",
            "Missouri University Extension"
        ],
        "note": "Visual jawbone aging is most accurate for deer up to 5.5 years. For older deer, cementum annuli analysis is recommended."
    })


@app.get('/history')
async def get_history(usermail: str):
    usermail = usermail.strip()
    if not usermail:
        raise HTTPException(status_code=400, detail="Query parameter 'usermail' is required")

    history = get_prediction_history_by_usermail(SQLITE_DB_PATH, usermail)
    return JSONResponse(content={
        "usermail": usermail,
        "count": len(history),
        "history": history,
    })


@app.post('/predict')
async def predict(file: UploadFile = File(...), usermail: Optional[str] = None):
    """
    Predict deer age from jawbone image.

    Returns age estimate with confidence score, reliability assessment,
    and relevant notes about the prediction accuracy.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
    except Exception:
        await file.seek(0)
        image = Image.open(file.file).convert('RGB')

    # Create date-named directory in upload folder
    current_date = datetime.now().strftime('%Y-%m')
    date_folder = Path(UPLOAD_DIR) / current_date
    date_folder.mkdir(parents=True, exist_ok=True)

    # Save the uploaded image with a random filename
    suffix = Path(file.filename).suffix.lower() if file.filename else ''
    if not suffix:
        suffix = '.jpg'
    random_filename = f"{uuid4().hex}{suffix}"
    image_path = date_folder / random_filename
    image.save(str(image_path))

    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = min(TOP_K, probs.shape[0])
        confs, idxs = torch.topk(probs, k=topk)

    # Build detailed predictions
    predictions = []
    for c, i in zip(confs, idxs):
        age_class = idx_to_class[int(i.item())]
        confidence = float(c.item())
        class_info = AGE_CLASS_INFO.get(age_class, {})

        predictions.append({
            "age_class": age_class,
            "description": class_info.get("description", f"{age_class} years old"),
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "characteristics": class_info.get("characteristics", ""),
        })

    # Primary prediction with full details
    primary = predictions[0]
    primary_class = primary["age_class"]
    primary_confidence = primary["confidence"]
    primary_info = AGE_CLASS_INFO.get(primary_class, {})

    response = {
        "filename": file.filename,
        "prediction": {
            "age_estimate": primary_class,
            "description": primary["description"],
            "confidence": primary["confidence"],
            "confidence_percent": primary["confidence_percent"],
            "reliability": get_reliability_level(primary_class, primary_confidence),
            "characteristics": primary["characteristics"],
        },
        "alternatives": predictions[1:] if len(predictions) > 1 else [],
        "methodology": "Jawbone tooth wear analysis",
    }

    # Add warnings for older deer or low confidence
    warnings = []
    if primary_class == "5+":
        warnings.append(primary_info.get("note", "Visual aging is unreliable for deer 5+ years old. Cementum annuli analysis recommended."))
    if primary_confidence < 0.5:
        warnings.append("Low confidence prediction. Image quality or angle may affect accuracy. Consider submitting a clearer image.")
    if primary_confidence < 0.3:
        warnings.append("Very low confidence. This prediction should not be relied upon.")

    if warnings:
        response["warnings"] = warnings

    save_prediction_result(
        db_path=SQLITE_DB_PATH,
        usermail=usermail,
        original_filename=file.filename,
        saved_filename=random_filename,
        saved_image_path=str(image_path),
        response_payload=response,
    )

    return JSONResponse(content=response)
