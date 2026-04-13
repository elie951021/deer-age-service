import io
from PIL import Image

# OpenAI vision limits: max 2048px on either side, min 768px on shortest side (high detail)
# A 10% buffer is applied on both ends to avoid boundary rejections
MAX_SIZE = int(2048 * 0.90)  # 1843px
MIN_SHORT_SIDE = int(768 * 1.10)  # 844px


def normalize_image(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = image.size

    # Scale down if either dimension exceeds MAX_SIZE
    if w > MAX_SIZE or h > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
        w, h = image.size

    # Scale up if shortest side is below MIN_SHORT_SIDE
    short_side = min(w, h)
    if short_side < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / short_side
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()
