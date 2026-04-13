import json
import logging
from datetime import datetime
from pathlib import Path

from app.loggers.logger import logger
from app.config import settings

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


class RequestSession:
    def __init__(self) -> None:
        self._enabled = settings.session_logging_enabled
        self._file_handler: logging.FileHandler | None = None

        if not self._enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = LOGS_DIR / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._file_handler = logging.FileHandler(
            self.session_dir / "session.log", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(pathname)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(self._file_handler)
        logger.info("Session started: %s", self.session_dir.name)

    def save_image(self, image_bytes: bytes, filename: str = "input.jpg") -> None:
        if not self._enabled:
            return
        path = self.session_dir / filename
        path.write_bytes(image_bytes)
        logger.info("Image saved: %s", filename)

    def save_prompt(self, messages: list) -> None:
        if not self._enabled:
            return
        path = self.session_dir / "prompt.txt"
        lines = []
        for m in messages:
            role = getattr(m, "type", "unknown").upper()
            content = m.content
            if isinstance(content, list):
                content = "[image content omitted]"
            lines.append(f"[{role}]\n{content}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Prompt saved")

    def save_response(self, data: dict) -> None:
        if not self._enabled:
            return
        path = self.session_dir / "response.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Response saved")

    def close(self) -> None:
        if not self._enabled or self._file_handler is None:
            return
        logger.info("Session closed: %s", self.session_dir.name)
        logger.removeHandler(self._file_handler)
        self._file_handler.close()
