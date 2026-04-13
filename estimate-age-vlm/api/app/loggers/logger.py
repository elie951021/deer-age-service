import logging
import sys


class AppLogger:
    _instance: "AppLogger | None" = None
    _logger: logging.Logger

    def __new__(cls) -> "AppLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._logger = logging.getLogger("deer_age_api")
        self._logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(pathname)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        self._logger.addHandler(handler)
        self._logger.propagate = False

    def get(self) -> logging.Logger:
        return self._logger


logger = AppLogger().get()
