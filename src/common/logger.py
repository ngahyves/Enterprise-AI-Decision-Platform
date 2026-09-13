import logging
import logging.config
from pathlib import Path
import yaml


class LoggerManager:
    """Manages logging config"""

    def __init__(self, config_path: str = "configs/logging.yaml"):
        self.config_path = Path(config_path)
        self._setup()

    def _setup(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Logging config not found: {self.config_path}")

        # Creates log folder if not exist
        log_dir=Path("logs")
        log_dir.mkdir(exist_ok=True)

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logging.config.dictConfig(config)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


logger_manager = LoggerManager()