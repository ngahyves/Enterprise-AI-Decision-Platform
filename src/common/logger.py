import logging
import logging.config
import os
from pathlib import Path

import yaml


class LoggerManager:
    """
    Manages logging config.

    """

    def __init__(self, config_path: str = "configs/logging.yaml"):
        self.config_path = Path(config_path)
        self._setup()

    def _setup(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Logging config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        use_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"

        if use_file:
            Path("logs").mkdir(exist_ok=True)
        else:
            self._strip_file_handlers(config)

        logging.config.dictConfig(config)

    @staticmethod
    def _strip_file_handlers(config: dict) -> None:
        """Remove 'file' handler from all loggers and from root."""
        file_handlers = {"file"}

        # Root
        root = config.get("root", {})
        if "handlers" in root:
            root["handlers"] = [h for h in root["handlers"] if h not in file_handlers]

        # Named loggers
        for _, logger_cfg in config.get("loggers", {}).items():
            if "handlers" in logger_cfg:
                logger_cfg["handlers"] = [
                    h for h in logger_cfg["handlers"] if h not in file_handlers
                ]

        config.get("handlers", {}).pop("file", None)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


logger_manager = LoggerManager()
