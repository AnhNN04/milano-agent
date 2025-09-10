# src/stock_assistant/shared/logging/logger.py
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "stock_assistant") -> logging.Logger:
        """
        Trả về một instance logger duy nhất (singleton).
        """
        if cls._instance is None:
            cls._instance = cls._setup_logger(name)
        return cls._instance

    @classmethod
    def _setup_logger(cls, name: str) -> logging.Logger:
        """
        Cấu hình logger để ghi cả console và file.
        File log sẽ đặt tên theo ngày hiện tại trong thư mục logs/
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Tránh thêm nhiều handler nếu logger đã có handler
        if logger.hasHandlers():
            return logger

        # Formatter hiển thị thời gian, level, file, dòng, hàm, message
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            # Thư mục logs/ nằm ở root project
            log_dir = (
                Path(__file__).resolve().parent.parent.parent.parent / "logs"
            )
            os.makedirs(log_dir, exist_ok=True)

            # File log theo ngày hiện tại: 2025-08-19.log
            today_str = datetime.now().strftime("%Y-%m-%d")
            log_file_path = log_dir / f"{today_str}.log"

            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            # Nếu không tạo được file log, vẫn ghi ra console
            logger.warning(f"Không thể tạo hoặc ghi vào tệp log: {e}")

        return logger
