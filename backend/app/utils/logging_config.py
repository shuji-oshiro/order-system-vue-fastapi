import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path
import os


class JSONFormatter(logging.Formatter):
    """JSON形式でログを出力するフォーマッター"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # エラーの場合は例外情報を追加
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 追加のフィールドがあれば追加
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = getattr(record, 'user_id')
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = getattr(record, 'request_id')
        if hasattr(record, 'endpoint'):
            log_entry["endpoint"] = getattr(record, 'endpoint')
        if hasattr(record, 'method'):
            log_entry["method"] = getattr(record, 'method')
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = getattr(record, 'status_code')
        if hasattr(record, 'response_time'):
            log_entry["response_time"] = getattr(record, 'response_time')
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """ログ設定を初期化"""
    
    # ログディレクトリを作成
    log_dir = Path("./backend/logs")
    log_dir.mkdir(exist_ok=True)
    
    # 環境に応じてログレベルを設定
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "filters": {
            "uvicorn_access_filter": {
                "()": UvicornAccessFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": log_dir / "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": log_dir / "error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "uvicorn_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": log_dir / "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
                "filters": ["uvicorn_access_filter"]
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "backend.app": {
                "handlers": ["console", "file", "error_file"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["uvicorn_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # アプリケーション開始ログ
    logger = logging.getLogger("backend.app.startup")
    logger.info("ログシステムが初期化されました", extra={
        "log_level": log_level,
        "log_dir": str(log_dir)
    })


def get_logger(name: str) -> logging.Logger:
    """名前付きロガーを取得"""
    return logging.getLogger(f"backend.app.{name}")


class UvicornAccessFilter(logging.Filter):
    """Uvicornアクセスログから監視系エンドポイントを除外するフィルター"""
    
    EXCLUDED_PATHS = {
        "/monitoring/health",
        "/monitoring/health/simple",
        "/monitoring/health/detailed",
        "/monitoring/metrics",
        "/monitoring/logs",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    def filter(self, record):
        """監視系エンドポイントのアクセスログを除外"""
        if hasattr(record, 'args') and record.args and len(record.args) >= 1:
            # Uvicornのアクセスログフォーマット: "IP:PORT - \"METHOD PATH HTTP/1.1\" STATUS"
            log_message = record.getMessage()
            for excluded_path in self.EXCLUDED_PATHS:
                if excluded_path in log_message:
                    return False
        return True
