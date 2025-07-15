"""
ログ設定ユーティリティ

機械学習プロジェクト用のログ設定を管理します。
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    機械学習用のログ設定をセットアップ
    
    Args:
        log_level: ログレベル ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: ログファイルパス（Noneの場合はコンソールのみ）
        log_format: ログフォーマット（Noneの場合はデフォルト）
        
    Returns:
        設定されたロガー
    """
    # デフォルトのログフォーマット
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ログレベルを設定
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # ロガーを取得
    logger = logging.getLogger('ml_project')
    logger.setLevel(numeric_level)
    
    # 既存のハンドラーをクリア
    logger.handlers.clear()
    
    # フォーマッターを作成
    formatter = logging.Formatter(log_format)
    
    # コンソールハンドラーを追加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラーを追加（指定された場合）
    if log_file:
        # ログディレクトリを作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"ログファイル設定: {log_file}")
    
    logger.info(f"ログ設定完了: レベル={log_level}")
    return logger


def get_ml_logger(name: str) -> logging.Logger:
    """
    機械学習プロジェクト用のロガーを取得
    
    Args:
        name: ロガー名（Noneの場合は'ml_project'）
        
    Returns:
        ロガー
    """
    if name is None:
        name = 'ml_project'
    return logging.getLogger(name)


class MLLoggerContext:
    """機械学習用ログコンテキストマネージャー"""
    
    def __init__(self, name: str, extra_info: dict):
        self.name = name
        self.extra_info = extra_info or {}
        self.logger = get_ml_logger(self.name)
        
    def __enter__(self):
        self.logger.info(f"開始: {self.name}", extra=self.extra_info)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"エラー終了: {self.name} - {exc_val}", extra=self.extra_info)
        else:
            self.logger.info(f"正常終了: {self.name}", extra=self.extra_info)


# デフォルトのML用ロガーを設定
def configure_default_ml_logging():
    """デフォルトの機械学習ログ設定を適用"""
    return setup_logging(
        log_level="INFO",
        log_file="backend/logs/ml.log",
        log_format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
