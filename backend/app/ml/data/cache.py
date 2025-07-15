"""
データキャッシュ管理モジュール

注文データなどの取得結果をメモリにキャッシュし、
繰り返しアクセスのパフォーマンスを向上させます。
"""
import datetime
import logging
from typing import Optional, List, Any
from sqlalchemy.orm import Session

from backend.app.crud import order_crud


class DataCache:
    """データキャッシュ管理クラス"""
    
    def __init__(self, default_cache_minutes: int = 60):
        """
        Args:
            default_cache_minutes: デフォルトキャッシュ有効期限（分）
        """
        self.default_cache_minutes = default_cache_minutes
        self._cached_orders: Optional[List[Any]] = None
        self._data_cache_timestamp: Optional[datetime.datetime] = None
        self._prepared_data_cache = None
        
    def load_and_cache_orders(self, db: Session) -> List[Any]:
        """
        データベースから注文データを読み込んでキャッシュ
        
        Args:
            db: データベースセッション
            
        Returns:
            注文データのリスト
        """
        logging.info("注文データのキャッシュを開始...")
        
        # 注文履歴を取得してキャッシュ
        self._cached_orders = order_crud.get_all_orders(db)
        self._data_cache_timestamp = datetime.datetime.now()
        
        if not self._cached_orders:
            logging.warning("注文データが見つかりません")
            return []
        
        logging.info(f"注文データキャッシュ完了: {len(self._cached_orders)}件")
        return self._cached_orders
    
    def get_cached_orders(self, db: Optional[Session] = None, force_reload: bool = False) -> Optional[List[Any]]:
        """
        キャッシュされた注文データを取得
        
        Args:
            db: データベースセッション（再読み込み用）
            force_reload: 強制的にデータを再読み込みするかどうか
            
        Returns:
            キャッシュされた注文データ、またはNone
        """
        # 強制再読み込みの場合
        if force_reload and db is not None:
            return self.load_and_cache_orders(db)
        
        # キャッシュが有効な場合
        if self.is_cache_valid() and self._cached_orders is not None:
            logging.info(f"キャッシュからデータを取得: {len(self._cached_orders)}件")
            return self._cached_orders
        
        # キャッシュが無効で、DBセッションが提供された場合
        if db is not None:
            return self.load_and_cache_orders(db)
        
        # キャッシュが無効で、DBセッションも無い場合
        if self._cached_orders is not None:
            logging.warning("期限切れキャッシュを使用（DBセッションが提供されていません）")
            return self._cached_orders
        
        return None
    
    def is_cache_valid(self, max_age_minutes: Optional[int] = None) -> bool:
        """
        キャッシュが有効かどうかを確認
        
        Args:
            max_age_minutes: キャッシュの有効期限（分）
            
        Returns:
            キャッシュが有効な場合True
        """
        if self._cached_orders is None or self._data_cache_timestamp is None:
            return False
        
        if max_age_minutes is None:
            max_age_minutes = self.default_cache_minutes
        
        age = datetime.datetime.now() - self._data_cache_timestamp
        return age.total_seconds() < (max_age_minutes * 60)
    
    def clear_cache(self) -> None:
        """データキャッシュをクリア"""
        self._cached_orders = None
        self._data_cache_timestamp = None
        self._prepared_data_cache = None
        logging.info("データキャッシュをクリアしました")
    
    def get_cache_info(self) -> dict:
        """キャッシュ情報を取得"""
        return {
            'has_cached_orders': self._cached_orders is not None,
            'cached_orders_count': len(self._cached_orders) if self._cached_orders else 0,
            'cache_timestamp': self._data_cache_timestamp,
            'is_cache_valid': self.is_cache_valid(),
            'cache_age_minutes': (
                (datetime.datetime.now() - self._data_cache_timestamp).total_seconds() / 60
                if self._data_cache_timestamp else None
            )
        }
