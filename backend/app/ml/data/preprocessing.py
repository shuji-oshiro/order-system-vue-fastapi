"""
メニューデータ前処理モジュール

注文データからメニュー間の関連性学習用データを準備します。
特徴量エンジニアリング、ポジティブ・ネガティブサンプリング等を含みます。
"""
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from backend.app.crud.order_crud import get_all_orders_for_ml_training  

from backend.app.models.model import Menu


class MenuDataPreprocessor:
    """メニューデータ前処理クラス"""
    
    def __init__(self):
        self.menu_encoder = LabelEncoder()
        
    def prepare_menu_pairs(
        self, 
        # orders: List[Any], 
        db: Optional[Session] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        メニュー間の関連性学習用データを準備
        
        Args:
            orders: 注文データのリスト
            db: データベースセッション（メニュー情報取得用）
            
        Returns:
            menu1_ids, menu2_ids, features, targets, _ のタプル
        """
        logging.info("メニュー関連性データの準備を開始...")
        
        # if not orders:
        #     raise ValueError("注文データが見つかりません")

        
        # メニュー情報を取得（カテゴリ情報含む）
        # menu_categories = self._get_menu_categories(orders, db)

        # DataFrameに変換
        # df = self._create_orders_dataframe(orders, menu_categories)

        if db is None:
            raise ValueError("データベースセッションが必要です")
        
        orders = get_all_orders_for_ml_training(db= db)

        # OrderSchemaをDataFrameに変換
        order_data = []
        for order in orders:
            order_data.append({
                'seat_id': order.seat_id,
                'menu_id': order.menu_id,
                'quantity': order.order_cnt,
                'order_datetime': order.order_date,
                'hour': order.order_date.hour if order.order_date else 12,
                'category_id': order.menu.category_id if order.menu and hasattr(order.menu, 'category_id') else 1,
                'category_name': order.menu.category.name if order.menu and hasattr(order.menu, 'category') and order.menu.category else '',
                'menu_name': order.menu.name if order.menu else '',
                'menu_price': order.menu.price if order.menu else 0
            })
        
        if not order_data:
            raise ValueError("有効な注文データが見つかりません")
            
        df = pd.DataFrame(order_data)
        logging.info(f"DataFrameに変換された注文データ: {len(order_data)}件")

        
        # 座席IDごとに注文されたメニューをグループ化
        seat_menus = df.groupby('seat_id')['menu_id'].apply(list).to_dict()
        
        # メニュー間の共起関係を構築
        menu_pairs = self._build_positive_pairs(df, seat_menus)
        
        # ネガティブサンプル（共起しないメニューペア）を生成
        negative_pairs = self._build_negative_pairs(df, menu_pairs)
        
        # 全データを結合して特徴量とターゲットを準備
        return self._finalize_training_data(menu_pairs, negative_pairs, df)
    
    # def _get_menu_categories(self, orders: List[Any], db: Optional[Session] = None) -> Dict[int, int]:
    #     """メニューカテゴリ情報を効率的に取得"""
    #     menu_categories = {}
    #     missing_menu_ids = set()
        
    #     # 1. 注文データからメニュー情報を抽出（Eagerロードされている場合）
    #     for order in orders:
    #         if hasattr(order, 'menu') and order.menu and hasattr(order.menu, 'category_id'):
    #             menu_categories[order.menu_id] = order.menu.category_id
    #         else:
    #             missing_menu_ids.add(order.menu_id)
        
    #     # 2. 不足しているメニュー情報のみをDBから取得
    #     if missing_menu_ids and db is not None:
    #         logging.info(f"DBから不足しているメニュー情報を取得: {len(missing_menu_ids)}件")
    #         missing_menus = db.query(Menu).filter(Menu.menu_id.in_(missing_menu_ids)).all()
    #         for menu in missing_menus:
    #             menu_categories[menu.menu_id] = menu.category_id
    #             missing_menu_ids.discard(menu.menu_id)
        
    #     # 3. それでも見つからないメニューに対する警告
    #     if missing_menu_ids:
    #         logging.warning(f"メニュー情報が見つからないメニューID: {missing_menu_ids}")
    #         # デフォルトカテゴリを割り当て
    #         for menu_id in missing_menu_ids:
    #             menu_categories[menu_id] = 0  # デフォルトカテゴリ
        
    #     logging.info(f"メニューカテゴリ情報取得完了: {len(menu_categories)}件")
    #     return menu_categories
    
    # def _create_orders_dataframe(self, orders: List[Any], menu_categories: Dict[int, int]) -> pd.DataFrame:
    #     """注文データをDataFrameに変換"""
    #     order_data = []
        
    #     for order in orders:
    #         # 必須フィールドのチェック
    #         if not hasattr(order, 'menu_id') or not hasattr(order, 'seat_id'):
    #             logging.warning(f"無効な注文データをスキップ: {order}")
    #             continue
                
    #         # メニューのカテゴリIDを取得
    #         category_id = menu_categories.get(order.menu_id, 0)  # デフォルト値0
            
    #         # 注文時刻の処理
    #         order_hour = 12  # デフォルト値
    #         if hasattr(order, 'order_date') and order.order_date:
    #             order_hour = order.order_date.hour
            
    #         # 注文数量の処理
    #         quantity = 1  # デフォルト値
    #         if hasattr(order, 'order_cnt') and order.order_cnt:
    #             quantity = order.order_cnt
            
    #         order_data.append({
    #             'seat_id': order.seat_id,
    #             'menu_id': order.menu_id,
    #             'quantity': quantity,
    #             'order_datetime': getattr(order, 'order_date', None),
    #             'hour': order_hour,
    #             'category_id': category_id
    #         })
        
    #     if not order_data:
    #         raise ValueError("有効な注文データが見つかりません")
            
    #     logging.info(f"DataFrameに変換された注文データ: {len(order_data)}件")
    #     return pd.DataFrame(order_data)
    
    def _build_positive_pairs(self, df: pd.DataFrame, seat_menus: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """ポジティブペア（共起するメニューペア）を構築"""
        menu_pairs = []
        
        for seat_id, menu_list in seat_menus.items():
            unique_menus = list(set(menu_list))  # 重複除去
            
            if len(unique_menus) < 2:
                continue  # 1つのメニューしかない場合はスキップ
            
            # 同一座席で注文されたメニューのペアを作成
            for i, menu1 in enumerate(unique_menus):
                for j, menu2 in enumerate(unique_menus):
                    if i != j:  # 異なるメニュー同士
                        # メニュー1の特徴量
                        menu1_orders = df[(df['seat_id'] == seat_id) & (df['menu_id'] == menu1)]
                        menu1_freq = len(menu1_orders)
                        menu1_avg_hour = menu1_orders['hour'].mean()
                        menu1_category = menu1_orders['category_id'].iloc[0]
                        
                        # メニュー2の特徴量
                        menu2_orders = df[(df['seat_id'] == seat_id) & (df['menu_id'] == menu2)]
                        menu2_freq = len(menu2_orders)
                        menu2_avg_hour = menu2_orders['hour'].mean()
                        menu2_category = menu2_orders['category_id'].iloc[0]
                        
                        # 時間と頻度の類似度計算（NaNチェック）
                        time_diff = abs(menu1_avg_hour - menu2_avg_hour)
                        time_similarity = 1.0 - (time_diff / 24.0) if not np.isnan(time_diff) else 0.5
                        
                        freq_max = max(menu1_freq, menu2_freq)
                        freq_similarity = min(menu1_freq, menu2_freq) / freq_max if freq_max > 0 else 0.0
                        
                        menu_pairs.append({
                            'menu1_id': menu1,
                            'menu2_id': menu2,
                            'co_occurrence': 1,  # 共起フラグ
                            'menu1_freq': menu1_freq,
                            'menu2_freq': menu2_freq,
                            'freq_similarity': freq_similarity,
                            'time_similarity': time_similarity,
                            'category_similarity': 1.0 if menu1_category == menu2_category else 0.0,
                        })
        
        logging.info(f"ポジティブペア構築完了: {len(menu_pairs)}件")
        return menu_pairs
    
    def _build_negative_pairs(self, df: pd.DataFrame, menu_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ネガティブペア（共起しないメニューペア）を構築"""
        all_menus = df['menu_id'].unique()
        negative_pairs = []
        
        if len(all_menus) < 2:
            logging.warning("メニュー数が不足しているため、ネガティブサンプルを生成できません")
            return negative_pairs
        
        # ポジティブサンプルの2倍のネガティブサンプルを生成
        positive_pairs_set = set((pair['menu1_id'], pair['menu2_id']) for pair in menu_pairs)
        
        neg_samples_needed = min(len(menu_pairs) * 2, len(all_menus) * (len(all_menus) - 1))
        attempts = 0
        max_attempts = neg_samples_needed * 5
        
        # メニュー統計の事前計算（効率化）
        menu_stats = {}
        for menu_id in all_menus:
            menu_data = df[df['menu_id'] == menu_id]
            menu_stats[menu_id] = {
                'freq': len(menu_data),
                'avg_hour': menu_data['hour'].mean(),
                'category': menu_data['category_id'].iloc[0]
            }
        
        while len(negative_pairs) < neg_samples_needed and attempts < max_attempts:
            menu1 = random.choice(all_menus)
            menu2 = random.choice(all_menus)
            
            if menu1 != menu2 and (menu1, menu2) not in positive_pairs_set:
                menu1_stats = menu_stats[menu1]
                menu2_stats = menu_stats[menu2]
                
                # 特徴量計算（NaNチェック）
                time_diff = abs(menu1_stats['avg_hour'] - menu2_stats['avg_hour'])
                time_similarity = 1.0 - (time_diff / 24.0) if not np.isnan(time_diff) else 0.5
                
                freq_max = max(menu1_stats['freq'], menu2_stats['freq'])
                freq_similarity = min(menu1_stats['freq'], menu2_stats['freq']) / freq_max if freq_max > 0 else 0.0
                
                negative_pairs.append({
                    'menu1_id': menu1,
                    'menu2_id': menu2,
                    'co_occurrence': 0,  # 非共起フラグ
                    'menu1_freq': menu1_stats['freq'],
                    'menu2_freq': menu2_stats['freq'],
                    'freq_similarity': freq_similarity,
                    'time_similarity': time_similarity,
                    'category_similarity': 1.0 if menu1_stats['category'] == menu2_stats['category'] else 0.0,
                })
            
            attempts += 1
        
        logging.info(f"ネガティブペア構築完了: {len(negative_pairs)}件（{attempts}回の試行）")
        return negative_pairs
    
    def _finalize_training_data(
        self, 
        menu_pairs: List[Dict[str, Any]], 
        negative_pairs: List[Dict[str, Any]], 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """データをエンコードして最終化"""
        # 全データを結合
        all_pairs = menu_pairs + negative_pairs
        
        if not all_pairs:
            raise ValueError("学習用データが生成されませんでした")
            
        pairs_df = pd.DataFrame(all_pairs)
        
        # メニューIDをエンコード
        unique_menus = sorted(df['menu_id'].unique())
        self.menu_encoder.fit(unique_menus)
        
        pairs_df['menu1_encoded'] = self.menu_encoder.transform(pairs_df['menu1_id'])
        pairs_df['menu2_encoded'] = self.menu_encoder.transform(pairs_df['menu2_id'])
        
        # データをシャッフル
        pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 特徴量の正規化（NaN値の処理）
        feature_columns = ['freq_similarity', 'time_similarity', 'category_similarity']
        feature_matrix = pairs_df[feature_columns].fillna(0.0).values.astype(np.float32)
        
        logging.info(f"学習データ準備完了:")
        logging.info(f"  - ポジティブサンプル: {len(menu_pairs)}件")
        logging.info(f"  - ネガティブサンプル: {len(negative_pairs)}件")
        logging.info(f"  - 総メニュー数: {len(unique_menus)}")
        logging.info(f"  - 特徴量次元: {feature_matrix.shape[1]}")
        
        return (
            pairs_df['menu1_encoded'].values.astype(np.int64),
            pairs_df['menu2_encoded'].values.astype(np.int64),
            feature_matrix,
            pairs_df['co_occurrence'].values.astype(np.float32),
            np.zeros_like(pairs_df['co_occurrence'].values.astype(np.float32))  # 予備用配列
        )
