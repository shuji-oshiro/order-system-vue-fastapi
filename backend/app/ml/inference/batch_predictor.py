"""
バッチ推論クラス

大量のデータに対する効率的なバッチ推論を実行します。
"""
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Iterator
from torch.utils.data import DataLoader, TensorDataset


class BatchPredictor:
    """バッチ推論実行クラス"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, batch_size: int = 256):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # モデルを評価モードに設定
        self.model.eval()
        
    def predict_batch(
        self, 
        menu1_ids: List[int], 
        menu2_ids: List[int], 
        features: List[List[float]]
    ) -> List[float]:
        """
        バッチで推論を実行
        
        Args:
            menu1_ids: 基準メニューIDのリスト
            menu2_ids: 対象メニューIDのリスト
            features: 特徴量のリスト
            
        Returns:
            予測スコアのリスト
        """
        if len(menu1_ids) != len(menu2_ids) or len(menu1_ids) != len(features):
            raise ValueError("入力データのサイズが一致しません")
        
        logging.info(f"バッチ推論開始: {len(menu1_ids)}件のサンプル")
        
        # データセットを作成
        dataset = TensorDataset(
            torch.LongTensor(menu1_ids),
            torch.LongTensor(menu2_ids),
            torch.FloatTensor(features)
        )
        
        # データローダーを作成
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch_menu1, batch_menu2, batch_features in dataloader:
                # データをデバイスに移動
                batch_menu1 = batch_menu1.to(self.device)
                batch_menu2 = batch_menu2.to(self.device)
                batch_features = batch_features.to(self.device)
                
                # 推論実行
                batch_predictions = self.model.forward(batch_menu1, batch_menu2, batch_features)
                
                # CPUに移動してリストに追加
                predictions.extend(batch_predictions.cpu().numpy().tolist())
        
        logging.info(f"バッチ推論完了: {len(predictions)}件の予測")
        return predictions
    
    def predict_menu_pairs_bulk(
        self, 
        menu_pairs: List[tuple], 
        features: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        メニューペアのリストに対してバッチ推論を実行
        
        Args:
            menu_pairs: (menu1_id, menu2_id)のタプルリスト
            features: 各ペアの特徴量リスト
            
        Returns:
            結果の辞書リスト
        """
        menu1_ids = [pair[0] for pair in menu_pairs]
        menu2_ids = [pair[1] for pair in menu_pairs]
        
        # バッチ推論実行
        scores = self.predict_batch(menu1_ids, menu2_ids, features)
        
        # 結果を整形
        results = []
        for i, (menu1_id, menu2_id) in enumerate(menu_pairs):
            results.append({
                'menu1_id': menu1_id,
                'menu2_id': menu2_id,
                'relationship_score': scores[i],
                'features': {
                    'freq_similarity': features[i][0],
                    'time_similarity': features[i][1],
                    'category_similarity': features[i][2]
                }
            })
        
        return results
    
    def generate_recommendations_bulk(
        self, 
        base_menu_ids: List[int], 
        candidate_menu_ids: List[int],
        top_k: int = 5
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        複数の基準メニューに対してバッチで推薦を生成
        
        Args:
            base_menu_ids: 基準メニューIDのリスト
            candidate_menu_ids: 候補メニューIDのリスト
            top_k: 各基準メニューに対する推薦数
            
        Returns:
            基準メニューIDをキーとした推薦結果の辞書
        """
        logging.info(f"バルク推薦開始: 基準メニュー{len(base_menu_ids)}件, 候補{len(candidate_menu_ids)}件")
        
        all_pairs = []
        all_features = []
        pair_to_base = {}  # ペアインデックスから基準メニューIDへのマッピング
        
        # 全ペアの組み合わせを生成
        pair_index = 0
        for base_id in base_menu_ids:
            for candidate_id in candidate_menu_ids:
                if base_id != candidate_id:
                    all_pairs.append((base_id, candidate_id))
                    all_features.append([0.5, 0.5, 0.0])  # デフォルト特徴量
                    pair_to_base[pair_index] = base_id
                    pair_index += 1
        
        # バッチ推論実行
        results = self.predict_menu_pairs_bulk(all_pairs, all_features)
        
        # 基準メニューごとに結果を整理
        recommendations = {}
        for i, result in enumerate(results):
            base_id = pair_to_base[i]
            if base_id not in recommendations:
                recommendations[base_id] = []
            
            recommendations[base_id].append({
                'menu_id': result['menu2_id'],
                'relationship_score': result['relationship_score'],
                'features': result['features']
            })
        
        # 各基準メニューの推薦をスコア順でソートし、top_kを選択
        for base_id in recommendations:
            recommendations[base_id].sort(key=lambda x: x['relationship_score'], reverse=True)
            recommendations[base_id] = recommendations[base_id][:top_k]
        
        logging.info(f"バルク推薦完了: {len(recommendations)}件の基準メニュー")
        return recommendations
