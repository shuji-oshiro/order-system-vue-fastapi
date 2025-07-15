# ML フォルダ構成リファクタリング提案

## 🎯 提案する新しい構成

```
backend/app/ml/
├── __init__.py
├── base/                           # 基底クラス・共通機能
│   ├── __init__.py
│   ├── abstract_model.py           # BaseRecommendModel (現在のinference/base_model.py)
│   ├── pytorch_base.py             # PyTorchBaseModel (現在のmodels/py_torch_base_model.py)
│   └── config.py                   # 共通設定・定数
├── models/                         # 実装モデル定義
│   ├── __init__.py
│   ├── neural_cf/                  # Neural Collaborative Filtering
│   │   ├── __init__.py
│   │   ├── model.py                # モデル定義 (現在のtraining/neural_cf.py)
│   │   ├── config.py               # モデル固有設定
│   │   └── README.md               # モデル説明
│   ├── matrix_factorization/       # 将来の拡張用
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── config.py
│   └── ensemble/                   # アンサンブルモデル
│       ├── __init__.py
│       └── model.py
├── data/                           # データ処理関連
│   ├── __init__.py
│   ├── preprocessing.py            # データ前処理
│   ├── feature_engineering.py     # 特徴量エンジニアリング
│   ├── loaders.py                  # データローダー
│   └── cache.py                    # キャッシュ管理
├── training/                       # 学習関連
│   ├── __init__.py
│   ├── trainer.py                  # 汎用学習クラス
│   ├── callbacks.py                # 学習コールバック
│   ├── schedulers.py               # 学習率スケジューラー
│   └── metrics.py                  # 評価指標
├── inference/                      # 推論関連
│   ├── __init__.py
│   ├── predictor.py                # 推論実行クラス
│   ├── batch_predictor.py          # バッチ推論
│   └── model_loader.py             # モデル読み込み
├── evaluation/                     # 評価・実験
│   ├── __init__.py
│   ├── evaluator.py                # モデル評価
│   ├── cross_validation.py         # クロスバリデーション
│   ├── benchmarks.py               # ベンチマーク
│   └── experiment_tracker.py       # 実験管理
├── utils/                          # ユーティリティ
│   ├── __init__.py
│   ├── device_manager.py           # GPU/CPU管理
│   ├── model_utils.py              # モデル関連ユーティリティ
│   ├── logging_config.py           # ログ設定
│   └── visualization.py            # 可視化ツール
├── hyperparameter/                 # ハイパーパラメータ調整
│   ├── __init__.py
│   ├── tuner.py                    # ハイパーパラメータチューナー
│   ├── search_spaces.py            # 探索空間定義
│   └── optimization.py             # 最適化アルゴリズム
├── saved_models/                   # 保存モデル
│   ├── neural_cf/
│   │   ├── latest.pth
│   │   ├── best_model.pth
│   │   └── checkpoints/
│   └── ensemble/
│       └── latest.pth
├── experiments/                    # 実験結果
│   ├── logs/
│   ├── configs/
│   └── results/
└── docs/                          # ドキュメント
    ├── TRAINING_PARAMETERS_GUIDE.md (現在のmodels/TRAINING_PARAMETERS_GUIDE.md)
    ├── MODEL_ARCHITECTURE.md
    ├── DEPLOYMENT_GUIDE.md
    └── API_REFERENCE.md
```

## 🔄 移行計画

### Phase 1: 基底クラスの整理
1. `ml/base/` フォルダ作成
2. `inference/base_model.py` → `base/abstract_model.py`
3. `models/py_torch_base_model.py` → `base/pytorch_base.py`

### Phase 2: モデル実装の整理
1. `ml/models/neural_cf/` フォルダ作成
2. `training/neural_cf.py` → `models/neural_cf/model.py`
3. モデル固有設定の分離

### Phase 3: 機能別フォルダの追加
1. `data/`, `evaluation/`, `utils/` フォルダ作成
2. 既存機能の適切な場所への移動
3. 新機能の実装

### Phase 4: ドキュメント整理
1. `docs/` フォルダ作成
2. 既存ドキュメントの移動・整理

## 💡 各フォルダの役割

### `base/` - 基底クラス・共通機能
- **目的**: 全モデル共通の抽象クラスと基本機能
- **メリット**: 新しいモデル追加時の統一インターフェース

### `models/` - モデル実装
- **目的**: 具体的なML模型の実装
- **メリット**: モデル毎に独立したフォルダで管理、拡張性向上

### `data/` - データ処理
- **目的**: データの前処理、特徴量エンジニアリング
- **メリット**: データ処理ロジックの中央集約

### `training/` - 学習プロセス
- **目的**: 学習アルゴリズム、コールバック、メトリクス
- **メリット**: 学習プロセスの標準化

### `inference/` - 推論プロセス
- **目的**: 学習済みモデルを使った予測・推薦
- **メリット**: 本番環境での推論最適化

### `evaluation/` - 評価・実験
- **目的**: モデル性能評価、A/Bテスト、実験管理
- **メリット**: 科学的なモデル比較・改善

## 🚀 期待される効果

### 1. **開発効率の向上**
- 目的別フォルダ分けで開発者が迷わない
- 新機能追加時の配置場所が明確

### 2. **コードの再利用性**
- 共通機能の中央集約
- モデル間での機能共有が容易

### 3. **保守性の向上**
- 責任の明確な分離
- テストしやすい構造

### 4. **スケーラビリティ**
- 新しいモデルの追加が容易
- チーム開発での競合回避

### 5. **MLOps対応**
- 実験管理の仕組み
- モデルバージョニング

## 📋 実装手順

1. **新フォルダ構造の作成**
2. **既存ファイルの段階的移行**
3. **import文の更新**
4. **テストケースの作成**
5. **ドキュメント更新**

## ⚠️ 注意事項

- 既存APIとの互換性維持
- 段階的移行でサービス停止回避
- 十分なテストでリグレッション防止
