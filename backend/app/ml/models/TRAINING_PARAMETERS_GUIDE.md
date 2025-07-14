# Neural Collaborative Filtering 学習パラメータ設定ガイド

## 📊 学習パラメータの詳細解説

### 🎯 基本パラメータ

| パラメータ | デフォルト値 | 説明 | 推奨範囲 | 影響 |
|-----------|-------------|------|----------|------|
| `epochs` | 100 | 最大学習エポック数 | 50-200 | 学習時間、過学習リスク |
| `batch_size` | 256 | バッチサイズ | 64-512 | メモリ使用量、学習安定性 |
| `learning_rate` | 0.001 | 学習率 | 0.0001-0.01 | 収束速度、学習安定性 |
| `test_size` | 0.2 | テストデータ割合 | 0.15-0.25 | 評価精度、学習データ量 |
| `patience` | 10 | 早期停止許容エポック数 | 5-20 | 過学習防止、学習時間 |

### 🔧 パラメータの詳細説明

#### 1. **epochs (学習エポック数)**
```python
epochs = kwargs.get('epochs', 100)  # デフォルト: 100
```
- **意味**: 全データセットを何回繰り返し学習するか
- **学習曲線**: epochs ↑ → 訓練精度 ↑、しかし過学習リスク ↑
- **早期停止**: patienceパラメータで自動的に最適点で停止
- **計算時間**: epochs × バッチ数 × フォワード・バックプロパゲーション時間

**推奨設定**:
- **小規模データ**: 50-100 epochs
- **中規模データ**: 100-200 epochs  
- **大規模データ**: 200+ epochs（早期停止に依存）

#### 2. **batch_size (バッチサイズ)**
```python
batch_size = kwargs.get('batch_size', 256)  # デフォルト: 256
```
- **意味**: 1回の勾配更新で処理するサンプル数
- **メモリ影響**: batch_size ↑ → GPU/CPUメモリ使用量 ↑
- **学習安定性**: batch_size ↑ → 勾配推定が安定、学習が滑らか
- **計算効率**: バッチ処理により並列計算効率が向上

**GPU メモリ別推奨値**:
- **4GB GPU**: batch_size = 64-128
- **8GB GPU**: batch_size = 128-256  
- **16GB+ GPU**: batch_size = 256-512
- **CPU環境**: batch_size = 32-64

#### 3. **learning_rate (学習率)**
```python
learning_rate = kwargs.get('learning_rate', 0.001)  # デフォルト: 0.001
```
- **意味**: パラメータ更新の幅（ステップサイズ）
- **Adam最適化器**: 適応的学習率なので0.001が標準的
- **収束性**: learning_rate ↑ → 高速だが不安定、↓ → 安定だが低速

**学習率スケジューリング例**:
```python
# 高い学習率で開始し、徐々に下げる
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

#### 4. **test_size (テストデータ割合)**
```python
test_size = kwargs.get('test_size', 0.2)  # デフォルト: 20%
```
- **意味**: 全データに対するテストデータの比率
- **層化分割**: `stratify=ratings`で正例・負例の比率を保持
- **評価信頼性**: test_size ↑ → 評価精度 ↑、しかし学習データ ↓

**データサイズ別推奨値**:
- **小規模 (<1万件)**: test_size = 0.25 (25%)
- **中規模 (1-10万件)**: test_size = 0.2 (20%)
- **大規模 (>10万件)**: test_size = 0.15 (15%)

#### 5. **patience (早期停止許容値)**
```python
patience = kwargs.get('patience', 10)  # デフォルト: 10
```
- **意味**: バリデーション損失が改善しない連続エポック数の上限
- **過学習防止**: 自動的に最適なタイミングで学習を停止
- **計算効率**: 無駄な学習時間を削減

**設定指針**:
- **patience ↑**: より慎重に停止判定（学習時間 ↑）
- **patience ↓**: 早めに停止（学習時間 ↓、最適化不足リスク）

### 📈 パラメータチューニング戦略

#### **段階的最適化アプローチ**

1. **ベースライン設定**
```python
# 初期設定で動作確認
result = model.train(
    db=session,
    epochs=50,          # 少なめで動作確認
    batch_size=128,     # 標準的なサイズ
    learning_rate=0.001,
    test_size=0.2,
    patience=10
)
```

2. **バッチサイズ最適化**
```python
# GPUメモリに合わせて調整
for batch_size in [64, 128, 256, 512]:
    result = model.train(db=session, batch_size=batch_size)
    print(f"Batch Size {batch_size}: Accuracy = {result['test_accuracy']:.3f}")
```

3. **学習率探索**
```python
# 学習率の影響を確認
for lr in [0.0001, 0.001, 0.01]:
    result = model.train(db=session, learning_rate=lr)
    print(f"LR {lr}: Final Loss = {result['final_test_loss']:.4f}")
```

### 🎯 実際の使用例

#### **高精度重視設定**
```python
# 精度を最重視する場合
result = model.train(
    db=session,
    epochs=200,           # 多めのエポック数
    batch_size=512,       # 大きなバッチサイズで安定化
    learning_rate=0.0005, # やや小さめの学習率
    test_size=0.25,       # 多めのテストデータで信頼性向上
    patience=20           # 慎重な早期停止
)
```

#### **高速学習重視設定**
```python
# 学習速度を重視する場合
result = model.train(
    db=session,
    epochs=100,          # 標準的なエポック数
    batch_size=256,      # バランスの取れたバッチサイズ
    learning_rate=0.002, # やや大きめの学習率で高速化
    test_size=0.15,      # 少なめのテストデータで学習データを多く
    patience=8           # 早めの早期停止
)
```

#### **メモリ制約環境設定**
```python
# メモリが限られた環境
result = model.train(
    db=session,
    epochs=150,          # エポック数を増やして補償
    batch_size=64,       # 小さなバッチサイズ
    learning_rate=0.001, # 標準的な学習率
    test_size=0.2,       # 標準的なテスト割合
    patience=15          # やや長めの待機
)
```

### ⚡ パフォーマンス最適化のコツ

#### **1. 動的バッチサイズ調整**
```python
import torch
# 利用可能メモリに応じて調整
available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
optimal_batch_size = min(512, int(available_memory / 5e7))  # 経験的な計算
```

#### **2. 学習率スケジューリング**
```python
# エポック数に応じて学習率を調整
def get_optimal_lr(epoch, base_lr=0.001):
    if epoch < 30:
        return base_lr
    elif epoch < 60:
        return base_lr * 0.1
    else:
        return base_lr * 0.01
```

#### **3. 適応的patience調整**
```python
# データサイズに応じてpatience調整
data_size = len(user_ids)
adaptive_patience = max(5, min(20, data_size // 1000))
```

### 📊 パラメータ監視指標

学習中にこれらの指標を監視してパラメータを調整:

- **train_loss vs test_loss**: 差が大きい → 過学習 → patience ↓, learning_rate ↓
- **学習速度**: 遅い → learning_rate ↑, batch_size ↑  
- **メモリ使用量**: 高い → batch_size ↓
- **精度**: 低い → epochs ↑, learning_rate調整, test_size ↑

これらのパラメータを適切に設定することで、Neural Collaborative Filteringモデルの性能を最大化できます。
