# order-system-vue-fastapi

## 概要

飲食店向けの注文管理アプリケーションのプロトタイプです。  
音声認識、レコメンド機能、Vuetifyによるモダンなデザインの機能検証

## 特徴

- Vue 3 + Vuetify 
- FastAPI + SQLite + SQLAlchemy
- Whisper | vosk
- **レコメンド機能**: 5段階のフェーズで段階的に高度化されたメニュー推薦システム

## 環境
- Python 3.13+
- Node.js 18+
- `uv` (`pip install uv`)
- `npm` / `pnpm`


## スクリーンショット


## デモ 仮想環境

```bash
python -m venv .venv
source backend/.venv/Scripts/activate

source backend/.venv/Scripts/activate  # Windows (Git Bash)
source backend/.venv/bin/activate       # macOS/Linux"

cp .env.example .env　 #開発用の環境変数コピー

setup.sh

npm run dev

python backend/scripts/insert_test_data.py　# 初期起動時はスクリプトダミーデータを追加

```

## レコメンド機能（検証中）

本アプリケーションは、過去の注文履歴データを分析して最適なメニューを推薦する高度なレコメンド機能を搭載しています。

詳細な設計思想と各フェーズの考え方については、**[レコメンドシステム設計書](./RECOMMEND_SYSTEM.md)** をご参照ください。

AI学習アルゴリズムの詳細設計については、**[AI学習アルゴリズム設計書](./AI_RECOMMEND_DESIGN.md)** をご参照ください。

### 概要

5段階のフェーズで段階的に高度化されたアルゴリズムを提供：

- **Phase 1**: 頻度ベースレコメンド（基本）
- **Phase 2**: 時間帯・カテゴリ親和性レコメンド（中級）
- **Phase 3**: 価格帯考慮レコメンド（上級）
- **Phase 4**: 複合スコアリングシステム（最上級）
- **Phase 5**: AI学習アルゴリズム（Neural Collaborative Filtering - メニュー間関連性学習）

### API使用方法

```bash
# Phase 1: 基本的な頻度ベース推薦
GET /recommend/{menu_id}?phase=1

# Phase 2: 時間帯・カテゴリ考慮推薦  
GET /recommend/{menu_id}?phase=2

# Phase 3: 価格帯考慮推薦
GET /recommend/{menu_id}?phase=3

# Phase 4: 複合スコアリング推薦
GET /recommend/{menu_id}?phase=4

# Phase 5: AI学習アルゴリズム推薦
GET /recommend/{menu_id}?phase=5

```

## 開発ステータス

このプロジェクトは現在も開発中です。  
実用的な飲食店向けアプリケーション技術を検証中

**完了済み機能**:
- ✅ レコメンド機能 Phase 1-4 の完全実装
- ✅ レコメンドAPI の実装とテスト
- ✅ 注文履歴データの蓄積と分析機能
- ✅ Phase 5: AI学習アルゴリズム（メニュー間関連性学習）の実装
  - メニュー×メニューの関連性学習
  - 注文頻度・時間帯・カテゴリ類似度を特徴量として活用
  - ニューラルネットワークによる非線形関係性の学習

今後の予定：
- 

## 作者

**Shuji Oshiro**  
- GitHub: [@shuji-oshiro](https://github.com/shuji-oshiro)  

##ライセンス情報

本プロジェクトは [MIT License](./LICENSE) のもとで公開されています。

### 使用ライブラリとライセンス

以下の外部ライブラリを使用しており、各ライセンス条件に従っています。

- [Whisper (OpenAI)](https://github.com/openai/whisper) - MIT License
- [Vosk (AlphaCep)](https://github.com/alphacep/vosk-api) - Apache License 2.0
- [Vuetify](https://github.com/vuetifyjs/vuetify) - MIT License
- [ffmpeg](https://ffmpeg.org/) - LGPL/GPL ライセンス（再配布時に注意）
- その他：FastAPI, SQLAlchemy, sounddevice なども MIT / Apache系

詳細は各ライブラリの公式ページおよび `LICENSE`, `NOTICE` を参照してください。
