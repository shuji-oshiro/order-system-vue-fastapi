# order-system-vue-fastapi

## 概要

飲食店向けの注文管理アプリケーションのプロトタイプです。  
音声認識、レコメンド機能、Vuetifyによるモダンなデザインの機能検証

## 特徴

- Vue 3 + Vuetify 
- FastAPI + SQLite + SQLAlchemy
- Whisper | vosk

## 環境
- Python 3.11+
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

setup.sh

npm run dev
```

## 初期起動時はスクリプトダミーデータを追加
python backend/scripts/insert_test_data.py


## 開発ステータス

このプロジェクトは現在も開発中です。  
実用的な飲食店向けアプリケーション技術を検証中

今後の予定：
- [ ] 注文履歴の一覧表示
- [ ] 注文履歴に基づいたレコメンド機能

## 作者

**Shuji Oshiro**  
- GitHub: [@shuji-oshiro](https://github.com/shuji-oshiro)  
