# Vuetify Order Control App

## 🎯 概要（What）

飲食店向けの注文管理アプリです。音声認識とレコメンド機能を搭載し、注文を効率化します。

## 🚀 特徴（Why）

- Vue 3 + Vuetify によるモダンなUI
- FastAPI + SQLite による軽量なバックエンド
- Whisper を用いた音声認識
- 注文履歴・レコメンドの表示

## 🛠 使用技術（How）

| フロントエンド | バックエンド | その他 |
|----------------|--------------|--------|
| Vue.js 3       | FastAPI      | Whisper |
| Pinia（状態管理） | SQLAlchemy | Docker（検証用） |

## 📷 スクリーンショット

|----------|

## 💻 デモ

```bash
git clone ...
cd project/vuetify-pj-order_control
python main.py

```markdown

## 🚧 開発ステータス

このプロジェクトは現在も開発中です。  
音声認識・レコメンド・注文管理といった複数の機能を組み合わせ、実用的な飲食店向けアプリケーションのプロトタイプ構築を目指しています。

今後の予定：
- [x] 基本UIの構築（Vuetify）
- [x] Whisperによる音声認識機能
- [x] FastAPIによるバックエンド連携
- [ ] 注文履歴の一覧表示
- [ ] ユーザーごとのレコメンド強

## 👨‍💻 作者

**大城 修二（Shuji Oshiro）**  
- GitHub: [@shuji-oshiro](https://github.com/shuji-oshiro)  
- 経歴：IT専門職を経て、食品メーカーにて商品企画・営業・マーケティング部門を統括。自社の業務課題をITで解決するため、業務分析・要件定義・システム構築を内製で推進。  
- 技術：FastAPI、Vue 3、SQLAlchemy、Whisper、Pinia、Dockerなど  
- 特徴：非エンジニア領域での深い業務理解を武器に、現場のニーズに即したアプリ設計・構築が可能。UI/UXとデータ設計の両面を考慮し、課題解決に直結するプロトタイプを迅速に開発。  
- 本アプリは「飲食店現場の省力化」を想定し、音声認識 × レコメンドによる注文支援を実現。バックエンドからフロントまで全工程を一人で構築。
