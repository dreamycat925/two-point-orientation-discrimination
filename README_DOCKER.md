# Docker / Docker Compose (local-only)

このリポジトリには、two-point orientation discrimination の Streamlit アプリを
ローカルで安定して動かすための Docker / Docker Compose 設定を入れています。

## 置く場所
以下のファイルを、`two_point_orientation_discrimination_streamlit_app.py` と同じ階層に置きます。

- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `.dockerignore`

## 1. 安定運用（通常はこちら）

```bash
docker compose up --build -d
```

開くURL:

```text
http://localhost:40000
```

停止:

```bash
docker compose down
```

## 2. 開発モード（live edit）

```bash
docker compose -f docker-compose.dev.yml up --build
```

このモードでは、カレントディレクトリをコンテナに mount するので、
ファイル編集が再 build なしで反映されやすくなります。

## メモ

- ホスト側は `127.0.0.1:40000` に bind しています。LAN からは見えません。
- コンテナ内では Streamlit を `0.0.0.0:8501` で待ち受けます。
