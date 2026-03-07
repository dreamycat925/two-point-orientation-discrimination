# two-point-orientation-discrimination

JVP dome を用いた two-point orientation discrimination のための
Streamlit ベース検査補助アプリです。

## Files

- `two_point_orientation_discrimination_streamlit_app.py` : アプリ本体
- `requirements.txt` : Python 依存関係
- `Dockerfile` : ローカル Docker 実行用
- `docker-compose.yml` : 通常運用用
- `docker-compose.dev.yml` : 開発用
- `README_DOCKER.md` : Docker 実行手順

## Local run (without Docker)

```bash
pip install -r requirements.txt
streamlit run two_point_orientation_discrimination_streamlit_app.py
```

## Local run (with Docker)

```bash
docker compose up --build -d
```

開くURL:

```text
http://localhost:40000
```

## Streamlit Community Cloud

GitHub のこのリポジトリを Streamlit Community Cloud に接続し、
entrypoint として `two_point_orientation_discrimination_streamlit_app.py` を指定してください。
