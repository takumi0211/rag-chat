# ベースイメージとして公式のPython 3.10 (slim版) を使用
FROM python:3.10-slim

# 環境変数: Pythonの出力をバッファリングしない (ログが見やすくなる)
ENV PYTHONUNBUFFERED True

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルをコピー
COPY requirements.txt requirements.txt

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードとvectorstoreをコピー
COPY . .

# vectorstoreディレクトリと主要ファイルが存在するか確認 (ビルド時チェック)
RUN if [ ! -d "vectorstore" ] || [ ! -f "vectorstore/index.faiss" ] || [ ! -f "vectorstore/index.pkl" ]; then \
      echo "Error: vectorstore directory or its contents (index.faiss, index.pkl) not found." >&2; \
      exit 1; \
    fi

# アプリケーションの実行コマンド (Gunicornを使用)
# ポート5001でリッスンするよう設定 (app.pyに合わせる)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]