from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama # Ollamaを削除
from langchain_openai import ChatOpenAI # ChatOpenAI をインポート
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Prompt Template (変更なし)
PROMPT_TEMPLATE = """
以下のコンテキストに基づいて、うまく活用させながら質問に答えてください：

{context}

---

上記のコンテキストに基づいて、次の質問に答えてください：{question}
"""

def query_rag(query_text: str) -> dict:
    """
    RAGパイプラインを実行し、指定された質問に対する回答とソースを取得します。
    LLMとしてGPT-4oを使用します。
    """
    try:
        # 1. EmbeddingsとVectorstoreのロード (変更なし)
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # vectorstoreが存在するか確認
        if not os.path.exists("vectorstore"):
             raise FileNotFoundError("The 'vectorstore' directory does not exist. Please ensure it's created and populated.")
        db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)

        # 2. Retrieverの準備と関連ドキュメントの取得 (変更なし)
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents(query_text)
        if not docs:
            print(f"Warning: No relevant documents found for query: '{query_text}'")
            # コンテキストなしで続行するか、エラーを返すかを選択できます
            # ここではコンテキストなしで続行します
            context_text = "利用可能なコンテキストはありません。"
        else:
            # 取得したドキュメントからコンテキストを作成 (上位5件を使用)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in docs[:5]])

        # 3. プロンプトの準備 (変更なし)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # 4. LLMの初期化 (Ollama -> ChatOpenAI with gpt-4o)
        # model = Ollama(model="gemma3:12b") # 古いOllamaモデル
        model = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o", # モデルをgpt-4oに変更
            temperature=0.7 # 必要に応じて温度を設定 (0は決定論的な出力)
        )

        # 5. LLMの実行と応答の取得
        response = model.invoke(prompt)

        # 6. ソースの準備 (変更なし、ただしdocsが空の場合を考慮)
        sources = []
        if docs:
            sources = [{"title": doc.metadata.get('title', 'N/A'), "url": doc.metadata.get('source', 'N/A')} for doc in docs[:5]]
            
        # 🔥 printログ（Cloud Runログに出る）
        print("🧑‍💬 User query:", query_text)
        print("🤖 AI answer:", response.content)
        print("📚 Sources:", sources)

        # 7. 結果の返却 (response.contentを使用)
        return {
            # "answer": response, # Ollamaの応答は通常文字列だが、ChatModelはAIMessageオブジェクト
            "answer": response.content, # AIMessageオブジェクトからテキスト内容を取得
            "sources": sources
        }

    except FileNotFoundError as e:
        print(f"Error: {e}")
        # エラー発生時に返す情報（例）
        return {"answer": "エラー: ベクターストアが見つかりません。", "sources": []}
    except Exception as e:
        print(f"An unexpected error occurred in query_rag: {e}")
        # その他の予期せぬエラー発生時に返す情報（例）
        return {"answer": f"エラーが発生しました: {e}", "sources": []}