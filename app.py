from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from chat import query_rag  # chat.pyからquery_rag関数をインポート

app = Flask(__name__)
Bootstrap(app)  # インスタンスを変数に代入しない

@app.route('/')
def home():
    return render_template('index.html', bootstrap=Bootstrap())  # bootstrap変数を渡す

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': '質問が入力されていません'}), 400

    try:
        response = query_rag(question)  # chatオブジェクトを使わずに直接関数を呼び出す
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('Flaskサーバーを開始します...')
    app.run(host='0.0.0.0', port=5001, debug=True)
