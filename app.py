"""
Flask 메인 앱: 국내-국제 금 가격 괴리율 웹사이트
"""

import logging

from flask import Flask, jsonify, render_template

from data_fetcher import get_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    try:
        data = get_data()
        return jsonify(data)
    except Exception as e:
        logging.exception("Data fetch failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
