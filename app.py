"""
Flask 메인 앱: 김치프리미엄 멀티 자산 대시보드
"""

import logging
import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from data_fetcher import get_all_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data.json")
def data_json():
    """리포 루트 data.json 서빙 — 로컬 미리보기 전용(배포는 정적 GitHub Pages가 직접 서빙)."""
    response = send_from_directory(BASE_DIR, "data.json")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/config.json")
def config_json():
    """리포 루트 config.json 서빙 — 로컬 미리보기 전용(배포는 정적 GitHub Pages가 직접 서빙)."""
    response = send_from_directory(BASE_DIR, "config.json")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/api/data")
def api_data():
    try:
        force_refresh = request.args.get("force") in {"1", "true", "yes"}
        data = get_all_data(force_refresh=force_refresh)
        return jsonify(data)
    except Exception as e:
        logging.exception("Data fetch failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
