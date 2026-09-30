"""Value Compass 생태계 연동 구조 계약 — 셸 태그·벤더링 자산·테마 부트 마커·배포 산출물.

행위(테마 토글·교차 링크)는 tests/js/ecosystem.test.mjs가 검증한다. 여기서는 마크업/배포 설정이
서로 어긋나지 않는지만 문자열로 고정한다(허브 sync-ecosystem.mjs verify와 같은 기준).
"""

import json
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_TAG = "?v=20260930-vc"


def _read(*parts):
    with open(os.path.join(REPO_ROOT, *parts), "r", encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def html():
    return _read("templates", "index.html")


def _head(html):
    return html[: html.index("</head>")]


def test_theme_boot_block_is_filled_once_and_precedes_stylesheets(html):
    head = _head(html)
    assert head.count("<!-- vc:theme-boot -->") == 1
    assert head.count("<!-- /vc:theme-boot -->") == 1
    start = head.index("<!-- vc:theme-boot -->")
    end = head.index("<!-- /vc:theme-boot -->")
    block = head[start:end]
    assert "vc-theme-boot v1" in block  # sync-ecosystem.mjs --write가 채운 내용
    assert "prefers-color-scheme" in block and "'theme'" in block and "preferred-theme" in block
    first_css = min(head.index("<style"), head.index('rel="stylesheet"'))
    assert start < first_css
    # 구 자체 테마 부트 스크립트는 제거됐다(테마 결정은 공용 블록 하나)
    assert "localStorage.getItem('preferred-theme')" not in head


def test_tokens_before_own_css_and_shell_script_deferred(html):
    head = _head(html)
    tokens = head.index(f'href="./static/vc-tokens.css{VERSION_TAG}"')
    own = head.index(f'href="static/style.css{VERSION_TAG}"')
    assert tokens < own
    assert re.search(r'<script defer src="\./static/vc-shell\.js\?v=20260930-vc"></script>', head)


def test_shell_tag_is_first_in_body_with_hub_link_fallback(html):
    body = html[html.index("<body>") + len("<body>"):]
    first_tag = re.search(r"<(?!!--)([a-zA-Z-]+)", body)
    assert first_tag.group(1) == "vc-shell"
    m = re.search(r'<vc-shell tool="gold_gap">(.*?)</vc-shell>', body, re.S)
    assert m, "vc-shell tag missing"
    assert 'class="hub-link"' in m.group(1) and "https://ducklove.duckdns.org:3691" in m.group(1)
    # 헤더의 수제 허브 링크는 셸 폴백으로 옮겨졌다 — 페이지 전체에 하나만
    assert html.count('class="hub-link"') == 1


def test_related_all_about_gold_link(html):
    m = re.search(r'<a class="related-link" id="relatedAllAboutGold"\s+href="([^"]+)"', html)
    assert m and m.group(1) == "https://ducklove.github.io/all-about-gold/"
    main_js = _read("static", "js", "main.js")
    assert "bindSiblingLink(document.getElementById('relatedAllAboutGold')" in main_js


@pytest.mark.parametrize("name", ["vc-shell.js", "vc-tokens.css"])
def test_vendored_assets_present(name):
    text = _read("static", name)
    assert "value-invest" in text.splitlines()[0] or "value-invest" in text[:400]
    if name == "vc-shell.js":
        assert "window.VCShell" in text and '"gold_gap"' in text  # 인라인 레지스트리에 이 도구 포함
    else:
        assert "--vc-up" in text and "--vc-down" in text and "--vc-font-sans" in text


def test_vendored_python_helper_header():
    first = _read("goldgap", "vc_publish.py").splitlines()[0]
    assert first.startswith("# vendored from value-invest")


def test_style_aliases_direction_colours_and_font():
    css = _read("static", "style.css")
    alias = re.search(r':root,\s*:root\[data-theme="light"\],\s*:root\[data-theme="dark"\]\s*\{([^}]*)\}', css)
    assert alias, "direction-colour alias block missing"
    assert "--up: var(--vc-up" in alias.group(1) and "--down: var(--vc-down" in alias.group(1)
    # alias 블록은 자체 라이트/다크 정의 뒤에 와야 이긴다
    assert css.index(':root[data-theme="light"] {') < alias.start()
    assert re.search(r"body\s*\{[^}]*font-family:\s*var\(--vc-font-sans", css)


def test_correlation_cells_use_diluted_base_in_dark():
    """다크의 파스텔 --vc-up/--vc-down을 60% 혼합하면 --text 대비가 3.3:1로 떨어진다 — 다크 기준색은 희석."""
    css = _read("static", "style.css")
    dark = re.search(r':root,\s*:root\[data-theme="dark"\]\s*\{([^}]*--corr-up[^}]*)\}', css)
    assert dark, "dark --corr-up/--corr-down block missing"
    assert "--corr-up: color-mix(in srgb, var(--up) 67%, var(--surface))" in dark.group(1)
    assert "--corr-down: color-mix(in srgb, var(--down) 67%, var(--surface))" in dark.group(1)
    light = re.search(r':root\[data-theme="light"\]\s*\{([^}]*--corr-up[^}]*)\}', css)
    assert light and "--corr-up: var(--up)" in light.group(1)
    charts = _read("static", "js", "charts.js")
    assert "var(--corr-up, var(--up))" in charts and "var(--corr-down, var(--down))" in charts


def test_service_worker_precaches_versioned_assets(html):
    sw = _read("sw.js")
    for asset in ("static/vc-tokens.css", "static/vc-shell.js", "static/style.css"):
        assert f"'{asset}{VERSION_TAG}'" in sw
        assert f"{asset}{VERSION_TAG}" in html
    assert "'static/js/ecosystem.js'" in sw
    assert int(re.search(r"const CACHE = 'goldgap-v(\d+)'", sw).group(1)) >= 5


def test_deploy_ships_static_dir_and_hub_summary():
    deploy = _read(".github", "workflows", "deploy.yml")
    assert "cp -r static _site/static" in deploy  # vc-shell.js·vc-tokens.css 포함
    assert "cp templates/index.html _site/index.html" in deploy
    assert 'git show "FETCH_HEAD:$f"' in deploy and 'cp "$f" "_site/$f"' in deploy
    assert "goldgap/vc_publish.py validate summary.json" in deploy


def test_update_workflow_gates_on_content_change():
    wf = _read(".github", "workflows", "update-data.yml")
    assert "python -m goldgap.hub_summary --data data.json --prev /tmp/data_prev.json" in wf
    assert "steps.publish.outputs.data_changed == 'true'" in wf  # og 생략
    assert "cmp -s data.json" not in wf  # 타임스탬프 때문에 항상 달랐던 옛 비교
    assert "git diff --cached --quiet" in wf


def test_npm_test_script_works_on_node_20_and_22_plus():
    pkg = json.loads(_read("package.json"))
    script = pkg["scripts"]["test"]
    # 디렉터리 인자(tests/js)는 Node 22+에서 실패, 따옴표 친 glob은 Node 20이 해석하지 못한다
    assert script == "node --test tests/js/*.test.mjs"
    ci = _read(".github", "workflows", "ci.yml")
    assert "node --test tests/js/*.test.mjs" in ci
