"""src/eval/visual_debugger.py.

Interactive Visual Debug Dashboard for Zalo AI Challenge 2026 (Task 1: LifeMate).
Xuất file HTML tự chứa (standalone, zero-external-dependencies) giúp quan sát chi tiết:
- Transcript & User Profile
- Phản hồi thô của LFM3-1.7B
- Lịch sử ReAct từng bước & kết quả Observation
- So sánh đối chứng Tool Calls (Gold vs Model)
- So sánh đối chứng State Hash (SHA-256)
- Nhật ký Validate & Self-Correction Retries
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def generate_dashboard_html(records: list[dict[str, Any]], title: str = "LifeMate LFM3-1.7B Debug Dashboard") -> str:
    """Tạo chuỗi HTML hoàn chỉnh chứa giao diện dashboard chẩn đoán."""
    total = len(records)
    hash_matches = sum(1 for r in records if r.get("hash_match", False))
    call_matches = sum(1 for r in records if r.get("call_match", False))
    avg_latency = (sum(r.get("execution_time_ms", 0.0) for r in records) / total) if total > 0 else 0.0
    total_val_retries = sum(r.get("validation_retries", 0) for r in records)

    hash_pct = (hash_matches / total * 100.0) if total > 0 else 0.0
    call_pct = (call_matches / total * 100.0) if total > 0 else 0.0

    cards_html = []
    for r in records:
        idx = r.get("idx", 0)
        sid = html.escape(str(r.get("sample_id", "")))
        transcript = html.escape(str(r.get("transcript", "")))
        raw_output = html.escape(str(r.get("raw_model_response", "") or "(Rỗng / Không có response)"))
        model_calls = r.get("model_tool_calls", [])
        gold_calls = r.get("gold_tool_calls", [])
        model_hash = str(r.get("model_state_hash", ""))
        gold_hash = str(r.get("gold_state_hash", ""))
        hash_match = r.get("hash_match", False)
        call_match = r.get("call_match", False)
        latency = r.get("execution_time_ms", 0.0)
        steps = r.get("steps_taken", 0)
        retries = r.get("validation_retries", 0)
        val_logs = r.get("validation_logs", [])
        profile = r.get("user_profile", {})
        apps = r.get("installed_apps", [])

        # Phân loại trạng thái
        if hash_match:
            status_class = "status-pass"
            status_text = "HASH KHỚP [✓]"
        elif call_match:
            status_class = "status-tool-match"
            status_text = "TOOL KHỚP [~]"
        else:
            status_class = "status-fail"
            status_text = "CHƯA KHỚP [X]"

        # Format Tool Calls
        model_calls_json = html.escape(json.dumps(model_calls, ensure_ascii=False, indent=2))
        gold_calls_json = html.escape(json.dumps(gold_calls, ensure_ascii=False, indent=2))
        profile_json = html.escape(json.dumps(profile, ensure_ascii=False, indent=2))
        apps_str = html.escape(", ".join(apps))

        val_logs_html = ""
        if val_logs:
            logs_content = "<br>".join(f"• {html.escape(l)}" for l in val_logs)
            val_logs_html = f"""
            <div class="log-section">
                <div class="section-title">⚠️ Nhật ký sửa lỗi Validate ({len(val_logs)} lần):</div>
                <div class="val-logs">{logs_content}</div>
            </div>
            """

        card = f"""
        <div class="sample-card {status_class}" data-status="{'pass' if hash_match else ('tool' if call_match else 'fail')}">
            <div class="card-header" onclick="toggleCard(this)">
                <div class="header-left">
                    <span class="sample-badge">#{idx}</span>
                    <span class="sample-id">{sid}</span>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                <div class="header-right">
                    <span class="meta-pill">⏱ {latency:.0f}ms</span>
                    <span class="meta-pill">🔄 {steps} ReAct steps</span>
                    <span class="meta-pill">⚡ {retries} retries</span>
                    <span class="toggle-icon">▼</span>
                </div>
            </div>
            <div class="card-body">
                <div class="meta-grid">
                    <div class="meta-box">
                        <div class="box-label">🗣️ ASR Transcript:</div>
                        <div class="box-value transcript-value">"{transcript}"</div>
                    </div>
                    <div class="meta-box">
                        <div class="box-label">📱 Installed Apps ({len(apps)}):</div>
                        <div class="box-value">{apps_str}</div>
                    </div>
                </div>

                <div class="comparison-grid">
                    <div class="col">
                        <div class="col-header">🎯 Target Gold Calls ({len(gold_calls)}):</div>
                        <pre class="code-block gold-code">{gold_calls_json}</pre>
                    </div>
                    <div class="col">
                        <div class="col-header">🤖 Extracted Model Calls ({len(model_calls)}):</div>
                        <pre class="code-block model-code">{model_calls_json}</pre>
                    </div>
                </div>

                <div class="hash-grid">
                    <div class="hash-box {'hash-match' if hash_match else 'hash-diff'}">
                        <span class="hash-label">Model State Hash:</span>
                        <code class="hash-value">{html.escape(model_hash)}</code>
                    </div>
                    <div class="hash-box {'hash-match' if hash_match else 'hash-diff'}">
                        <span class="hash-label">Gold State Hash:</span>
                        <code class="hash-value">{html.escape(gold_hash or '(Không có nhãn gold)')}</code>
                    </div>
                </div>

                {val_logs_html}

                <details class="raw-details">
                    <summary>🔍 Xem phản hồi thô của LFM3-1.7B (Raw Model Generation)</summary>
                    <pre class="code-block raw-code">{raw_output}</pre>
                </details>

                <details class="raw-details">
                    <summary>👤 Xem Hồ Sơ Người Dùng (User Profile Context)</summary>
                    <pre class="code-block raw-code">{profile_json}</pre>
                </details>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_str = "\n".join(cards_html)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --pass: #10b981;
    --pass-bg: rgba(16, 185, 129, 0.15);
    --tool-match: #3b82f6;
    --tool-bg: rgba(59, 130, 246, 0.15);
    --fail: #ef4444;
    --fail-bg: rgba(239, 68, 68, 0.15);
    --accent: #8b5cf6;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 24px;
    line-height: 1.5;
}}
.container {{ max-width: 1300px; margin: 0 auto; }}
.header {{
    margin-bottom: 24px;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}}
.title-group h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
.title-group p {{ color: var(--text-muted); font-size: 14px; }}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}
.kpi-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}
.kpi-val {{ font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 4px; }}
.kpi-card.pass .kpi-val {{ color: var(--pass); }}
.kpi-card.tool .kpi-val {{ color: var(--tool-match); }}
.kpi-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}

.filter-bar {{
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}}
.filter-btn {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}}
.filter-btn.active, .filter-btn:hover {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
}}

.sample-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    margin-bottom: 16px;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.sample-card.status-pass {{ border-left: 5px solid var(--pass); }}
.sample-card.status-tool-match {{ border-left: 5px solid var(--tool-match); }}
.sample-card.status-fail {{ border-left: 5px solid var(--fail); }}

.card-header {{
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.02);
    user-select: none;
}}
.card-header:hover {{ background: rgba(255, 255, 255, 0.05); }}
.header-left, .header-right {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.sample-badge {{ font-weight: 700; color: var(--text-muted); font-size: 14px; }}
.sample-id {{ font-family: var(--font-mono); font-weight: 600; font-size: 15px; color: #fff; }}
.status-badge {{
    font-size: 12px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
}}
.status-badge.status-pass {{ background: var(--pass-bg); color: var(--pass); }}
.status-badge.status-tool-match {{ background: var(--tool-bg); color: var(--tool-match); }}
.status-badge.status-fail {{ background: var(--fail-bg); color: var(--fail); }}

.meta-pill {{
    background: #0f172a;
    border: 1px solid #334155;
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 12px;
    color: var(--text-muted);
}}
.toggle-icon {{ color: var(--text-muted); font-size: 12px; transition: transform 0.2s; }}
.sample-card.open .toggle-icon {{ transform: rotate(180deg); }}

.card-body {{
    display: none;
    padding: 18px;
    border-top: 1px solid var(--card-border);
}}
.sample-card.open .card-body {{ display: block; }}

.meta-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
    margin-bottom: 16px;
}}
.meta-box {{
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px 14px;
}}
.box-label {{ font-size: 12px; font-weight: 600; color: #38bdf8; margin-bottom: 4px; }}
.box-value {{ font-size: 14px; color: #e2e8f0; }}
.transcript-value {{ font-size: 15px; font-weight: 500; color: #fef08a; }}

.comparison-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}}
@media (max-width: 800px) {{
    .comparison-grid {{ grid-template-columns: 1fr; }}
}}
.col-header {{ font-size: 13px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px; }}
.code-block {{
    background: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 13px;
    color: #cbd5e1;
    overflow-x: auto;
    max-height: 320px;
    line-height: 1.4;
}}
.gold-code {{ border-left: 3px solid #38bdf8; }}
.model-code {{ border-left: 3px solid var(--accent); }}

.hash-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
}}
@media (max-width: 800px) {{
    .hash-grid {{ grid-template-columns: 1fr; }}
}}
.hash-box {{
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid;
    font-size: 12px;
}}
.hash-box.hash-match {{ background: var(--pass-bg); border-color: var(--pass); color: #a7f3d0; }}
.hash-box.hash-diff {{ background: var(--fail-bg); border-color: var(--fail); color: #fecaca; }}
.hash-label {{ font-weight: 700; display: block; margin-bottom: 2px; }}
.hash-value {{ font-family: var(--font-mono); word-break: break-all; }}

.log-section {{
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
}}
.val-logs {{ font-family: var(--font-mono); font-size: 13px; color: #fde68a; margin-top: 4px; }}

.raw-details {{
    margin-top: 10px;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.01);
}}
.raw-details summary {{
    cursor: pointer;
    font-size: 13px;
    color: var(--text-muted);
    font-weight: 600;
}}
.raw-details[open] summary {{ margin-bottom: 8px; color: #fff; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="title-group">
            <h1>📊 LifeMate Diagnostic Dashboard (LFM3-1.7B)</h1>
            <p>Zalo AI Challenge 2026 — Task 1: Spoken Voice Agent to Environment Actions</p>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Tổng Mẫu Kiểm Thử</div>
            <div class="kpi-val">{total}</div>
        </div>
        <div class="kpi-card pass">
            <div class="kpi-label">Khớp State Hash</div>
            <div class="kpi-val">{hash_matches}/{total} ({hash_pct:.1f}%)</div>
        </div>
        <div class="kpi-card tool">
            <div class="kpi-label">Khớp Chuỗi Tool</div>
            <div class="kpi-val">{call_matches}/{total} ({call_pct:.1f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Latency Trung Bình</div>
            <div class="kpi-val">{avg_latency:.0f} ms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Validate Retries</div>
            <div class="kpi-val">{total_val_retries}</div>
        </div>
    </div>

    <div class="filter-bar">
        <button class="filter-btn active" onclick="filterCards('all', this)">Tất cả ({total})</button>
        <button class="filter-btn" onclick="filterCards('pass', this)">State Hash Khớp ({hash_matches})</button>
        <button class="filter-btn" onclick="filterCards('tool', this)">Tool Khớp ({call_matches})</button>
        <button class="filter-btn" onclick="filterCards('fail', this)">Chưa khớp ({total - hash_matches})</button>
    </div>

    <div class="samples-list">
        {cards_str}
    </div>
</div>

<script>
function toggleCard(header) {{
    const card = header.parentElement;
    card.classList.toggle('open');
}}

function filterCards(type, btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const cards = document.querySelectorAll('.sample-card');
    cards.forEach(card => {{
        const status = card.getAttribute('data-status');
        if (type === 'all') {{
            card.style.display = 'block';
        }} else if (type === 'pass') {{
            card.style.display = (status === 'pass') ? 'block' : 'none';
        }} else if (type === 'tool') {{
            card.style.display = (status === 'tool' || status === 'pass') ? 'block' : 'none';
        }} else if (type === 'fail') {{
            card.style.display = (status === 'fail' || status === 'tool') ? 'block' : 'none';
        }}
    }});
}}

// Mở sẵn card đầu tiên
document.addEventListener('DOMContentLoaded', () => {{
    const firstCard = document.querySelector('.sample-card');
    if (firstCard) firstCard.classList.add('open');
}});
</script>
</body>
</html>
"""


def save_dashboard(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Ghi dashboard ra file HTML."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_dashboard_html(records)
    out_p.write_text(html_content, encoding="utf-8")
    return out_p
