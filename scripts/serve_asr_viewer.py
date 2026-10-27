#!/usr/bin/env python3
"""
scripts/serve_asr_viewer.py

Web App siêu gọn (Zero-dependency, Python stdlib) để nghe file .wav và xem
kết quả phiên âm PhoASR từ `data/cache/asr_cache.parquet` song song với Gold Text BTC.

Tính năng:
- Audio player hỗ trợ tua (HTTP 206 Range), chọn tốc độ (0.75x, 1x, 1.25x, 1.5x)
- So sánh sai khác từng từ (Word Diff Highlighter trực quan)
- Bộ lọc nâng cao: Hoàn hảo (WER=0%), Sai nhiều (WER>30%), Train, Public Test
- Tìm kiếm tức thì theo ID hoặc từ khóa
- Tự động mở trình duyệt tại: http://localhost:8080
"""

from __future__ import annotations

import argparse
import contextlib
import json
import mimetypes
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")

import polars as pl


def compute_sample_wer(ref: str, hyp: str) -> float:
    """Tính WER đơn giản giữa ref và hyp (bằng Levenshtein từ ngữ)."""
    r_words = ref.strip().lower().split()
    h_words = hyp.strip().lower().split()
    if not r_words:
        return 0.0 if not h_words else 1.0
    if not h_words:
        return 1.0

    d = [[0] * (len(h_words) + 1) for _ in range(len(r_words) + 1)]
    for i in range(len(r_words) + 1):
        d[i][0] = i
    for j in range(len(h_words) + 1):
        d[0][j] = j

    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            if r_words[i - 1] == h_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )
    return float(d[len(r_words)][len(h_words)] / len(r_words))


class ASRDataStore:
    def __init__(self, parquet_path: Path, data_dir: Path):
        self.parquet_path = parquet_path
        self.data_dir = data_dir
        self.records: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        print(f"Đang đọc cache ASR từ: {self.parquet_path}...")
        df = pl.read_parquet(self.parquet_path)

        # Đọc Gold Audio Text từ train.jsonl
        train_texts: dict[str, str] = {}
        candidate_train_files = [
            self.data_dir / "train/train.jsonl",
            self.data_dir / "train.jsonl",
            Path("train/train.jsonl"),
            Path("data/train/train.jsonl"),
            Path("train+public_test/train/train.jsonl"),
        ]
        train_jsonl = next((p for p in candidate_train_files if p.exists()), None)
        if train_jsonl:
            print(f"Tìm thấy Gold Text tại: {train_jsonl}")
            with open(train_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        audio = item.get("audio")
                        text = item.get("audio_text")
                        if audio and text:
                            train_texts[audio] = text

        for row in df.iter_rows(named=True):
            audio_p = row["audio_path"]
            trans = row["transcription"] or ""
            dur = row.get("duration_sec", 0.0) or 0.0
            gold = train_texts.get(audio_p)

            # Tính WER nếu có nhãn Gold
            wer_val = compute_sample_wer(gold, trans) if gold else None

            # Phân loại dataset
            is_train = "train" in audio_p.lower()
            is_history = "_u" in audio_p.lower()

            self.records.append({
                "audio_path": audio_p,
                "transcription": trans,
                "gold_text": gold,
                "duration_sec": round(float(dur), 2),
                "wer": round(wer_val, 2) if wer_val is not None else None,
                "dataset": "train" if is_train else "public_test",
                "is_history": is_history,
                "id": Path(audio_p).stem,
            })

        print(f"Đã nạp xong {len(self.records)} mẫu ASR!")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PhoASR Voice & Text Inspector - Zalo AI Challenge 2026</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    audio::-webkit-media-controls-panel { background-color: #f8fafc; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased min-h-screen">
  <!-- Header -->
  <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white flex items-center justify-center font-bold text-xl shadow-sm">
          🎙️
        </div>
        <div>
          <h1 class="font-bold text-lg text-slate-900 leading-tight">PhoASR Voice-to-Text Studio</h1>
          <p class="text-xs text-slate-500">Zalo AI Challenge 2026 • Task 1: LifeMate • 6,969 Samples Audio</p>
        </div>
      </div>
      <div class="flex items-center gap-2 text-xs font-semibold">
        <span class="px-3 py-1 bg-slate-100 text-slate-700 rounded-full border border-slate-200" id="stat-total">Tổng: 6,969</span>
        <span class="px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200" id="stat-gold">Có Gold: 1,545</span>
        <span class="px-3 py-1 bg-blue-50 text-blue-700 rounded-full border border-blue-200" id="stat-perfect">Khớp 100%: ...</span>
      </div>
    </div>
  </header>

  <!-- Control & Filter Bar -->
  <div class="max-w-7xl mx-auto px-4 py-4">
    <div class="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs space-y-3">
      <div class="flex flex-col md:flex-row gap-3 items-center justify-between">
        <!-- Search Input -->
        <div class="relative w-full md:w-96">
          <span class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">🔍</span>
          <input 
            type="text" 
            id="search-input" 
            placeholder="Tìm theo ID hoặc từ khóa (vd: Zalo, Sếp Dũng, Đà Nẵng, 50 nghìn)..."
            class="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
          >
        </div>

        <!-- Word Diff Toggle -->
        <div class="flex items-center gap-4 w-full md:w-auto justify-between md:justify-end">
          <label class="flex items-center gap-2 cursor-pointer select-none text-xs font-medium text-slate-700 bg-slate-50 px-3 py-1.5 rounded-xl border border-slate-200">
            <input type="checkbox" id="diff-toggle" checked onchange="toggleDiff()" class="rounded text-blue-600 focus:ring-blue-500">
            <span>✨ So sánh lệch từng từ (Word Diff)</span>
          </label>
        </div>
      </div>

      <!-- Quick Filter Tabs -->
      <div class="flex flex-wrap items-center gap-1.5 pt-1 border-t border-slate-100 text-xs font-medium">
        <span class="text-slate-400 text-[11px] mr-1">Bộ lọc:</span>
        <button onclick="setFilter('all')" id="tab-all" class="px-3 py-1 rounded-lg bg-slate-900 text-white shadow-xs">Tất cả (6,969)</button>
        <button onclick="setFilter('train')" id="tab-train" class="px-3 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200">Train (1,545)</button>
        <button onclick="setFilter('perfect')" id="tab-perfect" class="px-3 py-1 rounded-lg bg-slate-100 text-emerald-700 hover:bg-emerald-100">🎯 Khớp 100% (WER=0%)</button>
        <button onclick="setFilter('bad')" id="tab-bad" class="px-3 py-1 rounded-lg bg-slate-100 text-rose-700 hover:bg-rose-100">⚠️ Sai nhiều (WER > 30%)</button>
        <button onclick="setFilter('public_test')" id="tab-public_test" class="px-3 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200">Public Test (5,424)</button>
      </div>
    </div>
  </div>

  <!-- Legend -->
  <div class="max-w-7xl mx-auto px-4 mb-3 flex items-center justify-between text-xs text-slate-500">
    <div id="results-count" class="font-medium">Đang tải dữ liệu...</div>
    <div class="flex items-center gap-3 text-[11px] bg-white px-3 py-1 rounded-lg border border-slate-200">
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Khớp</span>
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-rose-400"></span> Từ bị bỏ sót (Gold)</span>
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> Từ nghe nhầm (ASR)</span>
    </div>
  </div>

  <!-- Main Content Cards -->
  <main class="max-w-7xl mx-auto px-4 pb-16">
    <div id="cards-container" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <!-- Cards rendered via JS -->
    </div>

    <!-- Pagination / Load more -->
    <div class="mt-8 flex justify-center">
      <button 
        id="btn-load-more" 
        onclick="loadMore()" 
        class="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs rounded-xl shadow-xs transition disabled:opacity-50"
      >
        Tải thêm 40 mẫu tiếp theo...
      </button>
    </div>
  </main>

  <script>
    let allRecords = [];
    let currentFilter = 'all';
    let currentSearch = '';
    let currentOffset = 0;
    let showWordDiff = true;
    const PAGE_SIZE = 40;

    function escapeHtml(text) {
      if (!text) return '';
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function computeWordDiff(gold, asr) {
      if (!gold) return { goldHtml: '', asrHtml: escapeHtml(asr) };
      const gWords = gold.trim().split(/\\s+/);
      const aWords = asr.trim().split(/\\s+/);

      const m = gWords.length, n = aWords.length;
      const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
      for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
          if (gWords[i - 1].toLowerCase().replace(/[^\\w]/g, '') === aWords[j - 1].toLowerCase().replace(/[^\\w]/g, '')) {
            dp[i][j] = dp[i - 1][j - 1] + 1;
          } else {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
          }
        }
      }

      let i = m, j = n;
      const gMatched = new Array(m).fill(false);
      const aMatched = new Array(n).fill(false);
      while (i > 0 && j > 0) {
        if (gWords[i - 1].toLowerCase().replace(/[^\\w]/g, '') === aWords[j - 1].toLowerCase().replace(/[^\\w]/g, '')) {
          gMatched[i - 1] = true;
          aMatched[j - 1] = true;
          i--; j--;
        } else if (dp[i - 1][j] >= dp[i][j - 1]) {
          i--;
        } else {
          j--;
        }
      }

      const goldHtml = gWords.map((w, idx) => 
        gMatched[idx] 
          ? `<span class="text-slate-800">${escapeHtml(w)}</span>` 
          : `<span class="bg-rose-100 text-rose-800 font-semibold px-1 rounded line-through">${escapeHtml(w)}</span>`
      ).join(' ');

      const asrHtml = aWords.map((w, idx) => 
        aMatched[idx] 
          ? `<span class="text-slate-900 font-medium">${escapeHtml(w)}</span>` 
          : `<span class="bg-amber-100 text-amber-900 font-bold px-1 rounded">${escapeHtml(w)}</span>`
      ).join(' ');

      return { goldHtml, asrHtml };
    }

    async function init() {
      const res = await fetch('/api/samples?limit=7000');
      allRecords = await res.json();
      const goldCount = allRecords.filter(r => r.gold_text).length;
      const perfectCount = allRecords.filter(r => r.wer === 0).length;

      document.getElementById('stat-total').innerText = `Tổng: ${allRecords.length}`;
      document.getElementById('stat-gold').innerText = `Có Gold: ${goldCount}`;
      document.getElementById('stat-perfect').innerText = `🎯 Khớp 100%: ${perfectCount}`;
      render();
    }

    function toggleDiff() {
      showWordDiff = document.getElementById('diff-toggle').checked;
      render(true);
    }

    function setFilter(f) {
      currentFilter = f;
      currentOffset = 0;
      document.querySelectorAll('[id^="tab-"]').forEach(el => {
        el.className = 'px-3 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200';
      });
      document.getElementById(`tab-${f}`).className = 'px-3 py-1 rounded-lg bg-slate-900 text-white shadow-xs';
      render();
    }

    document.getElementById('search-input').addEventListener('input', (e) => {
      currentSearch = e.target.value.trim().toLowerCase();
      currentOffset = 0;
      render();
    });

    function getFilteredRecords() {
      return allRecords.filter(r => {
        if (currentFilter === 'train' && r.dataset !== 'train') return false;
        if (currentFilter === 'public_test' && r.dataset !== 'public_test') return false;
        if (currentFilter === 'perfect' && r.wer !== 0) return false;
        if (currentFilter === 'bad' && (r.wer === null || r.wer <= 0.3)) return false;

        if (currentSearch) {
          const matchId = r.id.toLowerCase().includes(currentSearch);
          const matchTrans = r.transcription.toLowerCase().includes(currentSearch);
          const matchGold = r.gold_text && r.gold_text.toLowerCase().includes(currentSearch);
          if (!matchId && !matchTrans && !matchGold) return false;
        }
        return true;
      });
    }

    function changePlaybackRate(audioId, rate) {
      const audio = document.getElementById(audioId);
      if (audio) audio.playbackRate = parseFloat(rate);
    }

    function render(forceRedraw = false) {
      const filtered = getFilteredRecords();
      document.getElementById('results-count').innerText = `Hiển thị ${Math.min(currentOffset + PAGE_SIZE, filtered.length)} / ${filtered.length} mẫu khớp điều kiện`;
      
      const visible = filtered.slice(0, currentOffset + PAGE_SIZE);
      const container = document.getElementById('cards-container');
      
      if (currentOffset === 0 || forceRedraw) container.innerHTML = '';

      const sliceStart = forceRedraw ? 0 : currentOffset;
      const html = visible.slice(sliceStart).map((r, i) => {
        const audioDomId = `audio-${r.id}-${i}`;
        let werBadge = '';
        if (r.wer !== null) {
          let color = r.wer === 0 ? 'bg-emerald-100 text-emerald-800 border-emerald-300' :
                      r.wer <= 0.25 ? 'bg-blue-100 text-blue-800 border-blue-200' :
                      r.wer <= 0.5 ? 'bg-amber-100 text-amber-800 border-amber-200' :
                      'bg-rose-100 text-rose-800 border-rose-300';
          werBadge = `<span class="px-2 py-0.5 text-[11px] font-bold rounded-md border ${color}">WER: ${(r.wer * 100).toFixed(0)}%</span>`;
        }

        const tagDataset = r.dataset === 'train' 
          ? `<span class="px-2 py-0.5 text-[11px] font-medium rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">Train</span>` 
          : `<span class="px-2 py-0.5 text-[11px] font-medium rounded-md bg-slate-100 text-slate-700 border border-slate-200">Public Test</span>`;
        
        const tagTurn = r.is_history 
          ? `<span class="px-2 py-0.5 text-[11px] font-medium rounded-md bg-purple-50 text-purple-700 border border-purple-200">History Turn</span>`
          : `<span class="px-2 py-0.5 text-[11px] font-medium rounded-md bg-blue-50 text-blue-700 border border-blue-200">Main Utterance</span>`;

        let goldDisplay = escapeHtml(r.gold_text);
        let asrDisplay = escapeHtml(r.transcription);

        if (showWordDiff && r.gold_text) {
          const diff = computeWordDiff(r.gold_text, r.transcription);
          goldDisplay = diff.goldHtml;
          asrDisplay = diff.asrHtml;
        }

        return `
          <div class="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs flex flex-col justify-between hover:border-blue-400 hover:shadow-sm transition">
            <div>
              <!-- Card Header -->
              <div class="flex items-center justify-between mb-2.5 gap-2 flex-wrap">
                <div class="flex items-center gap-1.5">
                  <span class="font-mono font-bold text-xs text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">${r.id}</span>
                  ${tagDataset}
                  ${tagTurn}
                </div>
                <div class="flex items-center gap-2">
                  ${werBadge}
                  <span class="text-xs text-slate-400 font-medium">⏱️ ${r.duration_sec}s</span>
                </div>
              </div>

              <!-- Audio Player with Speed Control -->
              <div class="mb-3 bg-slate-50 p-2 rounded-xl border border-slate-100 flex items-center gap-2">
                <audio id="${audioDomId}" controls preload="none" class="w-full h-8 flex-1" src="/audio/${encodeURIComponent(r.audio_path)}"></audio>
                <select onchange="changePlaybackRate('${audioDomId}', this.value)" class="text-[11px] bg-white border border-slate-200 rounded-lg px-1.5 py-1 text-slate-600 focus:outline-none cursor-pointer">
                  <option value="1">1.0x</option>
                  <option value="1.25">1.25x</option>
                  <option value="1.5">1.5x</option>
                  <option value="0.75">0.75x</option>
                </select>
              </div>

              <!-- Text Comparison Section -->
              <div class="space-y-2 text-xs">
                ${r.gold_text ? `
                <div class="bg-emerald-50/70 border border-emerald-100 rounded-xl p-2.5">
                  <span class="font-bold text-emerald-800 uppercase text-[10px] tracking-wider block mb-0.5">🎯 Gold Text (BTC):</span>
                  <p class="text-slate-800 font-medium leading-relaxed">${goldDisplay}</p>
                </div>
                ` : ''}

                <div class="bg-blue-50/60 border border-blue-100 rounded-xl p-2.5">
                  <span class="font-bold text-blue-800 uppercase text-[10px] tracking-wider block mb-0.5">⚡ PhoASR-small Output:</span>
                  <p class="text-slate-900 leading-relaxed font-semibold">${asrDisplay || '<span class="text-slate-400 italic">Không có tiếng nói</span>'}</p>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');

      container.insertAdjacentHTML('beforeend', html);

      const btnMore = document.getElementById('btn-load-more');
      if (currentOffset + PAGE_SIZE >= filtered.length) {
        btnMore.disabled = true;
        btnMore.innerText = 'Đã hiển thị tất cả các mẫu';
      } else {
        btnMore.disabled = false;
        btnMore.innerText = 'Tải thêm 40 mẫu...';
      }
    }

    function loadMore() {
      currentOffset += PAGE_SIZE;
      render();
    }

    init();
  </script>
</body>
</html>
"""


def find_audio_file(rel_path: str, search_roots: list[Path]) -> Path | None:
    p = Path(rel_path)
    if p.exists():
        return p
    for root in search_roots:
        cand1 = root / rel_path
        if cand1.exists():
            return cand1
        if rel_path.startswith("audio/"):
            cand2 = root / rel_path[6:]
            if cand2.exists():
                return cand2
            cand3 = root / "audio" / rel_path[6:]
            if cand3.exists():
                return cand3
    return None


class ASRViewerHandler(BaseHTTPRequestHandler):
    store: ASRDataStore
    search_roots: list[Path]

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/samples":
            query = urllib.parse.parse_qs(parsed.query)
            limit = int(query.get("limit", [100])[0])
            offset = int(query.get("offset", [0])[0])

            data_slice = self.store.records[offset: offset + limit]
            json_bytes = json.dumps(data_slice, ensure_ascii=False).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(json_bytes)))
            self.end_headers()
            self.wfile.write(json_bytes)
            return

        if path.startswith("/audio/"):
            audio_rel = urllib.parse.unquote(path[7:])
            real_path = find_audio_file(audio_rel, self.search_roots)
            if not real_path or not real_path.exists():
                self.send_error(404, f"Audio not found: {audio_rel}")
                return

            mime_type = mimetypes.guess_type(real_path)[0] or "audio/wav"
            file_size = real_path.stat().st_size

            # Hỗ trợ HTTP 206 Partial Content cho việc tua audio mượt mà
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                bytes_range = range_header[6:].strip()
                start_str, _, end_str = bytes_range.partition("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(real_path, "rb") as f:
                    f.seek(start)
                    bytes_left = length
                    while bytes_left > 0:
                        chunk = f.read(min(65536, bytes_left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        bytes_left -= len(chunk)
                return

            # Chuẩn 200 OK
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(real_path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
            return

        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        if "404" in str(args):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Web Server xem và nghe kết quả ASR")
    parser.add_argument("--port", type=int, default=8080, help="Cổng web (mặc định: 8080)")
    parser.add_argument("--parquet", type=str, default="data/cache/asr_cache.parquet", help="Đường dẫn file parquet")
    parser.add_argument("--data-dir", type=str, default=".", help="Thư mục chứa dữ liệu")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        fallback = Path("asr_cache.parquet")
        if fallback.exists():
            parquet_path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file parquet tại {args.parquet}")

    data_dir = Path(args.data_dir)
    store = ASRDataStore(parquet_path=parquet_path, data_dir=data_dir)

    search_roots = [
        data_dir,
        data_dir / "train",
        data_dir / "public_test",
        data_dir / "train+public_test",
        data_dir / "train+public_test/train",
        data_dir / "train+public_test/public_test",
        Path("train"),
        Path("public_test"),
        Path("data/train"),
        Path("data/public_test"),
    ]

    handler_class = type(
        "CustomASRViewerHandler",
        (ASRViewerHandler,),
        {"store": store, "search_roots": search_roots},
    )

    port = args.port
    server = None
    for p in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler_class)
            port = p
            break
        except OSError:
            continue

    if not server:
        print(f"Lỗi: Không thể mở port từ {port} đến {port+10}")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print("\n" + "=" * 60)
    print("🚀 PHOASR WEB VIEWER ĐÃ SẴN SÀNG!")
    print(f"🔗 Mở trình duyệt tại: {url}")
    print("Bấm Ctrl + C để dừng web server.")
    print("=" * 60 + "\n")

    # Tự động mở trình duyệt
    with contextlib.suppress(Exception):
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng Web Server.")
        server.server_close()


if __name__ == "__main__":
    main()
