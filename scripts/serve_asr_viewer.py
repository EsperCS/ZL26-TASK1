#!/usr/bin/env python3
"""
scripts/serve_asr_viewer.py

Web Studio kiểm tra trực quan đa mô hình ASR (ZipFormer-150M vs Gipformer-68M vs PhoASR):
- Nghe file audio .wav với HTTP 206 Range (tua mượt mà, chọn tốc độ 0.75x, 1x, 1.25x, 1.5x)
- Xem kết quả phiên âm ZipFormer-150M song song với Gold Text của BTC
- So sánh song song (Side-by-side) ZipFormer vs Gipformer
- Word Diff Highlighter: tô màu trực quan từng từ đúng (xanh lá), từ nghe sai (cam), từ bỏ sót (đỏ)
- Bộ lọc thông minh: Khớp 100% (WER=0%), Sai ít (WER<=15%), Sai nhiều (WER>25%), ZipFormer thắng Gipformer
- Tự động mở tại: http://localhost:8080
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
        self.stats: dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        print(f"Đang đọc dữ liệu ASR từ: {self.parquet_path}...")
        df = pl.read_parquet(self.parquet_path)
        cols = set(df.columns)

        # Định dạng 1: File benchmark 500 mẫu (ZipFormer vs Gipformer)
        if "zip_final" in cols and "gip_final" in cols:
            print("[CHẾ ĐỘ] Đang tải bảng so sánh ZipFormer-150M vs Gipformer-68M...")
            zip_wers: list[float] = []
            gip_wers: list[float] = []
            exact_zip = 0

            for row in df.iter_rows(named=True):
                sample_id = str(row.get("id") or Path(row["audio_path"]).stem)
                audio_p = row["audio_path"]
                gold = row.get("gold_text") or ""
                dur = float(row.get("audio_duration_sec") or 0.0)

                # ZipFormer
                zip_final = str(row.get("zip_final") or "")
                zip_acoustic = str(row.get("zip_raw_acoustic") or "")
                zip_wer = float(row.get("zip_wer_sample") or 0.0)
                zip_lat = float(row.get("zip_latency_ms") or 0.0)
                zip_wers.append(zip_wer)
                if zip_wer == 0.0:
                    exact_zip += 1

                # Gipformer
                gip_final = str(row.get("gip_final") or "")
                gip_acoustic = str(row.get("gip_raw_acoustic") or "")
                gip_wer = float(row.get("gip_wer_sample") or 0.0)
                gip_lat = float(row.get("gip_latency_ms") or 0.0)
                gip_wers.append(gip_wer)

                self.records.append({
                    "id": sample_id,
                    "audio_path": audio_p,
                    "gold_text": gold,
                    "duration_sec": round(dur, 2),
                    # ZipFormer
                    "zip_final": zip_final,
                    "zip_acoustic": zip_acoustic,
                    "zip_wer": round(zip_wer, 1),
                    "zip_latency_ms": round(zip_lat, 1),
                    # Gipformer
                    "gip_final": gip_final,
                    "gip_acoustic": gip_acoustic,
                    "gip_wer": round(gip_wer, 1),
                    "gip_latency_ms": round(gip_lat, 1),
                    # Comparison
                    "wer_diff": round(float(row.get("wer_diff") or abs(zip_wer - gip_wer)), 1),
                    "zip_better": zip_wer < gip_wer,
                    "gip_better": gip_wer < zip_wer,
                    "is_multi_model": True,
                })

            n = len(self.records)
            self.stats = {
                "total": n,
                "avg_zip_wer": round(sum(zip_wers) / n, 2) if n else 0,
                "avg_gip_wer": round(sum(gip_wers) / n, 2) if n else 0,
                "exact_match_zip": exact_zip,
                "exact_match_zip_pct": round(exact_zip / n * 100, 1) if n else 0,
                "mode": "benchmark_500",
            }
        else:
            # Định dạng 2: File cache đơn lẻ (PhoASR hoặc đơn model)
            print("[CHẾ ĐỘ] Đang tải bảng ASR cache đơn lẻ...")
            train_texts: dict[str, str] = {}
            candidate_train_files = [
                self.data_dir / "train/train.jsonl",
                self.data_dir / "train.jsonl",
                Path("train/train.jsonl"),
                Path("data/train/train.jsonl"),
            ]
            train_jsonl = next((p for p in candidate_train_files if p.exists()), None)
            if train_jsonl:
                with open(train_jsonl, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            it = json.loads(line)
                            a = it.get("audio")
                            t = it.get("audio_text")
                            if a and t:
                                train_texts[a] = t

            for row in df.iter_rows(named=True):
                audio_p = row["audio_path"]
                trans = row.get("transcription") or row.get("text") or ""
                dur = float(row.get("duration_sec") or 0.0)
                gold = train_texts.get(audio_p)
                wer_val = compute_sample_wer(gold, trans) * 100.0 if gold else 0.0

                sample_id = Path(audio_p).stem
                self.records.append({
                    "id": sample_id,
                    "audio_path": audio_p,
                    "gold_text": gold or "",
                    "duration_sec": round(dur, 2),
                    "zip_final": trans,
                    "zip_acoustic": trans.lower(),
                    "zip_wer": round(wer_val, 1),
                    "zip_latency_ms": 0.0,
                    "gip_final": "",
                    "gip_acoustic": "",
                    "gip_wer": 0.0,
                    "gip_latency_ms": 0.0,
                    "wer_diff": 0.0,
                    "zip_better": False,
                    "gip_better": False,
                    "is_multi_model": False,
                })

            self.stats = {
                "total": len(self.records),
                "avg_zip_wer": 0,
                "avg_gip_wer": 0,
                "exact_match_zip": 0,
                "exact_match_zip_pct": 0,
                "mode": "single_cache",
            }

        print(f"Đã nạp xong {len(self.records)} mẫu ASR vào DataStore!")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ZipFormer & Gipformer ASR Studio - Zalo AI Challenge 2026</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    audio::-webkit-media-controls-panel { background-color: #f8fafc; }
    .word-match { color: #0f172a; }
    .word-missing { background-color: #fee2e2; color: #b91c1c; text-decoration: line-through; padding: 1px 4px; border-radius: 4px; font-weight: 500; }
    .word-inserted { background-color: #fef3c7; color: #b45309; padding: 1px 4px; border-radius: 4px; font-weight: 600; border-bottom: 2px solid #f59e0b; }
    .word-replaced { background-color: #fef08a; color: #854d0e; padding: 1px 4px; border-radius: 4px; font-weight: 600; border-bottom: 2px dashed #ca8a04; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased min-h-screen">
  <!-- Header -->
  <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white flex items-center justify-center font-bold text-xl shadow-sm">
          ⚡
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-bold text-lg text-slate-900 leading-tight">ZipFormer-150M Voice Studio</h1>
            <span class="px-2 py-0.5 text-[10px] font-bold uppercase rounded-md bg-emerald-100 text-emerald-800">Winner</span>
          </div>
          <p class="text-xs text-slate-500">Zalo AI Challenge 2026 • Task 1 (LifeMate) • 4-Tier Next-Gen ASR Pipeline</p>
        </div>
      </div>
      <!-- Stats Badges -->
      <div class="flex flex-wrap items-center gap-2 text-xs font-semibold" id="stats-banner">
        <span class="px-3 py-1 bg-slate-100 text-slate-700 rounded-full border border-slate-200" id="stat-total">500 Mẫu</span>
        <span class="px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200" id="stat-zip-wer">Zip WER: 16.4%</span>
        <span class="px-3 py-1 bg-blue-50 text-blue-700 rounded-full border border-blue-200" id="stat-gip-wer">Gip WER: 18.0%</span>
        <span class="px-3 py-1 bg-purple-50 text-purple-700 rounded-full border border-purple-200" id="stat-exact">Khớp 100%: 10.4%</span>
        <span class="px-3 py-1 bg-amber-50 text-amber-700 rounded-full border border-amber-200">RTF: 0.0606 (~424ms)</span>
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
            placeholder="Tìm ID (train_0001) hoặc từ khóa (Sếp Dũng, Lan, 50 nghìn)..."
            class="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:bg-white transition"
          >
        </div>

        <!-- View Mode & Diff Controls -->
        <div class="flex flex-wrap items-center gap-3 w-full md:w-auto justify-between md:justify-end">
          <div class="flex bg-slate-100 p-1 rounded-xl text-xs font-medium">
            <button onclick="setViewMode('zip')" id="view-zip" class="px-3 py-1 rounded-lg bg-white text-slate-900 shadow-xs">ZipFormer (Chuẩn)</button>
            <button onclick="setViewMode('both')" id="view-both" class="px-3 py-1 rounded-lg text-slate-600 hover:text-slate-900">Song song (Side-by-side)</button>
            <button onclick="setViewMode('gip')" id="view-gip" class="px-3 py-1 rounded-lg text-slate-600 hover:text-slate-900">Gipformer</button>
          </div>

          <label class="flex items-center gap-2 cursor-pointer select-none text-xs font-medium text-slate-700 bg-slate-50 px-3 py-1.5 rounded-xl border border-slate-200">
            <input type="checkbox" id="diff-toggle" checked onchange="toggleDiff()" class="rounded text-emerald-600 focus:ring-emerald-500">
            <span>✨ Word Diff</span>
          </label>
        </div>
      </div>

      <!-- Quick Filter Tabs -->
      <div class="flex flex-wrap items-center gap-1.5 pt-1 border-t border-slate-100 text-xs font-medium">
        <span class="text-slate-400 text-[11px] mr-1">Bộ lọc:</span>
        <button onclick="setFilter('all')" id="tab-all" class="px-3 py-1 rounded-lg bg-slate-900 text-white shadow-xs">Tất cả (500)</button>
        <button onclick="setFilter('perfect')" id="tab-perfect" class="px-3 py-1 rounded-lg bg-slate-100 text-emerald-700 hover:bg-emerald-100">🎯 Khớp 100% (WER=0%)</button>
        <button onclick="setFilter('good')" id="tab-good" class="px-3 py-1 rounded-lg bg-slate-100 text-blue-700 hover:bg-blue-100">⚡ Sai ít (WER ≤ 15%)</button>
        <button onclick="setFilter('bad')" id="tab-bad" class="px-3 py-1 rounded-lg bg-slate-100 text-rose-700 hover:bg-rose-100">⚠️ Sai nhiều (WER > 25%)</button>
        <button onclick="setFilter('zip_better')" id="tab-zip_better" class="px-3 py-1 rounded-lg bg-slate-100 text-teal-700 hover:bg-teal-100">🏆 Zip thắng Gip</button>
        <button onclick="setFilter('gip_better')" id="tab-gip_better" class="px-3 py-1 rounded-lg bg-slate-100 text-indigo-700 hover:bg-indigo-100">⚔️ Gip thắng Zip</button>
      </div>
    </div>
  </div>

  <!-- Legend -->
  <div class="max-w-7xl mx-auto px-4 mb-3 flex items-center justify-between text-xs text-slate-500">
    <div id="results-count" class="font-medium">Đang tải dữ liệu...</div>
    <div class="flex items-center gap-3 text-[11px] bg-white px-3 py-1 rounded-lg border border-slate-200">
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-slate-700"></span> Khớp</span>
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-rose-400"></span> Bỏ sót (Gold)</span>
      <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> Nghe sai / Thêm (ASR)</span>
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
        class="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs rounded-xl shadow-xs transition disabled:opacity-50"
      >
        Tải thêm 40 mẫu tiếp theo...
      </button>
    </div>
  </main>

  <script>
    let allRecords = [];
    let currentFilter = 'all';
    let currentSearch = '';
    let currentViewMode = 'zip'; // 'zip', 'gip', 'both'
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
          if (gWords[i - 1].toLowerCase() === aWords[j - 1].toLowerCase()) {
            dp[i][j] = dp[i - 1][j - 1] + 1;
          } else {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
          }
        }
      }

      let i = m, j = n;
      const gDiff = [], aDiff = [];
      while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && gWords[i - 1].toLowerCase() === aWords[j - 1].toLowerCase()) {
          gDiff.unshift(`<span class="word-match">${escapeHtml(gWords[i - 1])}</span>`);
          aDiff.unshift(`<span class="word-match">${escapeHtml(aWords[j - 1])}</span>`);
          i--; j--;
        } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
          aDiff.unshift(`<span class="word-inserted" title="Từ nghe thừa/sai">${escapeHtml(aWords[j - 1])}</span>`);
          j--;
        } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
          gDiff.unshift(`<span class="word-missing" title="Từ bị bỏ sót">${escapeHtml(gWords[i - 1])}</span>`);
          i--;
        }
      }

      return {
        goldHtml: gDiff.join(' '),
        asrHtml: aDiff.join(' '),
      };
    }

    function getWerBadgeClass(wer) {
      if (wer === null || wer === undefined) return 'bg-slate-100 text-slate-600';
      if (wer === 0) return 'bg-emerald-100 text-emerald-800 border-emerald-300';
      if (wer <= 15) return 'bg-blue-100 text-blue-800 border-blue-300';
      if (wer <= 30) return 'bg-amber-100 text-amber-800 border-amber-300';
      return 'bg-rose-100 text-rose-800 border-rose-300';
    }

    function filterRecords() {
      return allRecords.filter(r => {
        // Text Search
        if (currentSearch) {
          const q = currentSearch.toLowerCase();
          const matchId = (r.id || '').toLowerCase().includes(q);
          const matchGold = (r.gold_text || '').toLowerCase().includes(q);
          const matchZip = (r.zip_final || '').toLowerCase().includes(q);
          const matchGip = (r.gip_final || '').toLowerCase().includes(q);
          if (!matchId && !matchGold && !matchZip && !matchGip) return false;
        }

        // Quick Filters
        if (currentFilter === 'perfect') return r.zip_wer === 0;
        if (currentFilter === 'good') return r.zip_wer > 0 && r.zip_wer <= 15;
        if (currentFilter === 'bad') return r.zip_wer > 25;
        if (currentFilter === 'zip_better') return r.zip_better;
        if (currentFilter === 'gip_better') return r.gip_better;
        return true;
      });
    }

    function renderCards(append = false) {
      const container = document.getElementById('cards-container');
      const filtered = filterRecords();
      const countEl = document.getElementById('results-count');
      const btnMore = document.getElementById('btn-load-more');

      countEl.innerText = `Hiển thị ${Math.min(currentOffset + PAGE_SIZE, filtered.length)} trên ${filtered.length} mẫu`;

      if (!append) {
        container.innerHTML = '';
        currentOffset = 0;
      }

      const slice = filtered.slice(currentOffset, currentOffset + PAGE_SIZE);

      if (filtered.length === 0) {
        container.innerHTML = `
          <div class="col-span-full py-16 text-center text-slate-400">
            <div class="text-4xl mb-2">🔍</div>
            <p class="text-sm font-medium">Không tìm thấy mẫu nào khớp với bộ lọc hiện tại.</p>
          </div>
        `;
        btnMore.style.display = 'none';
        return;
      }

      slice.forEach((r, idx) => {
        const diffZip = showWordDiff ? computeWordDiff(r.gold_text, r.zip_final) : null;
        const diffGip = showWordDiff ? computeWordDiff(r.gold_text, r.gip_final) : null;

        const card = document.createElement('div');
        card.className = "bg-white rounded-2xl border border-slate-200/80 p-4 shadow-xs hover:shadow-md transition flex flex-col justify-between";

        // Card Header
        let headerHtml = `
          <div class="flex items-center justify-between gap-2 pb-3 border-b border-slate-100">
            <div class="flex items-center gap-2">
              <span class="font-mono font-bold text-xs px-2 py-0.5 bg-slate-100 text-slate-700 rounded-md">#${r.id}</span>
              <span class="text-[11px] font-medium text-slate-400">${r.duration_sec}s</span>
            </div>
            <div class="flex items-center gap-1.5 text-xs font-semibold">
              <span class="px-2.5 py-0.5 rounded-full border ${getWerBadgeClass(r.zip_wer)}" title="ZipFormer WER">
                Zip: ${r.zip_wer}%
              </span>
              ${r.is_multi_model ? `
                <span class="px-2 py-0.5 rounded-full border ${getWerBadgeClass(r.gip_wer)} text-[11px]" title="Gipformer WER">
                  Gip: ${r.gip_wer}%
                </span>
              ` : ''}
              ${r.zip_better ? '<span class="text-xs" title="ZipFormer tốt hơn Gipformer">🏆</span>' : ''}
            </div>
          </div>
        `;

        // Audio Player Section
        const audioUrl = `/audio/${encodeURIComponent(r.audio_path)}`;
        let audioHtml = `
          <div class="py-3 flex flex-col gap-2">
            <audio id="audio-${r.id}" controls preload="none" class="w-full h-8">
              <source src="${audioUrl}" type="audio/wav">
              Trình duyệt không hỗ trợ audio.
            </audio>
            <div class="flex items-center justify-between text-[11px] text-slate-400 px-1">
              <span>Tốc độ:</span>
              <div class="flex gap-1">
                <button onclick="setSpeed('${r.id}', 0.75)" class="px-1.5 py-0.5 rounded hover:bg-slate-100">0.75x</button>
                <button onclick="setSpeed('${r.id}', 1.0)" class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">1x</button>
                <button onclick="setSpeed('${r.id}', 1.25)" class="px-1.5 py-0.5 rounded hover:bg-slate-100">1.25x</button>
                <button onclick="setSpeed('${r.id}', 1.5)" class="px-1.5 py-0.5 rounded hover:bg-slate-100">1.5x</button>
              </div>
            </div>
          </div>
        `;

        // Text Content Section
        let textContentHtml = '<div class="space-y-3 pt-2 text-xs">';

        // Gold text
        textContentHtml += `
          <div class="p-2.5 rounded-xl bg-slate-50 border border-slate-200/60">
            <div class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1 flex items-center gap-1">
              <span>🎯 Ground Truth (BTC)</span>
            </div>
            <div class="text-slate-800 leading-relaxed font-normal">
              ${diffZip ? diffZip.goldHtml : escapeHtml(r.gold_text)}
            </div>
          </div>
        `;

        // ZipFormer Box
        if (currentViewMode === 'zip' || currentViewMode === 'both') {
          textContentHtml += `
            <div class="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-200/70">
              <div class="text-[10px] font-bold uppercase tracking-wider text-emerald-800 mb-1 flex items-center justify-between">
                <span>⚡ ZipFormer-150M Final</span>
                <span class="text-[10px] font-normal text-emerald-600">${r.zip_latency_ms ? r.zip_latency_ms + 'ms' : ''}</span>
              </div>
              <div class="text-slate-900 leading-relaxed font-medium">
                ${diffZip ? diffZip.asrHtml : escapeHtml(r.zip_final)}
              </div>
              <details class="mt-1.5 text-[11px] text-slate-400">
                <summary class="cursor-pointer hover:text-slate-600">Acoustic thuần (chữ thường):</summary>
                <div class="mt-1 p-1.5 rounded bg-white text-slate-600 font-mono text-[10px]">
                  ${escapeHtml(r.zip_acoustic)}
                </div>
              </details>
            </div>
          `;
        }

        // Gipformer Box
        if ((currentViewMode === 'gip' || currentViewMode === 'both') && r.is_multi_model) {
          textContentHtml += `
            <div class="p-2.5 rounded-xl bg-indigo-50/50 border border-indigo-200/70">
              <div class="text-[10px] font-bold uppercase tracking-wider text-indigo-800 mb-1 flex items-center justify-between">
                <span>🤖 Gipformer-68M Final</span>
                <span class="text-[10px] font-normal text-indigo-600">${r.gip_latency_ms ? r.gip_latency_ms + 'ms' : ''}</span>
              </div>
              <div class="text-slate-900 leading-relaxed font-medium">
                ${diffGip ? diffGip.asrHtml : escapeHtml(r.gip_final)}
              </div>
            </div>
          `;
        }

        textContentHtml += '</div>';

        card.innerHTML = headerHtml + audioHtml + textContentHtml;
        container.appendChild(card);
      });

      currentOffset += slice.length;
      btnMore.style.display = currentOffset >= filtered.length ? 'none' : 'block';
    }

    function setSpeed(id, speed) {
      const audio = document.getElementById(`audio-${id}`);
      if (audio) audio.playbackRate = speed;
    }

    function toggleDiff() {
      showWordDiff = document.getElementById('diff-toggle').checked;
      renderCards(false);
    }

    function setViewMode(mode) {
      currentViewMode = mode;
      ['zip', 'both', 'gip'].forEach(m => {
        const btn = document.getElementById(`view-${m}`);
        if (m === mode) {
          btn.className = "px-3 py-1 rounded-lg bg-white text-slate-900 shadow-xs";
        } else {
          btn.className = "px-3 py-1 rounded-lg text-slate-600 hover:text-slate-900";
        }
      });
      renderCards(false);
    }

    function setFilter(filter) {
      currentFilter = filter;
      const tabs = ['all', 'perfect', 'good', 'bad', 'zip_better', 'gip_better'];
      tabs.forEach(t => {
        const btn = document.getElementById(`tab-${t}`);
        if (!btn) return;
        if (t === filter) {
          btn.className = "px-3 py-1 rounded-lg bg-slate-900 text-white shadow-xs";
        } else {
          btn.className = "px-3 py-1 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200";
        }
      });
      renderCards(false);
    }

    function loadMore() {
      renderCards(true);
    }

    // Debounce search
    let searchTimeout = null;
    document.getElementById('search-input').addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        currentSearch = e.target.value.trim();
        renderCards(false);
      }, 250);
    });

    // Fetch initial data
    fetch('/api/data')
      .then(res => res.json())
      .then(data => {
        allRecords = data.records;
        const stats = data.stats;

        if (stats) {
          document.getElementById('stat-total').innerText = `${stats.total} Mẫu`;
          if (stats.avg_zip_wer) document.getElementById('stat-zip-wer').innerText = `Zip WER: ${stats.avg_zip_wer}%`;
          if (stats.avg_gip_wer) document.getElementById('stat-gip-wer').innerText = `Gip WER: ${stats.avg_gip_wer}%`;
          if (stats.exact_match_zip_pct) document.getElementById('stat-exact').innerText = `Khớp 100%: ${stats.exact_match_zip_pct}%`;
        }

        renderCards(false);
      })
      .catch(err => {
        document.getElementById('results-count').innerText = `Lỗi tải dữ liệu: ${err}`;
      });
  </script>
</body>
</html>
"""


def find_audio_file(audio_rel_or_id: str, search_roots: list[Path]) -> Path | None:
    """Tìm file audio thực tế trên ổ cứng cục bộ."""
    clean_path = audio_rel_or_id.replace("\\", "/").strip("/")
    sample_id = Path(clean_path).stem

    candidates: list[Path] = []
    # 1. Đường dẫn trực tiếp từ các search_roots
    for root in search_roots:
        candidates.append(root / clean_path)
        candidates.append(root / f"{clean_path}.wav")
        # Dạng audio/train_0001/train_0001.wav
        candidates.append(root / "audio" / sample_id / f"{sample_id}.wav")
        candidates.append(root / sample_id / f"{sample_id}.wav")
        # Bỏ prefix 'audio/'
        if "audio/" in clean_path:
            after_audio = clean_path.split("audio/", 1)[1]
            candidates.append(root / "audio" / after_audio)
            candidates.append(root / after_audio)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


class ASRViewerHandler(BaseHTTPRequestHandler):
    store: ASRDataStore
    search_roots: list[Path]

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/data":
            payload = {
                "records": self.store.records,
                "stats": self.store.stats,
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
    parser = argparse.ArgumentParser(description="Chạy Web Server xem và nghe kết quả ZipFormer/Gipformer ASR")
    parser.add_argument("--port", type=int, default=8080, help="Cổng web (mặc định: 8080)")
    parser.add_argument(
        "--parquet",
        type=str,
        default="logs/asr_benchmark_500_details.parquet",
        help="Đường dẫn file parquet (mặc định: logs/asr_benchmark_500_details.parquet)",
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Thư mục chứa dữ liệu")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        fallback_candidates = [
            Path("logs/asr_benchmark_500_details (1).parquet"),
            Path("data/cache/asr_cache.parquet"),
            Path("asr_cache.parquet"),
        ]
        for fb in fallback_candidates:
            if fb.exists():
                parquet_path = fb
                break

    data_dir = Path(args.data_dir)
    store = ASRDataStore(parquet_path=parquet_path, data_dir=data_dir)

    search_roots = [
        data_dir,
        data_dir / "train",
        data_dir / "train/audio",
        data_dir / "data/task1_lifemate/train+public_test/train",
        data_dir / "data/task1_lifemate/train+public_test/train/audio",
        Path("train"),
        Path("train/audio"),
        Path("data/task1_lifemate/train+public_test/train"),
        Path("data/task1_lifemate/train+public_test/train/audio"),
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

    if server is None:
        print(f"Lỗi: Không mở được cổng HTTP trong khoảng {port} - {port + 10}")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print("=" * 70)
    print("🚀 ASR VOICE & TEXT STUDIO ĐÃ KHỞI CHẠY!")
    print(f"👉 Truy cập ngay tại: {url}")
    print(f"📊 Dữ liệu đang hiển thị: {parquet_path.resolve()} ({len(store.records)} mẫu)")
    print("=" * 70)

    with contextlib.suppress(Exception):
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang tắt server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
