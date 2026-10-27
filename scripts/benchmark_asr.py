#!/usr/bin/env python3
"""
scripts/benchmark_asr.py

Thực nghiệm đánh giá chuẩn chỉnh hệ thống ASR 4 tầng trên 500 mẫu đầu tiên của dataset:
So sánh 2 mô hình Transducer:
1. g-group-ai-lab/gipformer1.5-68M-rnnt (~68M params)
2. hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (~153M params)

Cả 2 mô hình đều trải qua trọn vẹn 4 tầng:
- Tầng 1: Peak Normalization [-0.95, 0.95] + Silero VAD (min_silence_duration=0.4s)
- Tầng 2: Sherpa-ONNX Transducer Recognizer (modified_beam_search, max_active_paths=4)
- Tầng 3: XLM-RoBERTa Punctuation & Capitalization Restoration (dragonSwing/xlm-roberta-capu)
- Tầng 4: Vietnamese ITN (chuyển đổi số nói -> số viết cho ngày giờ, tiền tệ, tầng, số lượng)

6 Tiêu chuẩn định lượng (Corpus-level):
1. Normalized WER & CER (%)
2. Raw WER & CER (%)
3. Exact Sentence Match (%) [Normalized & Raw]
4. Number/Digit Recall (%)
5. Capitalized Entity Recall (%)
6. Latency (ms/sample) & RTF (Real-Time Factor)

Xuất kết quả:
- logs/asr_benchmark_500_summary.md
- logs/asr_benchmark_500_details.parquet
- logs/asr_500_error_analysis.md (50 mẫu có sai số lớn nhất hoặc lệch nhau)

Chạy lệnh:
    python scripts/benchmark_asr.py --train-jsonl train/train.jsonl --audio-root train --limit 500
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")

import jiwer
import pandas as pd
import torch
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path để import từ src
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.sherpa_xlm_pipeline import SherpaXLMPipeline
from src.eval.asr_metrics import (
    ASRCorpusMetrics,
    compute_corpus_metrics,
    normalize_text_for_acoustic_eval,
)


def find_dataset_files(jsonl_arg: str, audio_arg: str) -> tuple[Path, Path]:
    """Tự động tìm kiếm file train.jsonl và thư mục audio trên cả máy local lẫn Google Colab."""
    candidates_jsonl = [
        Path(jsonl_arg),
        REPO_ROOT / jsonl_arg,
        Path.cwd() / jsonl_arg,
        REPO_ROOT / "train" / "train.jsonl",
        REPO_ROOT / "data" / "train" / "train.jsonl",
        REPO_ROOT / "data" / "task1_lifemate" / "train+public_test" / "train" / "train.jsonl",
        # Các đường dẫn phổ biến trên Google Colab
        Path("/content") / jsonl_arg,
        Path("/content/train/train.jsonl"),
        Path("/content/train+public_test/train/train.jsonl"),
        Path("/content/data/train/train.jsonl"),
        Path("/content/data/task1_lifemate/train+public_test/train/train.jsonl"),
    ]

    resolved_jsonl: Path | None = None
    for p in candidates_jsonl:
        if p.exists() and p.is_file():
            resolved_jsonl = p.resolve()
            break

    # Nếu vẫn chưa tìm thấy, quét toàn diện trong /content và thư mục cha
    if resolved_jsonl is None:
        search_roots = [Path.cwd(), REPO_ROOT, REPO_ROOT.parent]
        colab_content = Path("/content")
        if colab_content.exists():
            search_roots.insert(0, colab_content)

        for root in search_roots:
            try:
                found = list(root.rglob("train.jsonl"))
                if found:
                    resolved_jsonl = found[0].resolve()
                    break
            except (OSError, RuntimeError):
                continue

    if resolved_jsonl is None:
        searched_info = "\n  - ".join(str(c) for c in candidates_jsonl)
        raise FileNotFoundError(
            f"Không tìm thấy file dataset '{jsonl_arg}'.\n"
            f"Đã kiểm tra các vị trí sau:\n  - {searched_info}\n"
            f"Gợi ý cho Colab: hãy truyền --train-jsonl /đường/dẫn/tới/train.jsonl"
        )

    # Tìm thư mục audio tương ứng
    candidates_audio = [
        Path(audio_arg),
        REPO_ROOT / audio_arg,
        Path.cwd() / audio_arg,
        resolved_jsonl.parent,
        resolved_jsonl.parent / "audio",
        resolved_jsonl.parent.parent / "audio",
        Path("/content") / audio_arg,
        Path("/content/train"),
        Path("/content/train+public_test/train"),
    ]

    resolved_audio: Path | None = None
    for p in candidates_audio:
        if p.exists() and (p / "audio").exists():
            resolved_audio = p.resolve()
            break
        if p.exists() and p.is_dir() and any(p.glob("*.wav")):
            resolved_audio = p.resolve()
            break

    if resolved_audio is None:
        # Fallback về thư mục chứa train.jsonl
        resolved_audio = resolved_jsonl.parent

    print(f"[TÌM THẤY DỮ LIỆU] Dataset: {resolved_jsonl}")
    print(f"[TÌM THẤY DỮ LIỆU] Audio Root: {resolved_audio}")
    return resolved_jsonl, resolved_audio


def resolve_sample_audio_path(audio_root: Path, item: dict[str, Any]) -> Path | None:
    """Xác định đường dẫn tuyệt đối của file wav tương ứng với 1 sample."""
    audio_field = item.get("audio") or item.get("audio_path")
    sample_id = item.get("id", "")

    candidates: list[Path] = []
    if audio_field:
        candidates.append(audio_root / audio_field)
        candidates.append(audio_root / "audio" / audio_field)
        clean_field = audio_field.replace("audio/", "").replace("audio\\", "")
        candidates.append(audio_root / clean_field)
        candidates.append(audio_root / "audio" / clean_field)
        candidates.append(audio_root.parent / audio_field)
        candidates.append(audio_root.parent / "audio" / clean_field)
    if sample_id:
        candidates.append(audio_root / "audio" / sample_id / f"{sample_id}.wav")
        candidates.append(audio_root / sample_id / f"{sample_id}.wav")
        candidates.append(audio_root.parent / "audio" / sample_id / f"{sample_id}.wav")

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def run_asr_benchmark(
    *,
    train_jsonl: Path,
    audio_root: Path,
    limit: int = 500,
    device: str = "cpu",
    output_summary: Path = Path("logs/asr_benchmark_500_summary.md"),
    output_details: Path = Path("logs/asr_benchmark_500_details.parquet"),
    output_errors: Path = Path("logs/asr_500_error_analysis.md"),
) -> None:
    print("=" * 80)
    print("ZALO AI CHALLENGE 2026 - TASK 1: BENCHMARK ASR 4 TẦNG (500 MẪU)")
    print("Mô hình so sánh: Gipformer-68M vs ZipFormer-150M (+ Silero VAD + Capu + ITN)")
    print("=" * 80)
    print(f"Dataset JSONL : {train_jsonl}")
    print(f"Audio Root    : {audio_root}")
    print(f"Giới hạn mẫu  : {limit}")
    print(f"Thiết bị      : {device}")
    print("-" * 80)

    # 1. Đọc dữ liệu
    samples: list[dict[str, Any]] = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            gold_text = item.get("audio_text") or item.get("transcript") or item.get("text")
            if not gold_text:
                continue
            audio_path = resolve_sample_audio_path(audio_root, item)
            if audio_path is None:
                continue
            item["_resolved_audio"] = audio_path
            item["_gold_text"] = gold_text
            samples.append(item)
            if len(samples) >= limit:
                break

    print(f"[OK] Đã nạp thành công {len(samples)} mẫu hợp lệ kèm file audio.")
    if not samples:
        raise ValueError("Không tìm thấy mẫu nào hợp lệ để đánh giá!")

    # 2. Khởi tạo Pipeline cho cả 2 model
    print("\n[1/3] Đang khởi tạo Pipeline Gipformer-68M (+ nạp chung tầng Capu)...")
    pipe_gip = SherpaXLMPipeline(
        model_name_or_dir="models/asr/gipformer-68m",
        vad_model_path="models/asr/vad/silero_vad.onnx",
        capu_model_dir="models/asr/xlm-roberta-capu",
        device=device,
        num_threads=4,
        enable_capu=True,
        enable_itn=True,
    )

    print("[2/3] Đang khởi tạo Pipeline ZipFormer-150M (tái sử dụng tầng Capu)...")
    pipe_zip = SherpaXLMPipeline(
        model_name_or_dir="models/asr/zipformer-150m",
        vad_model_path="models/asr/vad/silero_vad.onnx",
        capu_model=pipe_gip.capu_model,
        device=device,
        num_threads=4,
        enable_capu=True,
        enable_itn=True,
    )

    # Warmup
    print("[3/3] Đang warmup cả 2 pipeline...")
    first_wav = samples[0]["_resolved_audio"]
    pipe_gip.transcribe(first_wav)
    pipe_zip.transcribe(first_wav)
    print("[OK] Warmup hoàn tất!\n")

    # 3. Chạy thực nghiệm trên toàn bộ samples
    records: list[dict[str, Any]] = []

    print(f"Bắt đầu thực nghiệm ASR trên {len(samples)} mẫu...")
    pbar = tqdm(samples, desc="Benchmarking ASR", unit="sample")

    for s in pbar:
        wav_path: Path = s["_resolved_audio"]
        gold_raw: str = s["_gold_text"]
        gold_norm: str = normalize_text_for_acoustic_eval(gold_raw)

        # Chạy Gipformer
        res_gip = pipe_gip.transcribe(wav_path)
        gip_norm = normalize_text_for_acoustic_eval(res_gip.final_text)

        # Chạy ZipFormer
        res_zip = pipe_zip.transcribe(wav_path)
        zip_norm = normalize_text_for_acoustic_eval(res_zip.final_text)

        # WER riêng cho mẫu này để phân tích lỗi
        clean_g_norm = gold_norm if gold_norm else "<empty>"
        wer_gip_sample = float(jiwer.wer(clean_g_norm, gip_norm if gip_norm else "<empty>")) * 100.0
        wer_zip_sample = float(jiwer.wer(clean_g_norm, zip_norm if zip_norm else "<empty>")) * 100.0

        records.append({
            "id": s.get("id", ""),
            "audio_path": str(wav_path),
            "audio_duration_sec": res_gip.duration_sec,
            "gold_text": gold_raw,
            "gold_norm": gold_norm,
            # Gipformer
            "gip_raw_acoustic": res_gip.raw_acoustic_text,
            "gip_capu": res_gip.capu_text,
            "gip_final": res_gip.final_text,
            "gip_final_norm": gip_norm,
            "gip_latency_ms": res_gip.inference_time_sec * 1000.0,
            "gip_infer_time_sec": res_gip.inference_time_sec,
            "gip_wer_sample": wer_gip_sample,
            # ZipFormer
            "zip_raw_acoustic": res_zip.raw_acoustic_text,
            "zip_capu": res_zip.capu_text,
            "zip_final": res_zip.final_text,
            "zip_final_norm": zip_norm,
            "zip_latency_ms": res_zip.inference_time_sec * 1000.0,
            "zip_infer_time_sec": res_zip.inference_time_sec,
            "zip_wer_sample": wer_zip_sample,
            # Độ chênh lệch giữa 2 model
            "wer_diff": abs(wer_gip_sample - wer_zip_sample),
            "max_wer": max(wer_gip_sample, wer_zip_sample),
        })

    # 4. Tính toán Corpus Metrics
    gold_texts = [r["gold_text"] for r in records]
    durations = [r["audio_duration_sec"] for r in records]

    # Metrics Gipformer
    gip_finals = [r["gip_final"] for r in records]
    gip_times = [r["gip_infer_time_sec"] for r in records]
    gip_metrics: ASRCorpusMetrics = compute_corpus_metrics(
        gold_texts=gold_texts,
        pred_raw_texts=gip_finals,
        audio_durations_sec=durations,
        inference_times_sec=gip_times,
    )

    # Metrics ZipFormer
    zip_finals = [r["zip_final"] for r in records]
    zip_times = [r["zip_infer_time_sec"] for r in records]
    zip_metrics: ASRCorpusMetrics = compute_corpus_metrics(
        gold_texts=gold_texts,
        pred_raw_texts=zip_finals,
        audio_durations_sec=durations,
        inference_times_sec=zip_times,
    )

    # 5. In bảng kết quả Markdown
    summary_md = f"""# BÁO CÁO ĐÁNH GIÁ SO SÁNH ASR 4 TẦNG (500 MẪU)
**Dự án:** Zalo AI Challenge 2026 - Task 1 (Lifemate)  
**Thời gian thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Tập dữ liệu:** `{train_jsonl.name}` ({len(records)} mẫu) | Tổng thời lượng audio: {gip_metrics.total_audio_duration_sec:.1f}s ({gip_metrics.total_audio_duration_sec / 60:.1f} phút)  
**Cấu hình phần cứng & Thiết bị:** `{device.upper()}`

---

## 1. Bảng So Sánh Toàn Diện 6 Tiêu Chuẩn Định Lượng

| Tiêu chí đánh giá | Gipformer-68M (~68M params) | ZipFormer-150M (~153M params) | Chênh lệch (Zip - Gip) | Mô hình tốt hơn |
| :--- | :---: | :---: | :---: | :---: |
| **Normalized WER (%)** (Acoustic thuần) | **{gip_metrics.normalized_wer:.2f}%** | **{zip_metrics.normalized_wer:.2f}%** | {zip_metrics.normalized_wer - gip_metrics.normalized_wer:+.2f}% | {'ZipFormer' if zip_metrics.normalized_wer < gip_metrics.normalized_wer else 'Gipformer'} |
| **Normalized CER (%)** | **{gip_metrics.normalized_cer:.2f}%** | **{zip_metrics.normalized_cer:.2f}%** | {zip_metrics.normalized_cer - gip_metrics.normalized_cer:+.2f}% | {'ZipFormer' if zip_metrics.normalized_cer < gip_metrics.normalized_cer else 'Gipformer'} |
| **Raw WER (%)** (Sau Capu + ITN) | **{gip_metrics.raw_wer:.2f}%** | **{zip_metrics.raw_wer:.2f}%** | {zip_metrics.raw_wer - gip_metrics.raw_wer:+.2f}% | {'ZipFormer' if zip_metrics.raw_wer < gip_metrics.raw_wer else 'Gipformer'} |
| **Raw CER (%)** (Sau Capu + ITN) | **{gip_metrics.raw_cer:.2f}%** | **{zip_metrics.raw_cer:.2f}%** | {zip_metrics.raw_cer - gip_metrics.raw_cer:+.2f}% | {'ZipFormer' if zip_metrics.raw_cer < gip_metrics.raw_cer else 'Gipformer'} |
| **Exact Sentence Match (Normalized)** | **{gip_metrics.exact_match_norm_pct:.1f}%** | **{zip_metrics.exact_match_norm_pct:.1f}%** | {zip_metrics.exact_match_norm_pct - gip_metrics.exact_match_norm_pct:+.1f}% | {'ZipFormer' if zip_metrics.exact_match_norm_pct > gip_metrics.exact_match_norm_pct else 'Gipformer'} |
| **Exact Sentence Match (Raw)** | **{gip_metrics.exact_match_raw_pct:.1f}%** | **{zip_metrics.exact_match_raw_pct:.1f}%** | {zip_metrics.exact_match_raw_pct - gip_metrics.exact_match_raw_pct:+.1f}% | {'ZipFormer' if zip_metrics.exact_match_raw_pct > gip_metrics.exact_match_raw_pct else 'Gipformer'} |
| **Number / Digit Recall (%)** | **{gip_metrics.number_recall_pct:.1f}%** ({gip_metrics.matched_gold_numbers}/{gip_metrics.total_gold_numbers}) | **{zip_metrics.number_recall_pct:.1f}%** ({zip_metrics.matched_gold_numbers}/{zip_metrics.total_gold_numbers}) | {zip_metrics.number_recall_pct - gip_metrics.number_recall_pct:+.1f}% | {'ZipFormer' if zip_metrics.number_recall_pct > gip_metrics.number_recall_pct else 'Gipformer'} |
| **Capitalized Entity Recall (%)** | **{gip_metrics.capitalized_recall_pct:.1f}%** ({gip_metrics.matched_gold_entities}/{gip_metrics.total_gold_entities}) | **{zip_metrics.capitalized_recall_pct:.1f}%** ({zip_metrics.matched_gold_entities}/{zip_metrics.total_gold_entities}) | {zip_metrics.capitalized_recall_pct - gip_metrics.capitalized_recall_pct:+.1f}% | {'ZipFormer' if zip_metrics.capitalized_recall_pct > gip_metrics.capitalized_recall_pct else 'Gipformer'} |
| **Độ trễ trung bình (ms/sample)** | **{gip_metrics.avg_latency_ms:.1f} ms** | **{zip_metrics.avg_latency_ms:.1f} ms** | {zip_metrics.avg_latency_ms - gip_metrics.avg_latency_ms:+.1f} ms | {'Gipformer (Nhanh hơn)' if gip_metrics.avg_latency_ms < zip_metrics.avg_latency_ms else 'ZipFormer'} |
| **Real-Time Factor (RTF)** | **{gip_metrics.real_time_factor:.4f}** | **{zip_metrics.real_time_factor:.4f}** | {zip_metrics.real_time_factor - gip_metrics.real_time_factor:+.4f} | {'Gipformer' if gip_metrics.real_time_factor < zip_metrics.real_time_factor else 'ZipFormer'} |

---

## 2. Phân Tích Kỹ Thuật Chuyên Sâu (Falsify Protocol)

1. **Hiệu năng Âm học (Acoustic Performance):**
   - Gipformer-68M có Normalized WER là `{gip_metrics.normalized_wer:.2f}%`.
   - ZipFormer-150M có Normalized WER là `{zip_metrics.normalized_wer:.2f}%`.
2. **Vai trò của XLM-RoBERTa Capu & Vietnamese ITN:**
   - Khôi phục chính xác các thực thể viết hoa (Tên riêng như Lan, Dũng, SJC, Ba, Vy...) đạt `{max(gip_metrics.capitalized_recall_pct, zip_metrics.capitalized_recall_pct):.1f}%`.
   - Khôi phục số lượng và ngày giờ đạt Number Recall `{max(gip_metrics.number_recall_pct, zip_metrics.number_recall_pct):.1f}%`.
3. **Ngân sách tham số & Tốc độ thực thi:**
   - Gipformer Pipeline: ~348M tham số, RTF = `{gip_metrics.real_time_factor:.4f}`.
   - ZipFormer Pipeline: ~433M tham số, RTF = `{zip_metrics.real_time_factor:.4f}`.
   - Cả 2 phương án đều bảo toàn > 5.5B tham số cho LLM và đáp ứng trần < 6B của đề thi.
"""

    print("\n" + summary_md)

    # 6. Lưu file Summary
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"[OK] Đã lưu báo cáo tổng hợp tại: {output_summary}")

    # 7. Lưu file Parquet chi tiết
    output_details.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(output_details, index=False)
    print(f"[OK] Đã lưu chi tiết 500 mẫu tại: {output_details} ({len(df)} dòng)")

    # 8. Xuất file Error Analysis (Top 50 mẫu lỗi nhất / lệch nhất)
    # Sắp xếp theo max_wer giảm dần
    sorted_errors = sorted(records, key=lambda x: (x["max_wer"], x["wer_diff"]), reverse=True)[:50]

    err_lines = [
        "# PHÂN TÍCH 50 MẪU SAI SỐ LỚN NHẤT / LỆCH NHAU NHẤT (ERROR ANALYSIS)",
        f"**Tổng số mẫu khảo sát:** {len(records)} | **Số mẫu được phân tích chi tiết:** {len(sorted_errors)}",
        "",
        "Mỗi mẫu hiển thị 4 dòng định dạng chuẩn:",
        "- **Gold**: Văn bản đối chiếu gốc",
        "- **Raw Acoustic**: Đầu ra âm học chữ thường (Gipformer vs ZipFormer)",
        "- **Gipformer Final**: Văn bản hoàn chỉnh sau Capu + ITN của Gipformer-68M",
        "- **ZipFormer Final**: Văn bản hoàn chỉnh sau Capu + ITN của ZipFormer-150M",
        "",
        "---",
        "",
    ]

    for idx, item in enumerate(sorted_errors, start=1):
        err_lines.append(f"### Mẫu {idx} [ID: {item['id']}] - Gip WER: {item['gip_wer_sample']:.1f}% | Zip WER: {item['zip_wer_sample']:.1f}%")
        err_lines.append(f"- **Gold**: `{item['gold_text']}`")
        err_lines.append(f"- **Raw Acoustic**: Gip: `{item['gip_raw_acoustic']}` | Zip: `{item['zip_raw_acoustic']}`")
        err_lines.append(f"- **Gipformer Final**: `{item['gip_final']}`")
        err_lines.append(f"- **ZipFormer Final**: `{item['zip_final']}`")
        err_lines.append("")

    output_errors.parent.mkdir(parents=True, exist_ok=True)
    with open(output_errors, "w", encoding="utf-8") as f:
        f.write("\n".join(err_lines))
    print(f"[OK] Đã lưu 50 mẫu lỗi tại: {output_errors}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASR 4 tầng trên 500 mẫu dataset Zalo AI Challenge 2026")
    parser.add_argument("--train-jsonl", type=str, default="train/train.jsonl", help="Đường dẫn file train.jsonl")
    parser.add_argument("--audio-root", type=str, default="train", help="Thư mục gốc chứa audio")
    parser.add_argument("--limit", type=int, default=500, help="Số lượng mẫu đánh giá (mặc định: 500)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy Capu & sherpa (cuda hoặc cpu)",
    )
    parser.add_argument("--output-summary", type=str, default="logs/asr_benchmark_500_summary.md")
    parser.add_argument("--output-details", type=str, default="logs/asr_benchmark_500_details.parquet")
    parser.add_argument("--output-errors", type=str, default="logs/asr_500_error_analysis.md")
    args = parser.parse_args()

    jsonl_path, audio_path = find_dataset_files(args.train_jsonl, args.audio_root)

    run_asr_benchmark(
        train_jsonl=jsonl_path,
        audio_root=audio_path,
        limit=args.limit,
        device=args.device,
        output_summary=Path(args.output_summary),
        output_details=Path(args.output_details),
        output_errors=Path(args.output_errors),
    )


if __name__ == "__main__":
    main()
