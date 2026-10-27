#!/usr/bin/env python3
"""
scripts/benchmark_asr.py

Thực nghiệm so sánh 2 mô hình ASR cho Task 1 (Zalo AI Challenge 2026):
- PhoASR-whisper-small-ct2 (0.24B tham số)
- PhoWhisper-large-ct2 (1.54B tham số)

Cấu hình:
- faster_whisper.WhisperModel
- vad_filter=True (Silero VAD)
- beam_size=5, temperature=0.0, language='vi'
- without_timestamps=True (chỉ cần pure text, tối ưu tốc độ)

Tiêu chí đánh giá:
1. Raw WER & CER: So sánh trực tiếp với audio_text gốc (giữ nguyên hoa/thường, dấu câu).
2. Normalized WER & CER: Sau khi lowercase, chuẩn hóa NFC, loại bỏ dấu câu.
3. Tốc độ trung bình (ms/sample) & Peak VRAM / RAM.
4. Xuất file logs/asr_comparison_diff.md chứa 30 mẫu có chênh lệch WER lớn nhất.

Hỗ trợ chạy trên Google Colab qua lệnh:
    !python scripts/benchmark_asr.py --train-jsonl data/train/train.jsonl --audio-root data/train --limit 200
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")

# Khắc phục tương thích PyAV 19 (bỏ metadata_errors)
with contextlib.suppress(Exception):
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open

import faster_whisper
import jiwer
import torch


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    - Unicode NFC
    - Chuyển thành chữ thường (lowercase)
    - Loại bỏ dấu câu và ký tự đặc biệt
    - Xóa khoảng trắng thừa
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Loại bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text)
    # Gom khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_audio_path(audio_path_str: str, audio_roots: list[Path]) -> Path:
    """Tìm đường dẫn file audio thực tế từ danh sách các thư mục ứng viên."""
    candidate = Path(audio_path_str)
    if candidate.exists():
        return candidate

    for root in audio_roots:
        p1 = root / audio_path_str
        if p1.exists():
            return p1
        # Thử bỏ prefix "audio/" nếu root đã trỏ vào audio
        if audio_path_str.startswith("audio/"):
            p2 = root / audio_path_str[6:]
            if p2.exists():
                return p2

    raise FileNotFoundError(
        f"Không tìm thấy file audio '{audio_path_str}' trong các thư mục: {[str(r) for r in audio_roots]}"
    )


def load_dataset(jsonl_path: Path, limit: int) -> list[dict[str, Any]]:
    """Đọc dữ liệu từ file JSONL."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_peak_memory_mb(device: str) -> float:
    """Lấy lượng bộ nhớ đỉnh điểm (Peak VRAM nếu GPU, Peak RAM nếu CPU)."""
    if device == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    with contextlib.suppress(Exception):
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    return 0.0


def benchmark_single_model(
    model_name: str,
    model_path: str,
    samples: list[dict[str, Any]],
    audio_roots: list[Path],
    device: str,
    compute_type: str,
) -> tuple[list[str], float, float]:
    """
    Chạy suy luận cho 1 mô hình trên danh sách mẫu và đo thời gian, bộ nhớ.
    """
    print(f"\n--- Đang tải mô hình: {model_name} (path: {model_path}) ---")
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_load_start = time.perf_counter()
    model = faster_whisper.WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
    )
    t_load = time.perf_counter() - t_load_start
    print(f"[{model_name}] Thời gian nạp model: {t_load:.2f}s | Device: {device} | Compute: {compute_type}")

    predictions: list[str] = []
    latencies: list[float] = []

    print(f"[{model_name}] Đang suy luận trên {len(samples)} mẫu...")
    for idx, sample in enumerate(samples):
        wav_path = resolve_audio_path(sample["audio"], audio_roots)
        t0 = time.perf_counter()

        # Phiên âm với VAD filter, beam=5, greedy temperature=0.0
        segments, _ = model.transcribe(
            str(wav_path),
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
            language="vi",
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        predictions.append(text)
        latencies.append(elapsed_ms)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            avg_so_far = sum(latencies) / len(latencies)
            print(f"  -> Tiến độ: {idx + 1}/{len(samples)} | Trung bình: {avg_so_far:.1f} ms/mẫu")

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    peak_mem = get_peak_memory_mb(device)

    # Giải phóng bộ nhớ model
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, avg_latency_ms, peak_mem


def export_diff_report(
    output_path: Path,
    samples: list[dict[str, Any]],
    preds_small: list[str],
    preds_large: list[str],
    top_k: int = 30,
) -> None:
    """
    Xuất file markdown gồm top K câu có độ chênh lệch WER lớn nhất giữa 2 mô hình.
    """
    diff_records = []
    for i, sample in enumerate(samples):
        gold = sample.get("audio_text", "").strip()
        p_small = preds_small[i]
        p_large = preds_large[i]

        # Tính WER từng câu
        wer_small = jiwer.wer(gold, p_small) if gold else 0.0
        wer_large = jiwer.wer(gold, p_large) if gold else 0.0
        delta = abs(wer_small - wer_large)

        diff_records.append({
            "id": sample.get("id", f"sample_{i}"),
            "gold": gold,
            "p_small": p_small,
            "p_large": p_large,
            "wer_small": wer_small,
            "wer_large": wer_large,
            "delta": delta,
        })

    # Sắp xếp giảm dần theo chênh lệch WER
    diff_records.sort(key=lambda x: x["delta"], reverse=True)
    top_records = diff_records[:top_k]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐỐI SOÁT PHIÊN ÂM ASR (TOP 30 MẪU CHÊNH LỆCH LỚN NHẤT)\n\n")
        f.write("> **Mục đích:** Đánh giá khả năng viết hoa tên riêng (ví dụ: 'Sếp Dũng', 'Đà Nẵng'), ")
        f.write("chuẩn hóa chữ số/giờ giấc và độ bền âm học giữa `PhoASR-small` và `PhoWhisper-large`.\n\n")

        f.write("| STT | Question ID | Chênh lệch Delta | WER PhoASR-small | WER PhoWhisper-large |\n")
        f.write("|---|---|---|---|---|\n")
        f.writelines(f"| {rank} | `{r['id']}` | {r['delta']:.2f} | {r['wer_small']:.2f} | {r['wer_large']:.2f} |\n" for rank, r in enumerate(top_records, 1))

        f.write("\n---\n\n## CHI TIẾT TỪNG MẪU\n\n")
        for rank, r in enumerate(top_records, 1):
            f.write(f"### Mẫu {rank}: `{r['id']}` (Delta WER: {r['delta']:.2f})\n")
            f.write(f"- **Chuẩn BTC (Gold):** `{r['gold']}`\n")
            f.write(f"- **PhoASR-whisper-small (0.24B):** `{r['p_small']}` *(WER: {r['wer_small']:.2f})*\n")
            f.write(f"- **PhoWhisper-large (1.54B):** `{r['p_large']}` *(WER: {r['wer_large']:.2f})*\n\n")

    print(f"\n[OK] Đã xuất báo cáo đối soát tại: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASR Models for Zalo AI Challenge 2026")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default="train/train.jsonl",
        help="Đường dẫn file train.jsonl (mặc định: train/train.jsonl hoặc data/train/train.jsonl)",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default="train",
        help="Thư mục chứa audio gốc (mặc định: train hoặc data/train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Số lượng mẫu benchmark (mặc định: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị suy luận: cuda hoặc cpu (tự động phát hiện)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16" if torch.cuda.is_available() else "int8",
        help="Kiểu tính toán CTranslate2: float16, int8, float32 (mặc định: float16 trên GPU, int8 trên CPU)",
    )
    parser.add_argument(
        "--output-diff",
        type=str,
        default="logs/asr_comparison_diff.md",
        help="Đường dẫn file markdown xuất 30 câu khác nhau nhất",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/asr",
        help="Thư mục chứa các mô hình đã convert",
    )
    args = parser.parse_args()

    # Tìm file train.jsonl
    train_path = Path(args.train_jsonl)
    if not train_path.exists():
        fallback = Path("data/train/train.jsonl")
        if fallback.exists():
            train_path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file dataset: {args.train_jsonl}")

    # Danh sách các thư mục audio ứng viên
    audio_roots = [
        Path(args.audio_root),
        Path("train"),
        Path("data/train"),
        Path("train/audio"),
        Path("data/train/audio"),
    ]

    models_dir = Path(args.models_dir)
    model_small_path = models_dir / "PhoASR-whisper-small-ct2"
    model_large_path = models_dir / "PhoWhisper-large-ct2"

    if not model_small_path.exists():
        raise FileNotFoundError(f"Chưa có model: {model_small_path}. Hãy chạy scripts/setup_asr_models.py trước.")
    if not model_large_path.exists():
        raise FileNotFoundError(f"Chưa có model: {model_large_path}. Hãy chạy scripts/setup_asr_models.py trước.")

    print("=" * 70)
    print("THIẾT LẬP THỰC NGHIỆM ASR BENCHMARK (TASK 1 - LIFEMATE)")
    print("=" * 70)
    print(f"Dataset path   : {train_path}")
    print(f"Số mẫu kiểm thử: {args.limit}")
    print(f"Device         : {args.device}")
    print(f"Compute Type   : {args.compute_type}")
    print(f"Model Small    : {model_small_path}")
    print(f"Model Large    : {model_large_path}")

    # Đọc dữ liệu
    samples = load_dataset(train_path, limit=args.limit)
    print(f"Đã nạp {len(samples)} mẫu từ {train_path}")

    # Benchmark Model 1: PhoASR-whisper-small-ct2 (0.24B)
    preds_small, lat_small, mem_small = benchmark_single_model(
        model_name="PhoASR-whisper-small-ct2 (0.24B)",
        model_path=str(model_small_path),
        samples=samples,
        audio_roots=audio_roots,
        device=args.device,
        compute_type=args.compute_type,
    )

    # Benchmark Model 2: PhoWhisper-large-ct2 (1.54B)
    preds_large, lat_large, mem_large = benchmark_single_model(
        model_name="PhoWhisper-large-ct2 (1.54B)",
        model_path=str(model_large_path),
        samples=samples,
        audio_roots=audio_roots,
        device=args.device,
        compute_type=args.compute_type,
    )

    # Tính toán Metrics
    gold_raw = [s.get("audio_text", "").strip() for s in samples]
    gold_norm = [normalize_vietnamese_text(t) for t in gold_raw]

    preds_small_norm = [normalize_vietnamese_text(t) for t in preds_small]
    preds_large_norm = [normalize_vietnamese_text(t) for t in preds_large]

    # Raw metrics
    raw_wer_small = jiwer.wer(gold_raw, preds_small) * 100.0
    raw_cer_small = jiwer.cer(gold_raw, preds_small) * 100.0

    raw_wer_large = jiwer.wer(gold_raw, preds_large) * 100.0
    raw_cer_large = jiwer.cer(gold_raw, preds_large) * 100.0

    # Normalized metrics
    norm_wer_small = jiwer.wer(gold_norm, preds_small_norm) * 100.0
    norm_cer_small = jiwer.cer(gold_norm, preds_small_norm) * 100.0

    norm_wer_large = jiwer.wer(gold_norm, preds_large_norm) * 100.0
    norm_cer_large = jiwer.cer(gold_norm, preds_large_norm) * 100.0

    mem_unit = "MB VRAM (Peak)" if args.device == "cuda" else "MB RAM (RSS)"

    # In Bảng Markdown So Sánh
    print("\n" + "=" * 80)
    print("BẢNG SO SÁNH TRỰC TIẾP HIỆU NĂNG ASR TRÊN 200 MẪU ĐẦU TIÊN (TASK 1)")
    print("=" * 80)
    table_md = f"""| Chỉ số đánh giá | PhoASR-whisper-small-ct2 (0.24B) | PhoWhisper-large-ct2 (1.54B) | Chênh lệch (Large vs Small) |
|---|:---:|:---:|:---:|
| **Tổng số tham số** | **241,734,912** (~0.24B) | **1,543,304,960** (~1.54B) | +1.30B (+538%) |
| **Raw WER** (giữ hoa/thường, dấu) | **{raw_wer_small:.2f}%** | **{raw_wer_large:.2f}%** | {raw_wer_large - raw_wer_small:+.2f}% |
| **Raw CER** (giữ hoa/thường, dấu) | **{raw_cer_small:.2f}%** | **{raw_cer_large:.2f}%** | {raw_cer_large - raw_cer_small:+.2f}% |
| **Normalized WER** (lower, no-punct) | **{norm_wer_small:.2f}%** | **{norm_wer_large:.2f}%** | {norm_wer_large - norm_wer_small:+.2f}% |
| **Normalized CER** (lower, no-punct) | **{norm_cer_small:.2f}%** | **{norm_cer_large:.2f}%** | {norm_cer_large - norm_cer_small:+.2f}% |
| **Tốc độ trung bình (ms/sample)** | **{lat_small:.1f} ms** | **{lat_large:.1f} ms** | {lat_large / max(lat_small, 1e-3):.2f}x chậm hơn |
| **Bộ nhớ tiêu thụ đỉnh điểm** | **{mem_small:.1f} {mem_unit}** | **{mem_large:.1f} {mem_unit}** | {mem_large - mem_small:+.1f} MB |"""

    print(table_md)
    print("=" * 80 + "\n")

    # Xuất file diff markdown
    export_diff_report(
        output_path=Path(args.output_diff),
        samples=samples,
        preds_small=preds_small,
        preds_large=preds_large,
        top_k=30,
    )


if __name__ == "__main__":
    main()
