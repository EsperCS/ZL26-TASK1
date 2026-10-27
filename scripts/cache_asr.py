#!/usr/bin/env python3
"""
scripts/cache_asr.py

Phiên âm trước (offline pre-transcription) và lưu cache toàn bộ audio cho Zalo AI Challenge 2026:
- Quét toàn bộ train.jsonl và public_test.jsonl.
- Thu thập cả audio chính (`item['audio']`) lẫn audio trong lịch sử (`turn['audio']` trong `item['history']`).
- Có cơ chế checkpoint định kỳ mỗi 100 file (tự động khôi phục nếu Colab bị ngắt kết nối).
- Lưu kết quả cuối cùng ra file parquet (và json) tại `data/cache/asr_cache.parquet`.

Sử dụng trên Google Colab / Linux / Windows:
    !python scripts/cache_asr.py --model-dir models/asr/PhoASR-whisper-small-ct2 --data-dir . --output data/cache/asr_cache.parquet
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

# Tự động phát hiện và cài đặt các thư viện phụ thuộc nếu còn thiếu trong môi trường Python
def _ensure_dependencies() -> None:
    import subprocess
    required = {
        "faster_whisper": "faster-whisper",
        "ctranslate2": "ctranslate2",
        "polars": "polars",
        "pyarrow": "pyarrow",
        "torch": "torch",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[TỰ ĐỘNG CÀI ĐẶT] Thiếu thư viện trong {sys.executable}. Đang cài đặt: {missing}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-U"] + missing, check=True)
        except (subprocess.SubprocessError, OSError):
            with contextlib.suppress(Exception):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-U", "--break-system-packages"] + missing,
                    check=True,
                )

_ensure_dependencies()

# Khắc phục tương thích PyAV 19 (bỏ metadata_errors)
with contextlib.suppress(Exception):
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open

import faster_whisper
import polars as pl
import torch


def collect_audio_paths(data_dir: Path) -> list[str]:
    """
    Quét tất cả các file JSONL (train.jsonl, public_test.jsonl) trong data_dir
    để trích xuất danh sách duy nhất các đường dẫn file audio.
    """
    candidate_jsonl_patterns = [
        "train.jsonl",
        "train/train.jsonl",
        "data/train/train.jsonl",
        "train+public_test/train/train.jsonl",
        "data/train+public_test/train/train.jsonl",
        "public_test.jsonl",
        "public_test/public_test.jsonl",
        "data/public_test/public_test.jsonl",
        "train+public_test/public_test/public_test.jsonl",
        "data/train+public_test/public_test/public_test.jsonl",
    ]

    found_files: list[Path] = []
    for pattern in candidate_jsonl_patterns:
        p = data_dir / pattern
        if p.exists() and p not in found_files:
            found_files.append(p)

    # Quét đệ quy nếu chưa thấy
    if not found_files:
        for p in data_dir.glob("**/*.jsonl"):
            if "split" not in p.name and "val" not in p.name:
                found_files.append(p)

    print(f"Tìm thấy {len(found_files)} file dữ liệu JSONL:")
    for f in found_files:
        print(f"  - {f}")

    unique_audio_paths: set[str] = set()

    for jsonl_file in found_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 1. Audio chính
                main_audio = item.get("audio")
                if main_audio and isinstance(main_audio, str):
                    unique_audio_paths.add(main_audio)

                # 2. Audio trong history
                history = item.get("history", [])
                if isinstance(history, list):
                    for turn in history:
                        if isinstance(turn, dict):
                            h_audio = turn.get("audio")
                            if h_audio and isinstance(h_audio, str):
                                unique_audio_paths.add(h_audio)

    sorted_paths = sorted(unique_audio_paths)
    print(f"Tổng số file audio duy nhất cần phiên âm: {len(sorted_paths)}")
    return sorted_paths


def find_audio_file(audio_rel_path: str, search_roots: list[Path]) -> Path | None:
    """Xác định vị trí tồn tại thực tế của file audio trên đĩa."""
    p = Path(audio_rel_path)
    if p.exists():
        return p

    for root in search_roots:
        cand1 = root / audio_rel_path
        if cand1.exists():
            return cand1
        # Bỏ prefix "audio/" nếu root đã ở trong folder train hoặc audio
        if audio_rel_path.startswith("audio/"):
            cand2 = root / audio_rel_path[6:]
            if cand2.exists():
                return cand2
            cand3 = root / "audio" / audio_rel_path[6:]
            if cand3.exists():
                return cand3

    return None


def save_cache_files(cache_data: dict[str, dict[str, Any]], output_parquet: Path, checkpoint_json: Path) -> None:
    """Lưu dữ liệu cache ra cả Parquet và Checkpoint JSON."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # 1. Lưu checkpoint JSON
    with open(checkpoint_json, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    # 2. Lưu Parquet bằng Polars
    rows = []
    for rel_path, meta in cache_data.items():
        rows.append({
            "audio_path": rel_path,
            "transcription": meta.get("transcription", ""),
            "duration_sec": float(meta.get("duration_sec", 0.0)),
            "model_name": meta.get("model_name", ""),
        })

    if rows:
        df = pl.DataFrame(rows)
        df.write_parquet(output_parquet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo ASR Cache cho toàn bộ tập dữ liệu Task 1")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/asr/PhoASR-whisper-small-ct2",
        help="Thư mục chứa model ASR CTranslate2 (mặc định: models/asr/PhoASR-whisper-small-ct2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Thư mục gốc chứa dữ liệu train và public_test (mặc định: .)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cache/asr_cache.parquet",
        help="File đích xuất cache parquet (mặc định: data/cache/asr_cache.parquet)",
    )
    parser.add_argument(
        "--batch-save-every",
        type=int,
        default=100,
        help="Tần suất lưu checkpoint (mỗi N file)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số file cần xử lý (0: tất cả)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy (cuda hoặc cpu)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16" if torch.cuda.is_available() else "int8",
        help="Định dạng tính toán (float16 trên GPU, int8 trên CPU)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_parquet = Path(args.output)
    checkpoint_json = output_parquet.with_suffix(".checkpoint.json")

    # 1. Đọc checkpoint cũ nếu có
    cache_data: dict[str, dict[str, Any]] = {}
    if checkpoint_json.exists():
        try:
            with open(checkpoint_json, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print(f"[RECOVERY] Đã nạp {len(cache_data)} mẫu từ checkpoint: {checkpoint_json}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[CẢNH BÁO] Không thể đọc checkpoint cũ: {e}. Tạo mới.")

    # 2. Thu thập danh sách audio
    all_audio_paths = collect_audio_paths(data_dir)
    pending_audios = [p for p in all_audio_paths if p not in cache_data]

    if args.limit > 0:
        pending_audios = pending_audios[: args.limit]

    print(f"Tổng số: {len(all_audio_paths)} | Đã xong: {len(cache_data)} | Cần xử lý: {len(pending_audios)}")
    if not pending_audios:
        print("[HOÀN TẤT] Toàn bộ file audio đã có trong cache!")
        save_cache_files(cache_data, output_parquet, checkpoint_json)
        return

    # 3. Nạp mô hình Faster-Whisper
    model_path = Path(args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model directory: {model_path}")

    print(f"Đang nạp mô hình từ {model_path} (Device: {args.device}, Compute: {args.compute_type})...")
    model = faster_whisper.WhisperModel(
        str(model_path),
        device=args.device,
        compute_type=args.compute_type,
    )

    search_roots = [
        data_dir,
        data_dir / "train",
        data_dir / "public_test",
        data_dir / "train+public_test",
        data_dir / "train+public_test" / "train",
        data_dir / "train+public_test" / "public_test",
        Path("train"),
        Path("public_test"),
        Path("train+public_test"),
        Path("train+public_test/train"),
        Path("train+public_test/public_test"),
        Path("data/train"),
        Path("data/public_test"),
    ]

    # 4. Phiên âm từng file và cập nhật checkpoint
    t_start = time.perf_counter()
    processed_count = 0

    print("\n--- BẮT ĐẦU TIẾN TRÌNH PHIÊN ÂM VÀ LƯU CACHE ---")
    for idx, rel_path in enumerate(pending_audios, 1):
        actual_path = find_audio_file(rel_path, search_roots)
        if actual_path is None or not actual_path.exists():
            print(f"  [BỎ QUA] Không tìm thấy file: {rel_path}")
            cache_data[rel_path] = {
                "transcription": "",
                "duration_sec": 0.0,
                "model_name": model_path.name,
                "error": "File not found",
            }
            continue

        try:
            segments, info = model.transcribe(
                str(actual_path),
                vad_filter=True,
                beam_size=5,
                temperature=0.0,
                language="vi",
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            duration = float(getattr(info, "duration", 0.0))

            cache_data[rel_path] = {
                "transcription": text,
                "duration_sec": duration,
                "model_name": model_path.name,
            }
            processed_count += 1

            if processed_count % 10 == 0 or idx == len(pending_audios):
                elapsed = time.perf_counter() - t_start
                rate = processed_count / max(elapsed, 1e-3)
                print(f"[{processed_count}/{len(pending_audios)}] Đã xử lý: {rel_path} ({rate:.2f} file/s)")

            # Checkpoint định kỳ
            if processed_count % args.batch_save_every == 0:
                print(f"  [CHECKPOINT] Đang lưu checkpoint ({len(cache_data)} files)...")
                save_cache_files(cache_data, output_parquet, checkpoint_json)

        except (RuntimeError, ValueError, OSError) as e:
            print(f"  [LỖI] Xử lý file {rel_path}: {e}")
            cache_data[rel_path] = {
                "transcription": "",
                "duration_sec": 0.0,
                "model_name": model_path.name,
                "error": str(e),
            }

    # Lưu kết quả cuối cùng
    save_cache_files(cache_data, output_parquet, checkpoint_json)
    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"[HOÀN THÀNH TOÀN BỘ] Đã lưu cache tại: {output_parquet}")
    print(f"Tổng file đã xử lý đợt này : {processed_count}")
    print(f"Tổng kích thước cache      : {len(cache_data)} mẫu")
    print(f"Tổng thời gian chạy        : {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
