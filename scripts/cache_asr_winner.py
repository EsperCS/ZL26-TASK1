#!/usr/bin/env python3
"""scripts/cache_asr_winner.py.

Đóng băng & Cache toàn bộ kết quả phiên âm ASR của kiến trúc hạng 1:
Silero VAD + ZipFormer-150M + 418 Hotwords + XLM-Capu-FT + Vietnamese ITN (~0.433B params).

Chức năng:
1. Quét toàn bộ `train.jsonl` và `public_test.jsonl`.
2. Thu thập cả audio chính (`item['audio']`) và toàn bộ audio lịch sử (`turn['audio']` trong `item['history']`).
3. Khử trùng lặp đường dẫn audio.
4. Cơ chế checkpoint tự động định kỳ mỗi 100 mẫu (hỗ trợ phục hồi khi gián đoạn trên Colab).
5. Xuất kết quả ra `data/cache/asr_cache.parquet`.

Sử dụng:
    # Trên Colab / GPU Server:
    python scripts/cache_asr_winner.py --device cuda --checkpoint-every 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Đảm bảo UTF-8 trên Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr.sherpa_xlm_pipeline import SherpaXLMPipeline

WINNER_MODEL_NAME = "ZipFormer-150M + 418 Hotwords + XLM-Capu-FT + ITN"
DEFAULT_CHECKPOINT_INTERVAL = 100


def collect_unique_audio_tasks(
    data_dir: Path,
) -> list[dict[str, str]]:
    """Quét và gom toàn bộ audio chính và audio history từ train và public_test."""
    manifest_sources = [
        # 1. Thư mục cục bộ chuẩn
        ("train", data_dir / "train" / "train.jsonl", data_dir / "train"),
        ("public_test", data_dir / "public_test" / "public_test.jsonl", data_dir / "public_test"),
        # 2. Thư mục ghép train+public_test
        (
            "train",
            data_dir / "train+public_test" / "train" / "train.jsonl",
            data_dir / "train+public_test" / "train",
        ),
        (
            "public_test",
            data_dir / "train+public_test" / "public_test" / "public_test.jsonl",
            data_dir / "train+public_test" / "public_test",
        ),
        # 3. Thư mục data/task1_lifemate
        (
            "train",
            data_dir / "data" / "task1_lifemate" / "train+public_test" / "train" / "train.jsonl",
            data_dir / "data" / "task1_lifemate" / "train+public_test" / "train",
        ),
        (
            "public_test",
            data_dir / "data" / "task1_lifemate" / "train+public_test" / "public_test" / "public_test.jsonl",
            data_dir / "data" / "task1_lifemate" / "train+public_test" / "public_test",
        ),
        # 4. Các vị trí phổ biến trên Google Colab
        (
            "train",
            Path("/content/train+public_test/train/train.jsonl"),
            Path("/content/train+public_test/train"),
        ),
        (
            "public_test",
            Path("/content/train+public_test/public_test/public_test.jsonl"),
            Path("/content/train+public_test/public_test"),
        ),
        (
            "train",
            Path("/content/data/task1_lifemate/train+public_test/train/train.jsonl"),
            Path("/content/data/task1_lifemate/train+public_test/train"),
        ),
        (
            "public_test",
            Path("/content/data/task1_lifemate/train+public_test/public_test/public_test.jsonl"),
            Path("/content/data/task1_lifemate/train+public_test/public_test"),
        ),
        (
            "train",
            Path("/content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test/train/train.jsonl"),
            Path("/content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test/train"),
        ),
        (
            "public_test",
            Path("/content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test/public_test/public_test.jsonl"),
            Path("/content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test/public_test"),
        ),
    ]

    seen_rel_paths: set[str] = set()
    audio_tasks: list[dict[str, str]] = []

    for split_name, manifest_path, audio_root in manifest_sources:
        if not manifest_path.is_file():
            continue

        print(f"[*] Đang quét manifest: {manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_id = record.get("id", "")

                # 1. Audio chính
                main_audio = record.get("audio")
                if main_audio and main_audio not in seen_rel_paths:
                    seen_rel_paths.add(main_audio)
                    full_p = (audio_root / main_audio).resolve()
                    audio_tasks.append({
                        "sample_id": sample_id,
                        "split": split_name,
                        "type": "primary",
                        "audio_path": main_audio,
                        "full_path": str(full_p),
                    })

                # 2. Audio trong history
                for idx, turn in enumerate(record.get("history", [])):
                    hist_audio = turn.get("audio")
                    if hist_audio and hist_audio not in seen_rel_paths:
                        seen_rel_paths.add(hist_audio)
                        full_p = (audio_root / hist_audio).resolve()
                        audio_tasks.append({
                            "sample_id": sample_id,
                            "split": split_name,
                            "type": f"history_{idx}",
                            "audio_path": hist_audio,
                            "full_path": str(full_p),
                        })

    return audio_tasks


def load_existing_cache(
    parquet_path: Path,
    checkpoint_json: Path,
) -> dict[str, dict[str, Any]]:
    """Nạp cache đã có từ file parquet hoặc checkpoint json để resume."""
    cache: dict[str, dict[str, Any]] = {}

    # Ưu tiên checkpoint json gần nhất nếu đang dở dang
    if checkpoint_json.is_file():
        try:
            with open(checkpoint_json, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            v.setdefault("audio_path", k)
                            cache[k] = v
                elif isinstance(data, list):
                    for item in data:
                        cache[item["audio_path"]] = item
            print(f"[+] Đã khôi phục {len(cache)} mẫu từ checkpoint: {checkpoint_json}")
            return cache
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[!] Lỗi đọc checkpoint json ({e}), chuyển sang đọc parquet...")

    if parquet_path.is_file():
        try:
            df = pl.read_parquet(parquet_path)
            for row in df.iter_rows(named=True):
                cache[row["audio_path"]] = row
            print(f"[+] Đã nạp {len(cache)} mẫu từ file parquet có sẵn: {parquet_path}")
        except (OSError, ValueError, KeyError) as e:
            print(f"[!] Không thể đọc parquet ({e}), tạo cache mới từ đầu.")

    return cache


def save_checkpoint(
    records: list[dict[str, Any]],
    parquet_path: Path,
    checkpoint_json: Path,
) -> None:
    """Ghi checkpoint ra cả json và parquet an toàn bằng cơ chế atomic write."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ghi checkpoint json tạm rồi đổi tên
    tmp_json = checkpoint_json.with_suffix(".tmp.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp_json.replace(checkpoint_json)

    # 2. Ghi parquet
    if records:
        df = pl.DataFrame(records)
        tmp_parquet = parquet_path.with_suffix(".tmp.parquet")
        df.write_parquet(tmp_parquet)
        tmp_parquet.replace(parquet_path)


def run_cache_asr(
    data_dir: Path,
    output_parquet: Path,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    device: str = "cuda",
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    """Thực thi pipeline cache toàn bộ ASR với kiểm soát tiến trình chặt chẽ."""
    output_parquet = output_parquet.resolve()
    checkpoint_json = output_parquet.parent / "asr_cache.checkpoint.json"

    print("================================================================================")
    print(" ZALO AI CHALLENGE 2026 - ASR WINNER CACHE GENERATOR (TASK 1)")
    print(f" Model Pipeline: {WINNER_MODEL_NAME}")
    print(f" Output Target : {output_parquet}")
    print("================================================================================")

    # 1. Gom danh sách task
    tasks = collect_unique_audio_tasks(data_dir)
    print(f"[*] Tổng số file audio cần xử lý (Unique): {len(tasks)}")

    if not tasks:
        print("\n[!] LỖI: Không tìm thấy bất kỳ file audio nào từ các manifest JSONL!")
        print("    Nguyên nhân: Thư mục `train/` hoặc `public_test/` (chứa train.jsonl và audio/) chưa có mặt.")
        print("    Các file âm thanh nặng nằm trong .gitignore nên khi git clone sẽ không có sẵn.")
        print("    Hướng dẫn khắc phục trên Google Colab:")
        print("    1. Nếu bạn có file `train+public_test.zip` trên Google Drive, hãy giải nén bằng cell:")
        print("       !unzip -q /content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test.zip -d /content/zaloai-challenge-2026/")
        print("    2. Hoặc trỏ tham số `--data-dir /content/drive/MyDrive/...` đến đúng nơi lưu dataset.")
        return

    # 2. Nạp cache cũ
    cache = load_existing_cache(output_parquet, checkpoint_json)
    remaining_tasks = [t for t in tasks if t["audio_path"] not in cache]
    print(f"[*] Số file đã có trong cache: {len(cache)} | Số file còn lại cần phiên âm: {len(remaining_tasks)}")

    if limit is not None and limit > 0:
        remaining_tasks = remaining_tasks[:limit]
        print(f"[*] Áp dụng giới hạn --limit: chỉ xử lý {len(remaining_tasks)} file")

    if dry_run:
        print("[!] Chế độ DRY-RUN: Hoàn tất quét task, không tải mô hình và không phiên âm.")
        return

    if not remaining_tasks:
        print(f"[✓] Toàn bộ {len(tasks)} audio đã được phiên âm đầy đủ trong cache! Xuất file parquet cuối cùng...")
        save_checkpoint(list(cache.values()), output_parquet, checkpoint_json)
        print(f"[✓] Thành công: {output_parquet} ({len(cache)} rows)")
        return

    # 3. Khởi tạo Pipeline ASR Winner
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] Cảnh báo: Không có CUDA GPU, chuyển về CPU.")
        device = "cpu"

    hotwords_path = REPO_ROOT / "models" / "asr" / "zalo_hotwords.txt"
    capu_path = REPO_ROOT / "models" / "asr" / "xlm-roberta-capu-zalo-ft"

    print(f"[*] Đang khởi tạo SherpaXLMPipeline (Transducer: hynt/ZipFormer-150M, Device: {device})...")
    pipeline = SherpaXLMPipeline(
        transducer_model="hynt/ZipFormer-150M",
        hotwords_file=str(hotwords_path) if hotwords_path.is_file() else None,
        capu_model_dir=str(capu_path) if capu_path.is_dir() else None,
        hotwords_score=1.5,
        device=device,
    )

    records_list = list(cache.values())
    processed_count = 0
    t_start = time.perf_counter()

    for idx, task in enumerate(remaining_tasks, 1):
        full_audio_path = Path(task["full_path"])
        rel_audio_path = task["audio_path"]

        if not full_audio_path.is_file():
            print(f"[!] Bỏ qua (không tìm thấy file): {full_audio_path}")
            continue

        try:
            t0 = time.perf_counter()
            res = pipeline.transcribe(str(full_audio_path))
            dur = time.perf_counter() - t0

            entry = {
                "audio_path": rel_audio_path,
                "transcription": res.cleaned_text,
                "raw_acoustic_text": res.raw_acoustic_text,
                "duration_sec": round(res.duration_sec, 3),
                "inference_time_sec": round(dur, 4),
                "rtf": round(dur / max(res.duration_sec, 0.01), 4),
                "model_name": WINNER_MODEL_NAME,
                "created_at": datetime.now(UTC).isoformat(),
            }
            cache[rel_audio_path] = entry
            records_list.append(entry)
            processed_count += 1

            if processed_count % 10 == 0 or idx == len(remaining_tasks):
                elapsed = time.perf_counter() - t_start
                avg_time = elapsed / processed_count
                rem_time = (len(remaining_tasks) - processed_count) * avg_time
                print(
                    f"[{idx}/{len(remaining_tasks)}] Audio: {rel_audio_path} | "
                    f"Trans: \"{res.cleaned_text[:50]}...\" | "
                    f"RTF: {dur / max(res.duration_sec, 0.01):.3f}x | "
                    f"ETA: {rem_time / 60:.1f}m"
                )

            # Checkpoint định kỳ
            if processed_count % checkpoint_interval == 0:
                print(f"[+] Lưu checkpoint định kỳ ({len(records_list)} mẫu)...")
                save_checkpoint(records_list, output_parquet, checkpoint_json)

        except (OSError, RuntimeError, ValueError) as e:
            print(f"[!] Lỗi phiên âm file {full_audio_path}: {e}")

    # 4. Lưu kết quả cuối cùng
    save_checkpoint(records_list, output_parquet, checkpoint_json)
    print("================================================================================")
    print(f"[✓] HOÀN TẤT CACHE TOÀN BỘ ASR: {len(records_list)} file đã được lưu vào {output_parquet}")
    print("================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đóng băng và cache toàn bộ kết quả phiên âm ZipFormer-150M + XLM-Capu-FT cho Zalo AI Challenge 2026."
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT, help="Thư mục gốc chứa train và public_test.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "asr_cache.parquet",
        help="Đường dẫn file parquet đầu ra.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="Khoảng cách mẫu ghi checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Thiết bị chạy ASR.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu xử lý (dùng để test).")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ quét thống kê danh sách file, không chạy inference.")

    args = parser.parse_args()
    run_cache_asr(
        data_dir=args.data_dir,
        output_parquet=args.output,
        checkpoint_interval=args.checkpoint_every,
        device=args.device,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
