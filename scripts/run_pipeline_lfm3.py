#!/usr/bin/env python3
"""scripts/run_pipeline_lfm3.py.

Pipeline hoàn thiện và thực thi chính thức cho Task 1 (LifeMate) với mô hình `LiquidAI/LFM3-1.7B`.
Chạy trực tiếp từ ASR/Transcript -> ReAct Agent tương tác ToolEnv -> Sinh submission.jsonl.

Sử dụng:
  # Chạy thử 10 mẫu kiểm tra nhanh (Smoke Test) trên CPU hoặc GPU:
  python scripts/run_pipeline_lfm3.py --split train --input-file train/val_mini.jsonl --limit 10

  # Chạy thực tế với mô hình thật trên GPU (Colab / Server):
  python scripts/run_pipeline_lfm3.py --split train --input-file train/val_split.jsonl --real-model --device cuda

  # Sinh file nộp bài cho Public Test:
  python scripts/run_pipeline_lfm3.py --split public_test --input-file public_test/public_test.jsonl --real-model --output-file submission.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Đảm bảo UTF-8 cho console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.voice_agent import LFM3ModelRunner, LFM3VoiceAgent


def load_asr_cache(parquet_path: Path) -> dict[str, str]:
    """Nạp cache phiên âm thực tế từ ZipFormer-150M nếu có."""
    if not parquet_path.is_file():
        return {}
    try:
        import polars as pl
        df = pl.read_parquet(parquet_path)
        return dict(zip(df["audio_path"].to_list(), df["transcription"].to_list(), strict=False))
    except Exception as e:  # noqa: BLE001
        print(f"[!] Cảnh báo không đọc được ASR cache ({e}).")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Pipeline Hoàn Thiện Task 1 với LFM3-1.7B.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "public_test"], help="Phân vùng dữ liệu")
    parser.add_argument("--data-root", type=str, default=".", help="Thư mục gốc chứa dữ liệu")
    parser.add_argument("--input-file", type=str, default=None, help="Đường dẫn file .jsonl đầu vào")
    parser.add_argument("--catalog", type=str, default=None, help="Đường dẫn file tools catalog JSON")
    parser.add_argument("--output-file", type=str, default="submission.jsonl", help="Đường dẫn file kết quả .jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu thực thi")
    parser.add_argument("--offset", type=int, default=0, help="Vị trí mẫu bắt đầu")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: cuda hoặc cpu")
    parser.add_argument("--model-id", type=str, default="LiquidAI/LFM3-1.7B", help="ID mô hình Hugging Face")
    parser.add_argument("--real-model", action="store_true", help="Bật nạp mô hình thật trên GPU (thay vì mock)")
    parser.add_argument("--asr-cache", type=str, default="data/cache/asr_cache.parquet", help="Đường dẫn ASR cache parquet")
    parser.add_argument("--max-react-steps", type=int, default=4, help="Số bước ReAct tối đa cho mỗi mẫu")

    args = parser.parse_args()

    root_path = Path(args.data_root)
    split = args.split

    # Xác định đường dẫn file input
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_file() and (root_path / "train+public_test" / args.input_file).is_file():
            input_path = root_path / "train+public_test" / args.input_file
    else:
        candidates = [
            root_path / split / f"{split}.jsonl",
            root_path / "train+public_test" / split / f"{split}.jsonl",
        ]
        input_path = next((p for p in candidates if p.is_file()), candidates[0])

    if not input_path.is_file():
        print(f"[x] Lỗi: Không tìm thấy file đầu vào tại {input_path}")
        sys.exit(1)

    # Xác định catalog
    if args.catalog:
        catalog_path = Path(args.catalog)
        if not catalog_path.is_file() and (root_path / "train+public_test" / args.catalog).is_file():
            catalog_path = root_path / "train+public_test" / args.catalog
    else:
        candidates = [
            root_path / split / f"{split}_tools_catalog.json",
            root_path / "train+public_test" / split / f"{split}_tools_catalog.json",
        ]
        catalog_path = next((p for p in candidates if p.is_file()), candidates[0])

    if not catalog_path.is_file():
        print(f"[x] Lỗi: Không tìm thấy tools catalog tại {catalog_path}")
        sys.exit(1)

    with open(catalog_path, encoding="utf-8") as f:
        tools_catalog = json.load(f)

    # Nạp ASR cache nếu có
    asr_cache = load_asr_cache(Path(args.asr_cache))
    if asr_cache:
        print(f"[✓] Đã nạp {len(asr_cache)} bản ghi phiên âm ASR từ {args.asr_cache}")

    # Đọc danh sách mẫu
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                samples.append(json.loads(line_str))

    if args.offset > 0:
        samples = samples[args.offset :]
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    total_samples = len(samples)
    print("================================================================================")
    print(" ZALO AI CHALLENGE 2026 - LIFEMATE PIPELINE RUNNER (LFM3-1.7B)")
    print(f" Split: {split} | Tổng số mẫu: {total_samples}")
    print(f" Catalog: {catalog_path.name} | Input: {input_path.name}")
    print(f" Chế độ: {'MÔ HÌNH THẬT GPU' if args.real_model else 'FALLBACK / TEST LOCAL'}")
    print(f" File nộp bài xuất ra: {args.output_file}")
    print("================================================================================")

    # Khởi tạo Runner mô hình
    runner = None
    if args.real_model:
        runner = LFM3ModelRunner(model_id=args.model_id, device=args.device)

    # Khởi tạo Agent
    agent = LFM3VoiceAgent(
        tools_catalog=tools_catalog,
        runner=runner,
        max_react_steps=args.max_react_steps,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    match_count = 0
    total_evaluated = 0
    t_start = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples, 1):
            s_id = sample.get("id", f"sample_{idx}")
            audio_path = sample.get("audio", "")

            # Ưu tiên lấy từ ASR cache -> nếu không có lấy audio_text -> rỗng
            transcript = asr_cache.get(audio_path, sample.get("audio_text", ""))

            # Thực thi ReAct Agent tương tác trực tiếp ToolEnv
            res = agent.run_sample(
                split=split,
                sample=sample,
                transcript_override=transcript,
                root=root_path,
            )

            # Ghi dòng submission.jsonl chuẩn quy cách BTC
            record = {
                "id": res.sample_id,
                "tool_calls": res.tool_calls,
                "state_hash": res.state_hash,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Đối chiếu kết quả nếu sample có nhãn (như tập train/val)
            gold_label = sample.get("label")
            status_str = ""
            if gold_label:
                total_evaluated += 1
                call_match = (len(res.tool_calls) == len(gold_label.get("tool_calls", [])))
                if call_match:
                    match_count += 1
                    status_str = f" | Calls Match: [✓] ({len(res.tool_calls)})"
                else:
                    status_str = f" | Calls: {len(res.tool_calls)} vs Gold: {len(gold_label.get('tool_calls', []))}"

            print(
                f"[{idx:3d}/{total_samples:3d}] Sample: {s_id} | Steps: {res.steps_taken} | "
                f"Retries: {res.validation_retries} | Hash: {res.state_hash[:12]}... | "
                f"Lat: {res.execution_time_ms:.0f}ms{status_str}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start
    print("================================================================================")
    print(f"[✓] ĐÃ HOÀN TẤT PIPELINE {total_samples} MẪU TRONG {total_time:.2f} GIÂY!")
    print(f"[*] File nộp bài đã được ghi an toàn tại: {output_path.resolve()}")
    if total_evaluated > 0:
        print(f"[*] Tỷ lệ khớp số lượng tool: {match_count}/{total_evaluated} ({match_count/total_evaluated*100:.1f}%)")
    print("================================================================================")


if __name__ == "__main__":
    main()
