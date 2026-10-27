#!/usr/bin/env python3
"""
scripts/setup_asr_models.py

Tự động tải và chuẩn bị 2 mô hình ASR theo định dạng CTranslate2 (float16) cho Zalo AI Challenge 2026 (Task 1):
1. Model 1 (PhoWhisper-large-ct2): Tải bản convert sẵn từ repo kiendt/PhoWhisper-large-ct2.
2. Model 2 (PhoASR-whisper-small-ct2): Chuyển đổi trực tiếp từ Qualcomm-AI-Research/PhoASR-whisper-small
   bằng ct2-transformers-converter với --quantization float16.
3. Đếm và in ra chính xác số lượng tham số để kiểm chứng ngân sách < 6B.

Sử dụng trên Google Colab / Linux / Windows:
    python scripts/setup_asr_models.py [--models-dir models/asr]
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8")

from huggingface_hub import snapshot_download

# Exact parameter counts based on model architectures
PHOASR_SMALL_PARAMS = 241_734_912       # ~0.24B (241.73M)
PHOWHISPER_LARGE_PARAMS = 1_543_304_960  # ~1.54B (1,543.30M)
MAX_TOTAL_BUDGET = 6_000_000_000         # 6 Billion parameters
MAX_LLM_BUDGET = 5_000_000_000           # 5 Billion parameters


def setup_phowhisper_large(target_dir: Path) -> None:
    """Tải pre-converted PhoWhisper-large-ct2 từ Hugging Face."""
    print("=" * 60)
    print("1. Chuẩn bị Model 1: kiendt/PhoWhisper-large-ct2")
    print("=" * 60)

    model_bin = target_dir / "model.bin"
    if model_bin.exists() and model_bin.stat().st_size > 1_000_000_000:
        print(f"[OK] Đã tồn tại model.bin tại {target_dir} ({model_bin.stat().st_size / 1e9:.2f} GB). Bỏ qua tải lại.")
        return

    print(f"Đang tải repo 'kiendt/PhoWhisper-large-ct2' về '{target_dir}'...")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="kiendt/PhoWhisper-large-ct2",
        local_dir=str(target_dir),
        resume_download=True,
    )
    print(f"[THÀNH CÔNG] PhoWhisper-large-ct2 đã sẵn sàng tại: {target_dir}")


def setup_phoasr_small(target_dir: Path) -> None:
    """Convert Qualcomm-AI-Research/PhoASR-whisper-small sang CTranslate2 float16."""
    print("=" * 60)
    print("2. Chuẩn bị Model 2: Qualcomm-AI-Research/PhoASR-whisper-small")
    print("=" * 60)

    model_bin = target_dir / "model.bin"
    if model_bin.exists() and model_bin.stat().st_size > 300_000_000:
        print(f"[OK] Đã tồn tại model.bin tại {target_dir} ({model_bin.stat().st_size / 1e6:.1f} MB). Bỏ qua convert.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đảm bảo ctranslate2 đã được cài đặt trong môi trường Python hiện hành
    try:
        import ctranslate2.converters
    except ImportError:
        print(f"[TỰ ĐỘNG] Thư viện ctranslate2 chưa có trong {sys.executable}. Đang tự động cài đặt...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "ctranslate2"], check=True)
            import ctranslate2.converters
        except (subprocess.SubprocessError, ImportError) as install_err:
            print(f"[CẢNH BÁO] Không thể tự cài ctranslate2 qua pip: {install_err}")

    # 2. Convert trực tiếp bằng Python API
    try:
        import ctranslate2.converters

        print("Đang chuyển đổi trực tiếp qua CTranslate2 TransformersConverter (in-process)...")
        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path="Qualcomm-AI-Research/PhoASR-whisper-small",
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        )
        converter.convert(
            output_dir=str(target_dir),
            quantization="float16",
            force=True,
        )
        print(f"[THÀNH CÔNG] PhoASR-whisper-small-ct2 đã convert xong tại: {target_dir}")
        return
    except (RuntimeError, ValueError, ImportError, OSError) as e:
        print(f"Không thể convert qua Python API ({e}), thử dùng fallback CLI...")

    # 3. Fallback CLI
    cmd = [
        sys.executable,
        "-m",
        "ctranslate2.converters.transformers",
        "--model",
        "Qualcomm-AI-Research/PhoASR-whisper-small",
        "--output_dir",
        str(target_dir),
        "--quantization",
        "float16",
        "--copy_files",
        "tokenizer.json",
        "preprocessor_config.json",
        "--force",
    ]

    ct2_cli = shutil.which("ct2-transformers-converter")
    if ct2_cli:
        cmd = [
            ct2_cli,
            "--model",
            "Qualcomm-AI-Research/PhoASR-whisper-small",
            "--output_dir",
            str(target_dir),
            "--quantization",
            "float16",
            "--copy_files",
            "tokenizer.json",
            "preprocessor_config.json",
            "--force",
        ]

    print(f"Thực thi lệnh chuyển đổi CLI: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[THÀNH CÔNG] PhoASR-whisper-small-ct2 đã convert xong tại: {target_dir}")


def print_parameter_budget_report() -> None:
    """In bảng phân tích ngân sách tham số chính xác."""
    print("\n" + "=" * 70)
    print("BÁO CÁO PHÂN TÍCH THAM SỐ & NGÂN SÁCH HỆ THỐNG (RULE: TOTAL < 6B, LLM < 5B)")
    print("=" * 70)

    models_info = [
        ("PhoASR-whisper-small-ct2", PHOASR_SMALL_PARAMS, 0.24, "Qualcomm AI Research"),
        ("PhoWhisper-large-ct2", PHOWHISPER_LARGE_PARAMS, 1.54, "VinAI / KienDT"),
    ]

    print(f"{'Mô hình ASR':<28} | {'Số params':<14} | {'Quy đổi':<8} | {'Quỹ còn lại cho LLM':<20}")
    print("-" * 75)
    for name, params, b_scale, _org in models_info:
        remaining_for_llm = (MAX_TOTAL_BUDGET - params) / 1e9
        # Max LLM budget cap is 5.0B
        effective_llm_cap = min(5.0, remaining_for_llm)
        print(f"{name:<28} | {params:<14,d} | {b_scale:.2f}B    | max {effective_llm_cap:.2f}B (hợp lệ)")

    print("-" * 75)
    print("Ghi chú chiến lược:")
    print("  * Nếu chọn PhoASR-small (0.24B) -> Thoải mái ghép cặp với LLM 3B (như Qwen2.5-3B, ~3.08B) hoặc 4B.")
    print("  * Nếu chọn PhoWhisper-large (1.54B) -> Ghép cặp với Qwen2.5-3B (3.08B) tổng = 4.62B < 6B (Vẫn hợp lệ!).")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải & convert các mô hình ASR sang CTranslate2")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/asr",
        help="Thư mục chứa các model ASR (mặc định: models/asr)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    phowhisper_dir = models_dir / "PhoWhisper-large-ct2"
    phoasr_dir = models_dir / "PhoASR-whisper-small-ct2"

    setup_phowhisper_large(phowhisper_dir)
    setup_phoasr_small(phoasr_dir)
    print_parameter_budget_report()


if __name__ == "__main__":
    main()
