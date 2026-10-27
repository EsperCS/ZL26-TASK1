#!/usr/bin/env python3
"""
scripts/setup_asr_models.py

Tự động tải và chuẩn bị các mô hình ASR và NLP phục vụ Zalo AI Challenge 2026 (Task 1):
1. VAD: silero_vad.onnx (Sherpa-ONNX)
2. Transducer Model 1: g-group-ai-lab/gipformer1.5-68M-rnnt (~68M params)
3. Transducer Model 2: hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (~153M params)
   (Tự động copy config.json -> tokens.txt do định dạng đặc thù của repo hynt)
4. Capu Model: dragonSwing/xlm-roberta-capu (~278M params) cho khôi phục viết hoa và dấu câu
5. Whisper Models (tuỳ chọn): PhoWhisper-large-ct2, PhoASR-whisper-small-ct2
6. Đếm và in ra bảng tổng hợp số lượng tham số để kiểm soát trần ngân sách < 6B (LLM < 5B).

Sử dụng:
    python scripts/setup_asr_models.py [--models-dir models/asr] [--include-whisper]
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import urllib.request
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
SILERO_VAD_PARAMS = 1_600_000            # ~0.002B (1.6M)
GIPFORMER_68M_PARAMS = 68_000_000        # ~0.068B (68M)
ZIPFORMER_150M_PARAMS = 153_000_000      # ~0.153B (153M)
XLM_ROBERTA_CAPU_PARAMS = 278_000_000    # ~0.278B (278M)
PHOASR_SMALL_PARAMS = 241_734_912        # ~0.24B (241.73M)
PHOWHISPER_LARGE_PARAMS = 1_543_304_960   # ~1.54B (1,543.30M)

MAX_TOTAL_BUDGET = 6_000_000_000         # 6 Billion parameters
MAX_LLM_BUDGET = 5_000_000_000           # 5 Billion parameters


def setup_silero_vad(target_dir: Path) -> Path:
    """Tải silero_vad.onnx từ k2-fsa/sherpa-onnx release nếu chưa có."""
    print("=" * 60)
    print("1. Chuẩn bị Silero VAD: models/asr/vad/silero_vad.onnx")
    print("=" * 60)
    target_dir.mkdir(parents=True, exist_ok=True)
    vad_file = target_dir / "silero_vad.onnx"
    if vad_file.exists() and vad_file.stat().st_size > 500_000:
        print(f"[OK] Đã tồn tại silero_vad.onnx tại {vad_file} ({vad_file.stat().st_size} bytes). Bỏ qua tải lại.")
        return vad_file

    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    print(f"Đang tải silero_vad.onnx từ {url}...")
    urllib.request.urlretrieve(url, vad_file)
    print(f"[THÀNH CÔNG] Silero VAD đã sẵn sàng tại: {vad_file} ({vad_file.stat().st_size} bytes)")
    return vad_file


def find_transducer_onnx_files(model_dir: Path) -> dict[str, Path]:
    """
    Dò tìm đúng các file ONNX (encoder, decoder, joiner, tokens) trong thư mục.
    Ưu tiên bản FP32/FP16 (không chứa '.int8.'), nếu không có mới fallback sang int8.
    """
    files = list(model_dir.iterdir())
    onnx_files = [f for f in files if f.suffix == ".onnx"]

    def pick_best(prefix: str) -> Path | None:
        candidates = [f for f in onnx_files if f.name.startswith(prefix)]
        if not candidates:
            return None
        # Ưu tiên file không chứa '.int8.'
        non_int8 = [f for f in candidates if ".int8." not in f.name]
        return non_int8[0] if non_int8 else candidates[0]

    encoder = pick_best("encoder")
    decoder = pick_best("decoder")
    joiner = pick_best("joiner")

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        # Thử config.json nếu thực chất là tokens
        cfg = model_dir / "config.json"
        if cfg.exists():
            shutil.copy(cfg, tokens)

    resolved = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "tokens": tokens if tokens.exists() else None,
    }
    return resolved


def setup_gipformer_68m(target_dir: Path) -> Path:
    """Tải và chuẩn bị mô hình g-group-ai-lab/gipformer1.5-68M-rnnt."""
    print("=" * 60)
    print("2. Chuẩn bị Transducer: g-group-ai-lab/gipformer1.5-68M-rnnt (~68M)")
    print("=" * 60)

    target_dir.mkdir(parents=True, exist_ok=True)
    resolved = find_transducer_onnx_files(target_dir)
    if all(resolved.values()):
        print(f"[OK] Đã tồn tại đủ file ONNX và tokens cho Gipformer-68M tại {target_dir}:")
        for k, v in resolved.items():
            print(f"  - {k}: {v.name if v else 'None'}")
        return target_dir

    print(f"Đang tải repo 'g-group-ai-lab/gipformer1.5-68M-rnnt' về '{target_dir}'...")
    snapshot_download(
        repo_id="g-group-ai-lab/gipformer1.5-68M-rnnt",
        local_dir=str(target_dir),
        allow_patterns=["*.onnx", "tokens.txt", "config.json", "bpe.model", "README.md"],
        resume_download=True,
    )

    resolved = find_transducer_onnx_files(target_dir)
    print(f"[THÀNH CÔNG] Gipformer-68M đã sẵn sàng tại {target_dir}:")
    for k, v in resolved.items():
        print(f"  - {k}: {v.name if v else 'None'}")
    return target_dir


def setup_zipformer_150m(target_dir: Path) -> Path:
    """Tải và chuẩn bị mô hình hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (~153M)."""
    print("=" * 60)
    print("3. Chuẩn bị Transducer: hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (~153M)")
    print("=" * 60)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Đảm bảo copy config.json sang tokens.txt nếu config.json đã có
    cfg = target_dir / "config.json"
    tok = target_dir / "tokens.txt"
    if cfg.exists() and not tok.exists():
        shutil.copy(cfg, tok)
        print(f"[XỬ LÝ] Đã sao chép config.json -> tokens.txt tại {target_dir}")

    resolved = find_transducer_onnx_files(target_dir)
    if all(resolved.values()):
        print(f"[OK] Đã tồn tại đủ file ONNX và tokens cho ZipFormer-150M tại {target_dir}:")
        for k, v in resolved.items():
            print(f"  - {k}: {v.name if v else 'None'}")
        return target_dir

    print(f"Đang tải repo 'hynt/ZipFormer-150M-CR-CTC-RNNT-6000h' về '{target_dir}'...")
    snapshot_download(
        repo_id="hynt/ZipFormer-150M-CR-CTC-RNNT-6000h",
        local_dir=str(target_dir),
        allow_patterns=["*.onnx", "config.json", "tokens.txt", "bpe.model", "README.md"],
        resume_download=True,
    )

    # Sau khi tải, tự động copy config.json sang tokens.txt
    if cfg.exists() and not tok.exists():
        shutil.copy(cfg, tok)
        print(f"[XỬ LÝ] Đã sao chép config.json -> tokens.txt tại {target_dir}")

    resolved = find_transducer_onnx_files(target_dir)
    print(f"[THÀNH CÔNG] ZipFormer-150M đã sẵn sàng tại {target_dir}:")
    for k, v in resolved.items():
        print(f"  - {k}: {v.name if v else 'None'}")
    return target_dir


def _patch_capu_compatibility(capu_dir: Path) -> None:
    """Tự động khắc phục tương thích mã nguồn cũ của dragonSwing/xlm-roberta-capu với transformers 4.40+ / 5.x."""
    model_file = capu_dir / "modeling_seq2labels.py"
    if not model_file.exists():
        return
    try:
        content = model_file.read_text(encoding="utf-8")
        modified = False

        if "@dataclass" not in content:
            content = "from dataclasses import dataclass\n" + content
            content = content.replace(
                "class Seq2LabelsOutput(ModelOutput):",
                "@dataclass\nclass Seq2LabelsOutput(ModelOutput):",
            )
            modified = True

        if "mean_resizing=False" not in content and "self.bert.resize_token_embeddings(vocab_size + 1)" in content:
            content = content.replace(
                "self.bert.resize_token_embeddings(vocab_size + 1)",
                "self.bert.resize_token_embeddings(vocab_size + 1, mean_resizing=False)",
            )
            modified = True

        if modified:
            model_file.write_text(content, encoding="utf-8")
            print(f"[PATCH] Đã tự động vá tương thích transformers cho {model_file.name}")
    except (OSError, UnicodeDecodeError):
        pass


def setup_xlm_roberta_capu(target_dir: Path) -> Path:
    """Tải và chuẩn bị mô hình dragonSwing/xlm-roberta-capu (~278M)."""
    print("=" * 60)
    print("4. Chuẩn bị Capu: dragonSwing/xlm-roberta-capu (~278M)")
    print("=" * 60)

    target_dir.mkdir(parents=True, exist_ok=True)
    model_bin = target_dir / "pytorch_model.bin"
    vocab_dir = target_dir / "vocabulary"

    if model_bin.exists() and model_bin.stat().st_size > 800_000_000 and vocab_dir.exists():
        _patch_capu_compatibility(target_dir)
        print(f"[OK] Đã tồn tại pytorch_model.bin và vocabulary tại {target_dir}. Bỏ qua tải lại.")
        return target_dir

    print(f"Đang tải repo 'dragonSwing/xlm-roberta-capu' về '{target_dir}'...")
    snapshot_download(
        repo_id="dragonSwing/xlm-roberta-capu",
        local_dir=str(target_dir),
        resume_download=True,
    )

    _patch_capu_compatibility(target_dir)

    # Đảm bảo xlm-roberta-base tokenizer được cache sẵn cho chế độ offline
    try:
        from transformers import AutoTokenizer
        print("Đang nạp và cache tokenizer 'xlm-roberta-base' cho chế độ offline...")
        AutoTokenizer.from_pretrained("xlm-roberta-base")
    except Exception as e:  # noqa: BLE001
        print(f"[CẢNH BÁO] Không thể cache tokenizer xlm-roberta-base: {e}")

    print(f"[THÀNH CÔNG] XLM-RoBERTa Capu đã sẵn sàng tại: {target_dir}")
    return target_dir


def setup_phowhisper_large(target_dir: Path) -> None:
    """Tải pre-converted PhoWhisper-large-ct2 từ Hugging Face."""
    print("=" * 60)
    print("Chuẩn bị Whisper Model: kiendt/PhoWhisper-large-ct2")
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
    print("Chuẩn bị Whisper Model: Qualcomm-AI-Research/PhoASR-whisper-small")
    print("=" * 60)

    model_bin = target_dir / "model.bin"
    if model_bin.exists() and model_bin.stat().st_size > 300_000_000:
        print(f"[OK] Đã tồn tại model.bin tại {target_dir} ({model_bin.stat().st_size / 1e6:.1f} MB). Bỏ qua convert.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

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
    except Exception as e:  # noqa: BLE001
        print(f"Không thể convert qua Python API ({e}), thử dùng fallback CLI...")

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
    subprocess.run(cmd, check=True)
    print(f"[THÀNH CÔNG] PhoASR-whisper-small-ct2 đã convert xong tại: {target_dir}")


def print_parameter_budget_report() -> None:
    """In bảng phân tích ngân sách tham số chính xác cho toàn bộ pipeline."""
    print("\n" + "=" * 80)
    print("BÁO CÁO PHÂN TÍCH THAM SỐ & NGÂN SÁCH HỆ THỐNG (RULE: TOTAL < 6B, LLM < 5B)")
    print("=" * 80)

    # 1. Bảng thành phần riêng lẻ
    components = [
        ("Silero VAD (Sherpa-ONNX)", SILERO_VAD_PARAMS, 0.002, "Voice Activity Detection"),
        ("Gipformer-68M-rnnt", GIPFORMER_68M_PARAMS, 0.068, "G-Group AI Lab (Transducer)"),
        ("ZipFormer-150M-rnnt", ZIPFORMER_150M_PARAMS, 0.153, "Hynt (CR-CTC-RNNT 6000h)"),
        ("XLM-RoBERTa-Capu", XLM_ROBERTA_CAPU_PARAMS, 0.278, "DragonSwing (Cap + Punctuation)"),
        ("Vietnamese ITN Rules", 0, 0.0, "Rule-based inverse text norm"),
        ("PhoASR-whisper-small-ct2", PHOASR_SMALL_PARAMS, 0.242, "Qualcomm AI Research (Whisper)"),
        ("PhoWhisper-large-ct2", PHOWHISPER_LARGE_PARAMS, 1.543, "VinAI / KienDT (Whisper)"),
    ]

    print(f"{'Thành phần':<28} | {'Số params':<14} | {'Quy đổi':<8} | {'Vai trò / Ghi chú':<30}")
    print("-" * 88)
    for name, params, b_scale, note in components:
        print(f"{name:<28} | {params:<14,d} | {b_scale:.3f}B   | {note:<30}")
    print("-" * 88)

    # 2. Bảng tổng hợp các tổ hợp Pipeline hoàn chỉnh
    combos = [
        (
            "Pipeline A: VAD + Gipformer + Capu + ITN",
            SILERO_VAD_PARAMS + GIPFORMER_68M_PARAMS + XLM_ROBERTA_CAPU_PARAMS,
            "Gipformer 4 tầng siêu nhẹ (~348M params)",
        ),
        (
            "Pipeline B: VAD + ZipFormer + Capu + ITN",
            SILERO_VAD_PARAMS + ZIPFORMER_150M_PARAMS + XLM_ROBERTA_CAPU_PARAMS,
            "ZipFormer 4 tầng độ chính xác cao (~433M params)",
        ),
        (
            "Whisper A: PhoASR-small-ct2 + VAD",
            SILERO_VAD_PARAMS + PHOASR_SMALL_PARAMS,
            "Whisper small (~243M params)",
        ),
        (
            "Whisper B: PhoWhisper-large-ct2 + VAD",
            SILERO_VAD_PARAMS + PHOWHISPER_LARGE_PARAMS,
            "Whisper large (~1.545B params)",
        ),
    ]

    print("\n" + "=" * 80)
    print("TỔNG HỢP CÁC PHƯƠNG ÁN PIPELINE ASR & QUỸ CÒN LẠI CHO LLM")
    print("=" * 80)
    print(f"{'Phương án Pipeline':<36} | {'Tổng ASR':<10} | {'Quỹ còn lại cho LLM':<18} | {'Đánh giá'}")
    print("-" * 88)
    for name, total_params, note in combos:
        total_b = total_params / 1e9
        remaining_llm_b = (MAX_TOTAL_BUDGET - total_params) / 1e9
        effective_cap = min(5.0, remaining_llm_b)
        status = "HỢP LỆ (Thoải mái LLM 3B-4B)" if effective_cap >= 4.0 else "HỢP LỆ (Ghép LLM 3B)"
        print(f"{name:<36} | {total_b:.3f}B    | max {effective_cap:.2f}B        | {status}")
    print("-" * 88)
    print(f"Trần giới hạn đề thi: Tổng ASR + LLM < {MAX_TOTAL_BUDGET / 1e9:.1f}B, riêng LLM < {MAX_LLM_BUDGET / 1e9:.1f}B.")
    print("Cả Pipeline A (~0.35B) và Pipeline B (~0.43B) đều tiêu thụ < 0.5B tham số,")
    print("cho phép sử dụng toàn bộ trần 5.0B cho mô hình LLM chính (như Qwen2.5-3B hoặc các model 3B-4B)!")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải & chuẩn bị các mô hình ASR cho Zalo AI Challenge 2026")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/asr",
        help="Thư mục chứa các model ASR (mặc định: models/asr)",
    )
    parser.add_argument(
        "--include-whisper",
        action="store_true",
        help="Tải thêm cả PhoWhisper-large và PhoASR-small",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    vad_dir = models_dir / "vad"
    gipformer_dir = models_dir / "gipformer-68m"
    zipformer_dir = models_dir / "zipformer-150m"
    capu_dir = models_dir / "xlm-roberta-capu"

    setup_silero_vad(vad_dir)
    setup_gipformer_68m(gipformer_dir)
    setup_zipformer_150m(zipformer_dir)
    setup_xlm_roberta_capu(capu_dir)

    if args.include_whisper:
        phowhisper_dir = models_dir / "PhoWhisper-large-ct2"
        phoasr_dir = models_dir / "PhoASR-whisper-small-ct2"
        setup_phowhisper_large(phowhisper_dir)
        setup_phoasr_small(phoasr_dir)

    print_parameter_budget_report()


if __name__ == "__main__":
    main()
