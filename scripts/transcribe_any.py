"""
scripts/transcribe_any.py

Công cụ chuyển đổi bất kỳ file âm thanh nào (WAV, M4A, MP3, AAC, OGG, FLAC)
sang văn bản tiếng Việt chuẩn mực (có đầy đủ dấu câu, viết hoa và số hóa)
sử dụng Pipeline ASR 4 tầng vô địch (Silero VAD + ZipFormer-150M + XLM-Capu + ITN).

Tự động:
1. Mount Google Drive nếu chạy trên Google Colab.
2. Chuyển đổi định dạng m4a/mp3 sang 16kHz mono qua ffmpeg.
3. Phân đoạn âm thanh dài và hiển thị thanh tiến độ theo thời gian thực.
4. Gắn timestamp [MM:SS] cho từng đoạn văn và lưu thành file Markdown (.md)
   ngay cạnh file audio gốc trên Google Drive.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import torch

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # noqa: BLE001
    DEFAULT_DEVICE = "cpu"

# Đảm bảo in tiếng Việt chuẩn trên mọi terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.sherpa_xlm_pipeline import SherpaXLMPipeline


def ensure_colab_drive() -> bool:
    """Tự động mount Google Drive nếu chạy trên Google Colab."""
    try:
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            print("[DRIVE] Đang kết nối Google Drive vào /content/drive...")
            drive.mount("/content/drive")
        return True
    except ImportError:
        return False


def resolve_audio_path(path_str: str | Path) -> Path:
    """
    Tự động chuẩn hóa và tìm đúng đường dẫn file âm thanh,
    hỗ trợ mọi định dạng viết tắt trên Google Colab như:
      - mydrive/uit-cnxhkh/on_gk.m4a
      - /mydrive/uit-cnxhkh/on_gk.m4a
      - /content/drive/MyDrive/uit-cnxhkh/on_gk.m4a
      - uit-cnxhkh/on_gk.m4a (tìm trong MyDrive nếu có)
    """
    raw_path = Path(path_str)
    if raw_path.exists():
        return raw_path

    s = str(path_str).replace("\\", "/").strip()
    is_drive_target = any(k in s.lower() for k in ["mydrive", "drive", "/content/drive"])

    if is_drive_target or Path("/content").exists():
        ensure_colab_drive()

    if raw_path.exists():
        return raw_path

    # Loại bỏ tiền tố thừa để tìm trong MyDrive
    clean_sub = re.sub(
        r"^(?:/?content/)?(?:/?drive/)?(?:/?mydrive/)?", "", s, flags=re.IGNORECASE
    ).lstrip("/")
    candidates = [
        Path("/content/drive/MyDrive") / clean_sub,
        Path("/content/drive/MyDrive") / s.lstrip("/"),
        Path("/content/drive/Shareddrives") / clean_sub,
        PROJECT_ROOT / clean_sub,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    return raw_path


def format_timestamp(seconds: float) -> str:
    """Định dạng giây thành chuỗi MM:SS hoặc HH:MM:SS."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def load_and_convert_audio(audio_path: Path) -> tuple[np.ndarray, int, float]:
    """
    Nạp và chuẩn hóa mọi định dạng audio (m4a, mp3, ogg, wav) thành 16kHz mono.
    Trả về (samples_float32, sample_rate, duration_seconds).
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh tại: {audio_path.resolve()}")

    # Thử đọc trực tiếp nếu là WAV chuẩn
    try:
        samples, sr = sf.read(str(audio_path), dtype="float32")
    except Exception:  # noqa: BLE001
        # Dùng ffmpeg giải mã các định dạng m4a, mp3, aac, wma...
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            tmp_wav = tmp_f.name
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                tmp_wav,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            samples, sr = sf.read(tmp_wav, dtype="float32")
        finally:
            Path(tmp_wav).unlink(missing_ok=True)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sr != 16000:
        num_target = int(len(samples) * 16000 / sr)
        samples = np.interp(
            np.linspace(0, len(samples), num_target, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
        sr = 16000

    duration = float(len(samples)) / float(sr)
    return samples, sr, duration


def transcribe_long_file(
    pipeline: SherpaXLMPipeline,
    audio_path: Path,
    chunk_duration_sec: float = 30.0,
    output_md_path: Path | None = None,
    capu_mode_desc: str = "XLM-RoBERTa Capu",
) -> str:
    """
    Chuyển đổi file âm thanh (ngắn hoặc dài) thành văn bản Markdown chuyên nghiệp kèm timestamps.
    """
    start_total_time = time.perf_counter()
    print(f"\n{'=' * 60}")
    print(f"[AUDIO ASR] Bắt đầu xử lý: {audio_path.name}")
    print(f"{'=' * 60}")

    samples, sr, duration = load_and_convert_audio(audio_path)
    print(
        f"📊 Thông tin file: {format_timestamp(duration)} ({duration:.1f}s) | 16000 Hz Mono"
    )

    chunk_size = int(chunk_duration_sec * sr)
    total_samples = len(samples)

    paragraphs: list[tuple[float, float, str]] = []

    # Nếu file ngắn (< 45s), xử lý nguyên khối
    if duration <= 45.0:
        print("\n🎙️ Đang nhận dạng âm thanh...")
        res = pipeline.transcribe(samples, sample_rate=sr)
        paragraphs.append((0.0, duration, res.final_text))
    else:
        # Nếu file dài, chia thành các khối 30s
        num_chunks = int(np.ceil(total_samples / chunk_size))
        print(
            f"\n🎙️ File dài {format_timestamp(duration)}, tự động chia thành {num_chunks} đoạn để nhận dạng..."
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_samples = samples[start_idx:end_idx]

            t_start = start_idx / sr
            t_end = end_idx / sr

            res = pipeline.transcribe(chunk_samples, sample_rate=sr)
            text = res.final_text.strip()

            if text:
                paragraphs.append((t_start, t_end, text))
                print(
                    f"  [{format_timestamp(t_start)} - {format_timestamp(t_end)}] {text}"
                )
            else:
                print(
                    f"  [{format_timestamp(t_start)} - {format_timestamp(t_end)}] (khoảng lặng)"
                )

    total_proc_time = time.perf_counter() - start_total_time
    rtf = total_proc_time / max(duration, 0.001)

    full_text = "\n\n".join(
        f"**[{format_timestamp(t0)} - {format_timestamp(t1)}]** {p_text}"
        for t0, t1, p_text in paragraphs
    )

    total_words = sum(len(p[2].split()) for p in paragraphs)

    # Tạo nội dung Markdown chuyên nghiệp
    md_content = f"""# BẢN GHI ÂM THOẠI: {audio_path.name}

> **Thời lượng file:** {format_timestamp(duration)} ({duration:.1f} giây)  
> **Tổng số từ:** {total_words:,} từ  
> **Thời gian xử lý:** {total_proc_time:.1f} giây (Tốc độ RTF: {rtf:.4f}x - Nhanh gấp {1/max(rtf, 1e-4):.1f} lần thời gian thực)  
> **Mô hình sử dụng:** ZipFormer-150M + Hotwords + {capu_mode_desc} + Vietnamese ITN  

---

## 📝 NỘI DUNG CHI TIẾT THEO MỐC THỜI GIAN

{full_text}

---
*Bản ghi được tự động chuyển đổi bởi LifeMate ASR Pipeline (Zalo AI Challenge 2026)*
"""

    if output_md_path is None:
        output_md_path = audio_path.parent / f"{audio_path.stem}.md"

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(md_content, encoding="utf-8")

    # Lưu thêm 1 file .txt thuần túy cho người dùng dễ copy
    txt_path = audio_path.parent / f"{audio_path.stem}.txt"
    plain_text = " ".join(p[2] for p in paragraphs)
    txt_path.write_text(plain_text, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("🎉 NHẬN DẠNG HOÀN TẤT THÀNH CÔNG!")
    print(f"📁 File Markdown đã lưu tại: {output_md_path.resolve()}")
    print(f"📄 File Text thuần đã lưu tại: {txt_path.resolve()}")
    print(f"⏱️ Tổng thời gian: {total_proc_time:.1f}s | Tốc độ: {rtf:.4f} RTF")
    print(f"{'=' * 60}\n")

    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chuyển đổi âm thanh bất kỳ sang văn bản tiếng Việt chuẩn mực"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Đường dẫn file âm thanh (m4a, mp3, wav, flac, ogg, mydrive/...)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Đường dẫn file .md xuất ra"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Độ dài mỗi đoạn phân chia (giây)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Thiết bị tính toán (cuda hoặc cpu)",
    )
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio_path)
    if not audio_path.exists():
        print(f"❌ [LỖI] Không tìm thấy file âm thanh tại: {args.audio_path}")
        print(f"   Đã kiểm tra đường dẫn giải quyết: {audio_path.resolve()}")
        sys.exit(1)

    # Khởi tạo Pipeline ASR 4 tầng
    models_dir = PROJECT_ROOT / "models" / "asr"
    zipformer_dir = models_dir / "zipformer-150m"
    vad_file = models_dir / "vad" / "silero_vad.onnx"
    hotwords_file = models_dir / "zalo_hotwords.txt"
    capu_ft_dir = models_dir / "xlm-roberta-capu-zalo-ft"
    capu_base_dir = models_dir / "xlm-roberta-capu"
    drive_backup_capu = Path("/content/drive/MyDrive/zaloai_asr_backup/xlm-roberta-capu-zalo-ft")

    # Tự động nạp bản fine-tuned từ Drive nếu phiên Colab bị khởi động lại
    if (
        not (capu_ft_dir / "pytorch_model.bin").exists()
        and (drive_backup_capu / "pytorch_model.bin").exists()
    ):
        print(f"[RESTORE] Phát hiện Capu Fine-Tuned trên Google Drive! Đang khôi phục từ {drive_backup_capu}...")
        import shutil

        shutil.copytree(drive_backup_capu, capu_ft_dir, dirs_exist_ok=True)

    has_ft_weights = (
        (capu_ft_dir / "pytorch_model.bin").exists()
        or (capu_ft_dir / "model.safetensors").exists()
    )
    has_base_weights = (
        (capu_base_dir / "pytorch_model.bin").exists()
        or (capu_base_dir / "model.safetensors").exists()
    )

    if has_ft_weights:
        capu_dir = capu_ft_dir
        capu_mode = "XLM-RoBERTa Capu (Zalo Fine-Tuned) 🚀"
    elif has_base_weights:
        capu_dir = capu_base_dir
        capu_mode = "XLM-RoBERTa Capu (Pretrained Base)"
    else:
        print("[AUTO-DOWNLOAD] Chưa có sẵn Capu, đang tải dragonSwing/xlm-roberta-capu...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="dragonSwing/xlm-roberta-capu",
                local_dir=str(capu_base_dir),
                resume_download=True,
            )
            capu_dir = capu_base_dir
            capu_mode = "XLM-RoBERTa Capu (Pretrained Base - Auto-Downloaded)"
        except Exception as e:  # noqa: BLE001
            print(f"[CẢNH BÁO] Không thể tải Capu ({e}). Tạm tắt tầng Capu.")
            capu_dir = None
            capu_mode = "Không sử dụng Capu"

    print(f"[KHỞI TẠO] Thiết bị giải mã: {args.device.upper()}")
    print(f"[KHỞI TẠO] Mô hình Capu: {capu_mode}")
    print("[KHỞI TẠO] Đang nạp mô hình SherpaXLMPipeline...")
    pipeline = SherpaXLMPipeline(
        model_name_or_dir=zipformer_dir,
        vad_model_path=vad_file,
        vad_min_silence=0.5,
        vad_min_speech=0.25,
        hotwords_file=hotwords_file if hotwords_file.exists() else None,
        hotwords_score=1.5,
        capu_model_dir=capu_dir if capu_dir and capu_dir.exists() else None,
        device=args.device,
        provider=args.device,
    )

    transcribe_long_file(
        pipeline=pipeline,
        audio_path=audio_path,
        chunk_duration_sec=args.chunk_duration,
        output_md_path=args.output,
        capu_mode_desc=capu_mode,
    )


if __name__ == "__main__":
    main()
