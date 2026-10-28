"""
scripts/backup_to_drive.py

Tự động sao lưu và khôi phục các checkpoints & tài nguyên ASR lên Google Drive:
1. Sao lưu (Backup):
   - models/asr/zalo_hotwords.txt (Danh sách từ vựng Context-Biasing)
   - models/asr/xlm-roberta-capu-zalo-ft (Mô hình Capu Fine-tuned chiến thắng)
   - models/asr/phoasr_whisper_small_merged (Mô hình Whisper LoRA merged)
   - logs/ (Bảng xếp hạng Leaderboard và các báo cáo thực nghiệm)
2. Khôi phục (Restore):
   - Nạp toàn bộ tài nguyên từ Drive vào phiên Colab mới trong vài giây.
3. Đóng gói (Zip):
   - Hỗ trợ nén thành file .zip để lưu trữ hoặc tải về máy cá nhân offline.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn trên mọi terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def mount_google_drive() -> bool:
    """Tự động mount Google Drive nếu đang chạy trong môi trường Google Colab."""
    try:
        from google.colab import drive

        drive_root = Path("/content/drive")
        if not (drive_root / "MyDrive").exists():
            print("[DRIVE] Đang kết nối với Google Drive qua google.colab.drive...")
            drive.mount("/content/drive")
            print("[THÀNH CÔNG] Đã mount Google Drive tại: /content/drive/MyDrive")
        else:
            print("[DRIVE] Google Drive đã được mount sẵn.")
        return True
    except ImportError:
        print("[LOCAL] Không phát hiện môi trường Google Colab. Sử dụng đường dẫn thông thường.")
        return False


def get_dir_size_mb(path: Path) -> float:
    """Tính tổng dung lượng thư mục (MB)."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def backup_to_drive(
    drive_dir: Path,
    models_dir: Path,
    logs_dir: Path,
    include_whisper: bool = True,
    create_zip: bool = False,
) -> None:
    """Sao lưu các checkpoints và logs lên Google Drive."""
    mount_google_drive()
    drive_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[BACKUP] Bắt đầu sao lưu tài nguyên vào: {drive_dir.resolve()}...")

    # 1. Sao lưu zalo_hotwords.txt
    hw_src = models_dir / "zalo_hotwords.txt"
    if hw_src.exists():
        hw_dst = drive_dir / "zalo_hotwords.txt"
        shutil.copy2(hw_src, hw_dst)
        print(f"  [OK] Đã sao lưu Hotwords: {hw_dst.name} ({get_dir_size_mb(hw_src):.2f} MB)")
    else:
        print(f"  [BỎ QUA] Không tìm thấy file hotwords tại {hw_src}")

    # 2. Sao lưu XLM-RoBERTa Capu Fine-tuned (Mô hình quan trọng nhất)
    capu_src = models_dir / "xlm-roberta-capu-zalo-ft"
    if capu_src.exists():
        capu_dst = drive_dir / "xlm-roberta-capu-zalo-ft"
        capu_dst.mkdir(parents=True, exist_ok=True)
        print(f"  [ĐANG CHÉP] XLM-RoBERTa Capu Zalo-FT ({get_dir_size_mb(capu_src):.1f} MB)...")
        shutil.copytree(capu_src, capu_dst, dirs_exist_ok=True)
        print(f"  [OK] Đã sao lưu hoàn tất: {capu_dst.resolve()}")
    else:
        print(f"  [BỎ QUA] Không tìm thấy checkpoint Capu tại {capu_src}")

    # 3. Sao lưu PhoASR Whisper small merged (nếu được chọn)
    if include_whisper:
        whisper_src = models_dir / "phoasr_whisper_small_merged"
        if whisper_src.exists():
            whisper_dst = drive_dir / "phoasr_whisper_small_merged"
            whisper_dst.mkdir(parents=True, exist_ok=True)
            print(f"  [ĐANG CHÉP] PhoASR Whisper Merged ({get_dir_size_mb(whisper_src):.1f} MB)...")
            shutil.copytree(whisper_src, whisper_dst, dirs_exist_ok=True)
            print(f"  [OK] Đã sao lưu hoàn tất: {whisper_dst.resolve()}")
        else:
            print(f"  [BỎ QUA] Không tìm thấy mô hình Whisper merged tại {whisper_src}")

    # 4. Sao lưu các báo cáo và logs Markdown
    if logs_dir.exists():
        logs_dst = drive_dir / "logs"
        logs_dst.mkdir(parents=True, exist_ok=True)
        for md_file in logs_dir.glob("*.md"):
            shutil.copy2(md_file, logs_dst / md_file.name)
        print(f"  [OK] Đã sao lưu các báo cáo đánh giá vào: {logs_dst.resolve()}")

    # 5. Tạo file zip đóng gói nếu được yêu cầu
    if create_zip:
        zip_base_name = str(drive_dir / "zalo_asr_checkpoints")
        print(f"\n[ZIP] Đang nén toàn bộ thư mục thành '{zip_base_name}.zip'...")
        shutil.make_archive(zip_base_name, "zip", str(drive_dir))
        zip_path = Path(f"{zip_base_name}.zip")
        print(f"[THÀNH CÔNG] File nén đã sẵn sàng tại: {zip_path.resolve()} ({get_dir_size_mb(zip_path):.1f} MB)")

    print(f"\n{'=' * 60}")
    print("[HOÀN TẤT] Mọi tài nguyên đã được bảo vệ an toàn trên Google Drive tại:")
    print(f"👉 {drive_dir.resolve()}")
    print(f"{'=' * 60}\n")


def restore_from_drive(
    drive_dir: Path,
    models_dir: Path,
    logs_dir: Path,
) -> None:
    """Khôi phục toàn bộ checkpoints và tài nguyên từ Google Drive về môi trường làm việc."""
    mount_google_drive()
    if not drive_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục sao lưu trên Google Drive: {drive_dir.resolve()}")

    print(f"\n[RESTORE] Đang khôi phục tài nguyên từ Google Drive: {drive_dir.resolve()}...")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Khôi phục hotwords
    hw_drive = drive_dir / "zalo_hotwords.txt"
    if hw_drive.exists():
        shutil.copy2(hw_drive, models_dir / "zalo_hotwords.txt")
        print("  [OK] Đã khôi phục zalo_hotwords.txt")

    # 2. Khôi phục Capu FT
    capu_drive = drive_dir / "xlm-roberta-capu-zalo-ft"
    if capu_drive.exists():
        dest = models_dir / "xlm-roberta-capu-zalo-ft"
        shutil.copytree(capu_drive, dest, dirs_exist_ok=True)
        print(f"  [OK] Đã khôi phục XLM-Capu Zalo-FT ({get_dir_size_mb(dest):.1f} MB)")

    # 3. Khôi phục Whisper merged
    whisper_drive = drive_dir / "phoasr_whisper_small_merged"
    if whisper_drive.exists():
        dest = models_dir / "phoasr_whisper_small_merged"
        shutil.copytree(whisper_drive, dest, dirs_exist_ok=True)
        print(f"  [OK] Đã khôi phục PhoASR Whisper Merged ({get_dir_size_mb(dest):.1f} MB)")

    # 4. Khôi phục logs
    logs_drive = drive_dir / "logs"
    if logs_drive.exists():
        for md_file in logs_drive.glob("*.md"):
            shutil.copy2(md_file, logs_dir / md_file.name)
        print("  [OK] Đã khôi phục các file báo cáo logs")

    print(f"\n{'=' * 60}")
    print("[THÀNH CÔNG] Đã khôi phục toàn bộ checkpoints! Sẵn sàng sử dụng mà không cần huấn luyện lại.")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sao lưu và Khôi phục Checkpoints Zalo AI ASR với Google Drive")
    parser.add_argument(
        "--action",
        choices=["backup", "restore"],
        default="backup",
        help="Hành động: 'backup' (sao lưu lên Drive) hoặc 'restore' (khôi phục từ Drive)",
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/ZaloAI_2026_ASR"),
        help="Thư mục lưu trữ trên Google Drive (Mặc định: /content/drive/MyDrive/ZaloAI_2026_ASR)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/asr"),
        help="Thư mục models cục bộ",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Thư mục logs cục bộ",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Bỏ qua mô hình PhoASR Whisper merged (chỉ sao lưu Capu FT & Hotwords để tiết kiệm dung lượng)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Tạo thêm file .zip đóng gói trong Google Drive",
    )
    args = parser.parse_args()

    if args.action == "backup":
        backup_to_drive(
            drive_dir=args.drive_dir,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
            include_whisper=not args.no_whisper,
            create_zip=args.zip,
        )
    elif args.action == "restore":
        restore_from_drive(
            drive_dir=args.drive_dir,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
        )


if __name__ == "__main__":
    main()
