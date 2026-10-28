"""
src/asr/gpu_config.py

Tự động phát hiện kiến trúc GPU (Ampere, Hopper, Ada Lovelace, Turing)
và phân bổ cấu hình huấn luyện tối ưu (Precision, Batch Size, Gradient Accumulation, TF32).
"""

from __future__ import annotations

from typing import Any

import torch


def get_optimal_training_config(
    target_effective_batch_size: int = 32,
    base_lr: float = 1e-4,
) -> dict[str, Any]:
    """
    Tự động xác định cấu hình huấn luyện tối ưu theo thông số phần cứng GPU CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Không tìm thấy GPU CUDA! Vui lòng chạy script này trên Google Colab GPU.")

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    major_cc, minor_cc = torch.cuda.get_device_capability(0)

    # 1. Chọn Precision: Ampere trở lên (CC >= 8.0 như L4, A30, A100) dùng bf16 + tf32
    use_bf16 = major_cc >= 8
    use_fp16 = not use_bf16
    if major_cc >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 2. Chọn Batch Size theo dung lượng VRAM thực tế
    if vram_gb >= 35 or vram_gb >= 20:          # A100 40GB/80GB
        batch_size = 32
        num_workers = 4
        grad_ckpt = False
    elif vram_gb >= 14:        # Colab T4 (15-16GB)
        batch_size = 16
        num_workers = 2
        grad_ckpt = False
    else:                      # GPU < 14GB
        batch_size = 8
        num_workers = 2
        grad_ckpt = True

    grad_accum = max(1, target_effective_batch_size // batch_size)

    print(f"[GPU Auto-Config] {device_name} ({vram_gb:.1f} GB VRAM, CC {major_cc}.{minor_cc})")
    print(
        f" -> batch_size={batch_size}, grad_accum={grad_accum} "
        f"(Effective={batch_size * grad_accum}), bf16={use_bf16}, fp16={use_fp16}"
    )

    return {
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "gradient_accumulation_steps": grad_accum,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "tf32": major_cc >= 8,
        "gradient_checkpointing": grad_ckpt,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
        "learning_rate": base_lr,
        "auto_find_batch_size": True,  # Tự động giảm một nửa batch_size nếu lỡ bị CUDA OOM
    }
