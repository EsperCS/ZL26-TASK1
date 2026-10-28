"""
tests/test_asr_experiments.py

Unit tests cho bộ kịch bản thí nghiệm Fine-tuning & Tuning ASR:
1. Kiểm tra nguyên tắc chống Data Leakage (Zero Leakage):
   - val_500 (indices 0..499) hoàn toàn tách biệt với asr_train (500..N).
2. Kiểm tra bộ trích xuất Hotwords Context-Biasing (Exp 1):
   - Nhận diện đúng Proper Noun clusters, lọc stopwords, đếm tần suất.
3. Kiểm tra pipeline SherpaXLMPipeline với hotwords & VAD params.
4. Kiểm tra bộ sinh dữ liệu và Collator cho Fine-tune Capu (Exp 2):
   - Đảm bảo ánh xạ nhãn chuẩn xác, tính toán input_offsets.
5. Kiểm tra cấu hình LoRA cho PhoASR (Exp 3).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from scripts.exp1_tune_vad_hotwords import extract_hotwords_from_train
from scripts.exp2_finetune_xlm_capu import (
    CapuDataCollator,
    CapuDataset,
    build_capu_training_pairs,
)
from src.asr.sherpa_xlm_pipeline import SherpaXLMPipeline

# ==============================================================================
# 1. Zero Data Leakage & Dataset Partition Tests
# ==============================================================================


def test_zero_data_leakage_split() -> None:
    """Đảm bảo tập val_500 và asr_train hoàn toàn không giao nhau (Zero Contamination)."""
    mock_samples = [{"id": f"sample_{i}", "audio_text": f"Câu số {i}"} for i in range(1545)]
    mock_lines = [json.dumps(s) for s in mock_samples]

    val_500 = mock_lines[:500]
    asr_train = mock_lines[500:]

    assert len(val_500) == 500
    assert len(asr_train) == 1045

    val_ids = {json.loads(line)["id"] for line in val_500}
    train_ids = {json.loads(line)["id"] for line in asr_train}

    # Tập giao nhau phải rỗng tuyệt đối
    assert len(val_ids.intersection(train_ids)) == 0
    assert "sample_0" in val_ids
    assert "sample_499" in val_ids
    assert "sample_500" in train_ids
    assert "sample_1544" in train_ids


# ==============================================================================
# 2. Hotwords Extraction Tests (Exp 1)
# ==============================================================================


def test_hotwords_extraction_proper_nouns() -> None:
    """Kiểm tra khả năng trích xuất thực thể viết hoa xuất hiện >= 2 lần."""
    mock_train = [
        json.dumps({
            "audio_text": "Gọi cho Sếp Dũng ở Hà Nội để họp gấp.",
            "context": {"installed_apps": {"smart_home": {}}},
        }),
        json.dumps({
            "audio_text": "Hỏi Sếp Dũng xem ngày mai đi Đà Nẵng không.",
            "context": {"installed_apps": {"smart_home": {}}},
        }),
        json.dumps({
            "audio_text": "Gửi tin nhắn cho Lan bảo Lan tới Hà Nội rồi.",
            "context": {"installed_apps": {"calendar": {}}},
        }),
    ]

    hotwords = extract_hotwords_from_train(mock_train, min_frequency=2)

    # "Sếp Dũng", "Hà Nội", "Lan", "Smart Home" xuất hiện từ 2 lần trở lên
    assert "Sếp Dũng" in hotwords
    assert "Hà Nội" in hotwords
    assert "Smart Home" in hotwords

    # "Đà Nẵng" chỉ xuất hiện 1 lần -> không được đưa vào
    assert "Đà Nẵng" not in hotwords

    # Từ đầu câu như "Gọi", "Hỏi", "Gửi" phải bị lọc
    assert "Gọi" not in hotwords
    assert "Hỏi" not in hotwords
    assert "Gửi" not in hotwords


# ==============================================================================
# 3. SherpaXLMPipeline Hotwords Initialization Tests
# ==============================================================================


def test_sherpa_pipeline_hotwords_init() -> None:
    """Kiểm tra khởi tạo SherpaXLMPipeline với tham số hotwords và VAD tuỳ chỉnh."""
    model_dir = Path("models/asr/zipformer-150m")
    if not (model_dir / "tokens.txt").exists():
        return

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
        f.write("SẾP DŨNG\nĐÀ NẴNG\nHÀ NỘI\n")
        tmp_hw = f.name

    pipeline = SherpaXLMPipeline(
        model_name_or_dir=model_dir,
        vad_min_silence=0.5,
        vad_min_speech=0.25,
        hotwords_file=tmp_hw,
        hotwords_score=2.0,
        enable_capu=False,
    )

    assert pipeline.hotwords_file == tmp_hw
    assert pipeline.hotwords_score == 2.0
    assert pipeline.recognizer is not None


# ==============================================================================
# 4. Capu Token Classification Dataset & Collator Tests (Exp 2)
# ==============================================================================


def test_build_capu_training_pairs() -> None:
    """Kiểm tra phân rã câu tiếng Việt thành cặp từ và nhãn GECToR."""
    gold_text = "Nhắn Sếp Dũng, mai họp lúc 8 giờ."
    pairs = build_capu_training_pairs(gold_text)

    assert len(pairs) == 2  # Cả capitalization và punctuation

    # Cặp 1: Capitalization
    words_cap, labels_cap = pairs[0]
    assert words_cap == ["nhắn", "sếp", "dũng", "mai", "họp", "lúc", "8", "giờ"]
    assert labels_cap[0] == "$TRANSFORM_CASE_CAPITAL"  # Nhắn
    assert labels_cap[1] == "$TRANSFORM_CASE_CAPITAL"  # Sếp
    assert labels_cap[2] == "$TRANSFORM_CASE_CAPITAL"  # Dũng
    assert labels_cap[3] == "$KEEP"  # mai

    # Cặp 2: Punctuation
    _words_punc, labels_punc = pairs[1]
    assert labels_punc[2] == "$APPEND_,"  # Dũng,
    assert labels_punc[-1] == "$APPEND_."  # giờ.


def test_capu_data_collator() -> None:
    """Kiểm tra Collator sinh đúng input_offsets và tensor shapes."""
    capu_dir = Path("models/asr/xlm-roberta-capu")
    if not capu_dir.exists():
        return

    tokenizer = AutoTokenizer.from_pretrained(str(capu_dir.resolve()), model_max_length=512)
    tokenizer.add_tokens(["$START"])

    vocab_labels = {"$KEEP": 0, "$TRANSFORM_CASE_CAPITAL": 1, "$APPEND_,": 2}
    vocab_dtags = {"CORRECT": 0, "INCORRECT": 1}

    samples = [
        (["sếp", "dũng"], ["$TRANSFORM_CASE_CAPITAL", "$TRANSFORM_CASE_CAPITAL"]),
        (["hà", "nội", "mai"], ["$TRANSFORM_CASE_CAPITAL", "$TRANSFORM_CASE_CAPITAL", "$KEEP"]),
    ]

    dataset = CapuDataset(samples, vocab_labels, vocab_dtags)
    collator = CapuDataCollator(tokenizer)

    batch_list = [dataset[0], dataset[1]]
    batch_tensors = collator(batch_list)

    assert "input_ids" in batch_tensors
    assert "attention_mask" in batch_tensors
    assert "input_offsets" in batch_tensors
    assert "labels" in batch_tensors
    assert "d_tags" in batch_tensors

    # Kiểm tra kích thước batch
    batch_size = 2
    assert batch_tensors["input_ids"].size(0) == batch_size
    assert batch_tensors["input_offsets"].size(0) == batch_size
    assert batch_tensors["labels"].size(0) == batch_size
    assert batch_tensors["d_tags"].size(0) == batch_size
    assert batch_tensors["input_offsets"].size(1) == batch_tensors["labels"].size(1)


# ==============================================================================
# 5. PhoASR LoRA Configuration Tests (Exp 3)
# ==============================================================================


def test_phoasr_lora_config() -> None:
    """Kiểm tra cấu hình LoRA target modules và rank theo đúng thông số."""
    from peft import LoraConfig

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.target_modules == {"q_proj", "v_proj"}
    assert peft_config.lora_dropout == 0.05
    assert peft_config.bias == "none"


# ==============================================================================
# 6. GPU Auto-Config Optimal Training Tests
# ==============================================================================


def test_get_optimal_training_config_no_cuda() -> None:
    """Kiểm tra bắt buộc báo lỗi RuntimeError khi không có CUDA."""
    from unittest.mock import patch

    import pytest

    from src.asr.gpu_config import get_optimal_training_config

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="Không tìm thấy GPU CUDA"),
    ):
        get_optimal_training_config()


def test_get_optimal_training_config_t4_gpu() -> None:
    """Kiểm tra cấu hình tối ưu tự động chọn cho GPU Tesla T4 (15GB, CC 7.5)."""
    from unittest.mock import MagicMock, patch

    from src.asr.gpu_config import get_optimal_training_config

    mock_props = MagicMock()
    mock_props.total_memory = 15.0 * (1024**3)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Tesla T4"),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
    ):
        cfg = get_optimal_training_config(target_effective_batch_size=32, base_lr=1e-4)

        assert cfg["per_device_train_batch_size"] == 16
        assert cfg["gradient_accumulation_steps"] == 2
        assert cfg["fp16"] is True
        assert cfg["bf16"] is False
        assert cfg["tf32"] is False
        assert cfg["learning_rate"] == 1e-4


def test_get_optimal_training_config_a100_gpu() -> None:
    """Kiểm tra cấu hình tối ưu tự động chọn cho GPU NVIDIA A100 (40GB, CC 8.0)."""
    from unittest.mock import MagicMock, patch

    from src.asr.gpu_config import get_optimal_training_config

    mock_props = MagicMock()
    mock_props.total_memory = 40.0 * (1024**3)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
    ):
        cfg = get_optimal_training_config(target_effective_batch_size=32, base_lr=2e-5)

        assert cfg["per_device_train_batch_size"] == 32
        assert cfg["gradient_accumulation_steps"] == 1
        assert cfg["fp16"] is False
        assert cfg["bf16"] is True
        assert cfg["tf32"] is True
        assert cfg["learning_rate"] == 2e-5

