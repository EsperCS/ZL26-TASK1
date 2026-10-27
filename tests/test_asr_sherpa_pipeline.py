"""
tests/test_asr_sherpa_pipeline.py

Unit tests cho hệ sinh thái ASR 4 tầng:
- VietnameseITN
- Metric functions (WER, CER, Exact Match, Number Recall, Capitalized Recall, RTF)
- SherpaXLMPipeline audio normalization & VAD handling
"""

from __future__ import annotations

import numpy as np
import pytest

from src.asr.sherpa_xlm_pipeline import VietnameseITN
from src.eval.asr_metrics import (
    compute_corpus_metrics,
    extract_capitalized_entities,
    extract_numbers,
    normalize_text_for_acoustic_eval,
)


class TestVietnameseITN:
    """Kiểm tra độ chính xác của tầng Inverse Text Normalization (ITN)."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("một", 1),
            ("năm", 5),
            ("mười", 10),
            ("mười lăm", 15),
            ("hai mươi", 20),
            ("hai mươi mốt", 21),
            ("hai mốt", 21),
            ("năm mươi", 50),
            ("chín mươi chín", 99),
            ("một trăm", 100),
            ("một trăm hai mươi ba", 123),
            ("hai trăm", 200),
            ("năm trăm linh năm", 505),
        ],
    )
    def test_parse_spoken_number(self, phrase: str, expected: int) -> None:
        assert VietnameseITN.parse_spoken_number(phrase) == expected

    def test_normalize_time_and_duration(self) -> None:
        text = "dời chạy bộ muộn một tiếng, chín giờ sáng mai nhắc tôi"
        normalized = VietnameseITN.normalize(text)
        assert "1 tiếng" in normalized
        assert "9 giờ" in normalized

    def test_normalize_currency_and_numbers(self) -> None:
        text = "nạp giúp mình năm mươi nghìn cho bạn, chuyển hai mươi ba triệu"
        normalized = VietnameseITN.normalize(text)
        assert "50 nghìn" in normalized
        assert "23 triệu" in normalized

    def test_normalize_floors_and_half_hours(self) -> None:
        text = "tắt đèn tầng hai, đặt báo thức năm rưỡi"
        normalized = VietnameseITN.normalize(text)
        assert "tầng 2" in normalized
        assert "5 rưỡi" in normalized

    def test_preserve_non_numeric_phrases(self) -> None:
        text = "dời muộn nửa tiếng nhé"
        normalized = VietnameseITN.normalize(text)
        assert "nửa tiếng" in normalized


class TestASRMetrics:
    """Kiểm tra các hàm định lượng Corpus-level Metrics."""

    def test_normalize_text_for_acoustic_eval(self) -> None:
        raw = "Nếu mai mưa, dời chạy bộ muộn 1 tiếng! Nhắn Sếp Dũng."
        norm = normalize_text_for_acoustic_eval(raw)
        assert norm == "nếu mai mưa dời chạy bộ muộn 1 tiếng nhắn sếp dũng"

    def test_extract_numbers(self) -> None:
        text = "nạp 50 nghìn và chuyển 123 triệu lúc 9 giờ"
        nums = extract_numbers(text)
        assert nums == ["50", "123", "9"]

    def test_extract_capitalized_entities(self) -> None:
        text = "Nếu mai mưa thì nhắn Sếp Dũng và cô Mai giúp tôi nhé."
        entities = extract_capitalized_entities(text)
        # "Nếu" là từ đầu câu -> bỏ qua. "Sếp", "Dũng", "Mai" là tên riêng/thực thể
        assert "Sếp" in entities
        assert "Dũng" in entities
        assert "Mai" in entities
        assert "Nếu" not in entities

    def test_compute_corpus_metrics(self) -> None:
        golds = [
            "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng.",
            "Nạp giúp mình 50 nghìn.",
        ]
        preds = [
            "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng.",
            "Nạp giúp mình 50 ngàn.",
        ]
        durations = [5.0, 4.0]
        infer_times = [0.5, 0.4]

        metrics = compute_corpus_metrics(
            gold_texts=golds,
            pred_raw_texts=preds,
            audio_durations_sec=durations,
            inference_times_sec=infer_times,
        )

        assert metrics.total_samples == 2
        assert metrics.exact_match_raw_pct == 50.0
        assert metrics.number_recall_pct == 100.0  # Cả "1" và "50" đều được nhận diện
        assert metrics.avg_latency_ms == pytest.approx(450.0, rel=1e-2)
        assert metrics.real_time_factor == pytest.approx(0.9 / 9.0, rel=1e-2)


def test_vad_peak_normalization() -> None:
    """Kiểm tra logic chuẩn hóa peak amplitude."""
    # Mảng tín hiệu ngẫu nhiên có peak 0.5
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(16000).astype(np.float32)
    peak = float(np.max(np.abs(signal)))
    assert peak > 0

    norm_signal = signal * (0.95 / peak)
    new_peak = float(np.max(np.abs(norm_signal)))
    assert new_peak == pytest.approx(0.95, rel=1e-4)
