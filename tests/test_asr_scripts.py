"""
tests/test_asr_scripts.py

Unit tests cho module và script ASR (Task 1 - LIFEMATE).
"""

from scripts.setup_asr_models import (
    MAX_LLM_BUDGET,
    MAX_TOTAL_BUDGET,
    PHOASR_SMALL_PARAMS,
    PHOWHISPER_LARGE_PARAMS,
)
from src.eval.asr_metrics import normalize_text_for_acoustic_eval as normalize_vietnamese_text


def test_normalize_vietnamese_text():
    raw = "Nếu mai mưa, dời chạy bộ 1 tiếng; nhắn Sếp Dũng!"
    norm = normalize_vietnamese_text(raw)
    assert norm == "nếu mai mưa dời chạy bộ 1 tiếng nhắn sếp dũng"

    # Test Unicode NFC normalization
    raw_nfc = "Đà Nẵng - Cà Mau: 100k"
    norm_nfc = normalize_vietnamese_text(raw_nfc)
    assert "đà nẵng" in norm_nfc
    assert "cà mau" in norm_nfc
    assert "100k" in norm_nfc


def test_parameter_budget_bounds():
    # Model small
    assert PHOASR_SMALL_PARAMS == 241_734_912
    assert PHOASR_SMALL_PARAMS < 300_000_000

    # Model large
    assert PHOWHISPER_LARGE_PARAMS == 1_543_304_960
    assert PHOWHISPER_LARGE_PARAMS < 1_600_000_000

    # Total budget rule: ASR + LLM < 6B
    # Check pairing with Qwen2.5-3B (~3.08B)
    llm_3b_params = 3_080_000_000
    assert (PHOASR_SMALL_PARAMS + llm_3b_params) < MAX_TOTAL_BUDGET
    assert (PHOWHISPER_LARGE_PARAMS + llm_3b_params) < MAX_TOTAL_BUDGET
    assert llm_3b_params < MAX_LLM_BUDGET
