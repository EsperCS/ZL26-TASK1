"""tests/test_benchmark_slm_2026.py.

Unit tests for SLM benchmark candidate configurations, GPU config mapping, and summary formatting.
"""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark_slm_2026 import (
    CANDIDATE_REGISTRY,
    format_leaderboard_markdown,
    load_val_500_dataset,
    simulate_model_inference,
)
from src.eval.zalo_bfcl_evaluator import ZaloBFCLSummary

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_candidate_registry_specifications() -> None:
    assert len(CANDIDATE_REGISTRY) == 5

    names = [c.name for c in CANDIDATE_REGISTRY]
    assert "Qwen3.5-4B (Thinking)" in names
    assert "Qwen3.5-4B (No Thinking)" in names
    assert "Qwen3-4B-2507 (Thinking)" in names
    assert "Qwen3-4B-2507 (No Thinking)" in names
    assert "LiquidAI-LFM3-1.7B (Thinking)" in names

    # Kiểm tra tổng tham số < 5B cho SLM
    for c in CANDIDATE_REGISTRY:
        assert c.total_params_b < 5.0
        assert c.total_params_b + 0.433 < 6.0


def test_simulate_model_inference_thinking_on_vs_off() -> None:
    cand_think = CANDIDATE_REGISTRY[0]  # Thinking
    cand_no_think = CANDIDATE_REGISTRY[1]  # No Thinking

    sample = {
        "id": "sample_001",
        "audio_text": "bật đèn phòng khách",
        "context": {"installed_apps": {"smart_home": {"tools": ["smart_home.light_set"]}}},
        "label": {
            "tool_calls": [
                {"name": "smart_home.light_set", "arguments": {"device_id": "light_01", "power": "on"}},
                {"name": "smart_home.light_set", "arguments": {"device_id": "light_02", "power": "on"}},
            ]
        },
    }

    out_think = simulate_model_inference(cand_think, sample, "bật đèn phòng khách")
    assert out_think.parse_success is True
    assert len(out_think.reasoning_content) > 0
    assert out_think.thinking_token_count > 0
    assert len(out_think.tool_calls) == 2

    out_no_think = simulate_model_inference(cand_no_think, sample, "bật đèn phòng khách")
    assert out_no_think.parse_success is True
    assert out_no_think.reasoning_content == ""
    assert out_no_think.thinking_token_count == 0


def test_load_val_500_dataset_limit() -> None:
    train_path = REPO_ROOT / "train" / "train.jsonl"
    if not train_path.is_file():
        train_path = REPO_ROOT / "train+public_test" / "train" / "train.jsonl"

    samples = load_val_500_dataset(train_path, limit=10)
    assert len(samples) == 10
    assert "audio" in samples[0]
    assert "context" in samples[0]


def test_format_leaderboard_markdown() -> None:
    summaries = [
        ZaloBFCLSummary(
            model_name="Model A",
            input_mode="Oracle Mode",
            thinking_mode=True,
            total_samples=100,
            state_hash_match_pct=85.0,
            ast_tool_f1_pct=90.0,
            schema_adherence_pct=99.0,
            arg_exact_match_pct=82.0,
            avg_thinking_tokens=65.0,
            avg_latency_ms=120.0,
            slm_total_params_b=3.0,
            slm_active_params_b=3.0,
            total_system_params_b=3.433,
            budget_compliant=True,
        )
    ]
    md = format_leaderboard_markdown(summaries)
    assert "# BẢNG XẾP HẠNG SLM SOTA 2026" in md
    assert "Model A" in md
    assert "85.00%" in md
    assert "PASS" in md
