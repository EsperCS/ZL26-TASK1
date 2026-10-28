"""tests/test_zalo_bfcl_evaluator.py.

Unit tests for Zalo-BFCL Evaluator, Thinking Output Parser, and Metric Calculations.
"""

from __future__ import annotations

from src.eval.zalo_bfcl_evaluator import (
    ZaloBFCLEvaluator,
    parse_thinking_and_tool_calls,
)


def test_parse_thinking_and_tool_calls_with_think_tag() -> None:
    raw_output = """
    <think>
    Người dùng muốn đặt báo thức lúc 6 giờ sáng ngày mai.
    Ngày mai tương ứng với 2026-10-29.
    Tôi cần gọi tool alarm.create với tham số time = "2026-10-29T06:00:00+07:00".
    </think>
    ```json
    {
      "brief_plan": "Tạo báo thức 6h sáng mai",
      "tool_calls": [
        {
          "name": "alarm.create",
          "arguments": {"time": "2026-10-29T06:00:00+07:00", "label": "thức dậy"}
        }
      ]
    }
    ```
    """
    res = parse_thinking_and_tool_calls(raw_output)
    assert res.parse_success is True
    assert "Người dùng muốn đặt báo thức" in res.reasoning_content
    assert res.thinking_token_count > 10
    assert res.brief_plan == "Tạo báo thức 6h sáng mai"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "alarm.create"
    assert res.tool_calls[0]["arguments"]["label"] == "thức dậy"


def test_parse_thinking_and_tool_calls_direct_array() -> None:
    raw_output = """
    [
      {"name": "smart_home.light_set", "arguments": {"device_id": "light_01", "power": "off"}}
    ]
    """
    res = parse_thinking_and_tool_calls(raw_output)
    assert res.parse_success is True
    assert res.reasoning_content == ""
    assert res.thinking_token_count == 0
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "smart_home.light_set"


def test_parse_thinking_and_tool_calls_malformed() -> None:
    raw_output = "Tôi không thể giúp bạn thực hiện yêu cầu này vì thiếu thông tin."
    res = parse_thinking_and_tool_calls(raw_output)
    assert res.parse_success is False
    assert len(res.tool_calls) == 0


def test_zalo_bfcl_evaluator_metrics() -> None:
    catalog = {
        "apps": [
            {
                "id": "smart_home",
                "name_vi": "Nhà thông minh",
                "tools": [
                    {
                        "name": "smart_home.light_set",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "device_id": {"type": "string"},
                                "power": {"type": "string", "enum": ["on", "off"]},
                            },
                            "required": ["device_id", "power"],
                        },
                    }
                ],
            }
        ]
    }

    evaluator = ZaloBFCLEvaluator(
        catalog=catalog,
        slm_total_params_b=3.09,
        asr_params_b=0.433,
    )

    assert evaluator.budget_compliant is True
    assert evaluator.total_system_params_b == 3.523

    gold_record = {
        "id": "train_001",
        "context": {
            "installed_apps": {"smart_home": {"tools": ["smart_home.light_set"]}}
        },
        "label": {
            "tool_calls": [
                {"name": "smart_home.light_set", "arguments": {"device_id": "light_01", "power": "on"}}
            ],
            "state_hash": "dummy_hash_123",
        },
    }

    # Case 1: Khớp hoàn toàn
    parsed_pred = parse_thinking_and_tool_calls(
        '{"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_01", "power": "on"}}]}'
    )
    res = evaluator.evaluate_sample(parsed_pred, gold_record)
    assert res.ast_tool_f1 == 1.0
    assert res.schema_adherence_rate == 1.0
    assert res.arg_exact_match_rate == 1.0

    # Case 2: Sai tham số
    parsed_wrong = parse_thinking_and_tool_calls(
        '{"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_01", "power": "off"}}]}'
    )
    res_wrong = evaluator.evaluate_sample(parsed_wrong, gold_record)
    assert res_wrong.ast_tool_f1 == 1.0  # Chọn đúng tool name
    assert res_wrong.arg_exact_match_rate == 0.0  # Nhưng sai power "off" != "on"

    # Summarize
    summary = evaluator.summarize(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        input_mode="Oracle Mode",
        thinking_mode=False,
        sample_metrics=[res, res_wrong],
    )
    assert summary.ast_tool_f1_pct == 100.0
    assert summary.arg_exact_match_pct == 50.0
    assert summary.budget_compliant is True
