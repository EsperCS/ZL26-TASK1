"""tests/test_context_builder.py.

Unit tests for ContextBuilder, 21-day Calendar Generator, and Fuzzy Entity Grounding.
"""

from __future__ import annotations

from src.agent.context_builder import (
    ContextBuilder,
    build_calendar_table,
    build_prompt_for_sample,
    fuzzy_entity_grounding,
)


def test_build_calendar_table_structure() -> None:
    current_time_str = "2026-10-28T07:10:00+07:00"  # Thứ Tư
    table = build_calendar_table(current_time_str)

    assert "Mốc thời gian tương đối" in table
    assert "2026-10-28" in table
    assert "**Hôm nay (current_time)**" in table
    assert "Ngày mai" in table
    assert "Ngày kia" in table
    assert "Hôm qua" in table

    # Kiểm tra đủ 21 ngày (22 dòng bao gồm header và separator)
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert len(lines) == 23  # 1 header + 1 separator + 21 ngày


def test_fuzzy_entity_grounding_matches_close_names() -> None:
    user_profile = {
        "name": "Em Linh",
        "work_address": "Công viên phần mềm, Hải Châu, Đà Nẵng",
    }
    history = [
        {"role": "user", "text": "Hôm qua vừa gọi cho Sếp Dũng"},
        {"role": "assistant", "text": "Đã gửi tin nhắn cho Sếp Dũng"},
    ]

    # ASR nghe nhầm "Sếp Dụng" thay vì "Sếp Dũng"
    transcript = "nhắn cho Sếp Dụng bảo mai chạy bộ nhé"
    matches = fuzzy_entity_grounding(transcript, user_profile, history, threshold=75.0)

    assert len(matches) > 0
    # Tìm thấy gợi ý Sếp Dũng
    matched_canonical = [m[1] for m in matches]
    assert "Sếp Dũng" in matched_canonical


def test_fuzzy_entity_grounding_no_matches_when_unrelated() -> None:
    user_profile = {"name": "Em Linh"}
    history = []
    transcript = "bật điều hòa 24 độ phòng ngủ"
    matches = fuzzy_entity_grounding(transcript, user_profile, history, threshold=80.0)

    assert len(matches) == 0


def test_context_builder_system_prompt_generation() -> None:
    tools_catalog = {
        "apps": [
            {
                "id": "smart_home",
                "name_vi": "Nhà thông minh",
                "tools": [
                    {
                        "name": "smart_home.light_set",
                        "description": "Bật tắt đèn",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "device_id": {"type": "string", "description": "ID đèn"},
                                "power": {"type": "string", "enum": ["on", "off"]},
                            },
                            "required": ["device_id", "power"],
                        },
                    }
                ],
            }
        ]
    }

    builder = ContextBuilder(tools_catalog=tools_catalog)
    sample = {
        "id": "train_0001",
        "audio_text": "bật đèn phòng khách",
        "context": {
            "current_time": "2026-11-28T07:10:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "user_profile": {"name": "Linh"},
            "installed_apps": {
                "smart_home": {"tools": ["smart_home.light_set"]}
            },
        },
        "history": [],
    }

    messages = builder.build_messages(sample)
    assert len(messages) == 2  # System + User
    assert messages[0]["role"] == "system"
    assert "LifeMate — trợ lý ảo điều hành công cụ thông minh" in messages[0]["content"]
    assert "QUY TẮC BẮT BUỘC (HARD RULES)" in messages[0]["content"]
    
    assert messages[1]["role"] == "user"
    assert "[CALENDAR TABLE 21 DAYS" in messages[1]["content"]
    assert "2026-11-28" in messages[1]["content"]
    assert "[AVAILABLE TOOLS SCHEMA]" in messages[1]["content"]
    assert "smart_home.light_set" in messages[1]["content"]
    assert 'Primary Transcript: "bật đèn phòng khách"' in messages[1]["content"]


def test_build_self_correction_prompt() -> None:
    from src.agent.context_builder import build_self_correction_prompt

    err_prompt = build_self_correction_prompt(
        tool_name="calendar.update_event",
        error_message="Thiếu tham số bắt buộc event_id",
    )
    assert "[VALIDATION ERROR]" in err_prompt
    assert "calendar.update_event" in err_prompt
    assert "Thiếu tham số bắt buộc event_id" in err_prompt


def test_build_prompt_for_sample_helper() -> None:
    sample = {
        "id": "test_01",
        "audio_text": "mai có mưa không",
        "context": {"current_time": "2026-10-28T12:00:00+07:00"},
        "history": [],
    }
    messages = build_prompt_for_sample(sample, tools_catalog={})
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
