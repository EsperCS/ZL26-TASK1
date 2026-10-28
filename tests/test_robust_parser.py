from src.agent.lfm3_parser import parse_lfm3_response


def test_parser_with_inline_comments() -> None:
    sample2_raw = """{
  "brief_plan": "Phân tích câu hỏi của người dùng...",
  "tool_calls": [
    {
      "name": "medication.list",
      "arguments": {}
    },
    {
      "name": "medication.log_taken",
      "arguments": {
        "schedule_id": "<sẽ lấy từ medication.list>",
        "time": "2026-10-15T17:00:00+07:00" // Chiều hôm nay 16:25, có thể nhắc chiều 17:00? Cần xem lịch.
      }
    }
  ]
}"""
    parsed = parse_lfm3_response(sample2_raw)
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0]["name"] == "medication.list"
    assert parsed.tool_calls[1]["name"] == "medication.log_taken"


def test_parser_with_trailing_commas_and_python_literals() -> None:
    raw = """{
  "brief_plan": "Bật đèn",
  "tool_calls": [
    {
      "name": "smart_home.light_set",
      "arguments": {
        "device_id": "light_01",
        "power": "on",
        "enabled": True,
      },
    },
  ],
}"""
    parsed = parse_lfm3_response(raw)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "smart_home.light_set"
    assert parsed.tool_calls[0]["arguments"]["enabled"] is True


def test_parser_with_broken_outer_json_regex_fallback() -> None:
    raw = """Dưới đây là các lệnh cần thực thi:
    {"name": "robot_vacuum.start", "arguments": {"rooms": ["phòng khách"]}}
    Và sau đó:
    {"name": "tasks.add", "arguments": {"title": "Kiểm tra phòng"}}
    Xong rồi ạ!"""
    parsed = parse_lfm3_response(raw)
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0]["name"] == "robot_vacuum.start"
    assert parsed.tool_calls[1]["name"] == "tasks.add"


def test_detect_placeholder_arguments() -> None:
    from src.agent.lfm3_parser import tool_call_has_placeholder

    call_with_placeholder = {
        "name": "medication.log_taken",
        "arguments": {"schedule_id": "<sẽ lấy từ medication.list>", "time": "17:00"},
    }
    call_valid = {
        "name": "medication.log_taken",
        "arguments": {"schedule_id": "med_57", "time": "17:00"},
    }
    assert tool_call_has_placeholder(call_with_placeholder) is True
    assert tool_call_has_placeholder(call_valid) is False

