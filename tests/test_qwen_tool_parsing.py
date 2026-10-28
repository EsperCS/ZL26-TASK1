"""tests/test_qwen_tool_parsing.py.

Kiểm thử toàn diện bộ phân tích (Parser) cho định dạng đầu ra của Qwen3.5-4B:
- Định dạng chuẩn XML: <tool_call><function=...><parameter=...>...</parameter></function></tool_call>
- Gọi nhiều tool liên tiếp trong 1 turn
- Bóc tách khối suy luận <think> ... </think>
- Tham số phức tạp: chuỗi, số, boolean, mảng đối tượng JSON (e.g. food items)
- Bộ lọc placeholder ảo giác
"""

from __future__ import annotations

import json

from src.agent.lfm3_parser import parse_lfm3_response, tool_call_has_placeholder


def test_qwen_xml_single_tool_no_params() -> None:
    raw = """<think>
Người dùng muốn xem lịch uống thuốc. Ta cần gọi medication.list.
</think>
<tool_call>
<function=medication.list>
</function>
</tool_call>"""
    parsed = parse_lfm3_response(raw)
    assert parsed.parse_format == "xml"
    assert "medication.list" in parsed.thinking_content or "uống thuốc" in parsed.thinking_content
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "medication.list"
    assert parsed.tool_calls[0]["arguments"] == {}


def test_qwen_xml_complex_params_list_of_objects() -> None:
    items_json = json.dumps([{"item_id": "rest_01_m1", "quantity": 1}])
    raw = f"""<think>
Người dùng đổi ý gọi tô bún bò đặc biệt từ nhà hàng rest_286.
</think>
<tool_call>
<function=food_delivery.place_order>
<parameter=restaurant_id>
rest_286
</parameter>
<parameter=items>
{items_json}
</parameter>
</function>
</tool_call>"""
    parsed = parse_lfm3_response(raw)
    assert parsed.parse_format == "xml"
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call["name"] == "food_delivery.place_order"
    assert call["arguments"]["restaurant_id"] == "rest_286"
    assert call["arguments"]["items"] == [{"item_id": "rest_01_m1", "quantity": 1}]


def test_qwen_xml_multiple_consecutive_tool_calls() -> None:
    raw = """<think>
Cần dọn phòng tiếp khách và lập task kiểm tra vào 9h sáng mai (2026-10-02T09:00:00+07:00).
</think>
<tool_call>
<function=robot_vacuum.start>
<parameter=rooms>
["phòng tiếp khách"]
</parameter>
</tool_call>
<tool_call>
<function=tasks.add>
<parameter=title>
Kiểm tra lại phòng tiếp khách sau khi robot dọn
</parameter>
<parameter=due>
2026-10-02T09:00:00+07:00
</parameter>
</function>
</tool_call>"""
    parsed = parse_lfm3_response(raw)
    assert parsed.parse_format == "xml"
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0]["name"] == "robot_vacuum.start"
    assert parsed.tool_calls[0]["arguments"]["rooms"] == ["phòng tiếp khách"]
    assert parsed.tool_calls[1]["name"] == "tasks.add"
    assert parsed.tool_calls[1]["arguments"]["due"] == "2026-10-02T09:00:00+07:00"


def test_qwen_placeholder_filtering() -> None:
    raw = """<tool_call>
<function=medication.log_taken>
<parameter=schedule_id>
<sẽ lấy từ medication.list>
</parameter>
<parameter=time>
2026-10-15T17:30:00+07:00
</parameter>
</function>
</tool_call>"""
    parsed = parse_lfm3_response(raw)
    assert len(parsed.tool_calls) == 1
    assert tool_call_has_placeholder(parsed.tool_calls[0]) is True


def test_qwen_unclosed_xml_recovery() -> None:
    # Trường hợp mô hình bị ngắt token hoặc quên đóng thẻ </function></tool_call>
    raw = """<think>
Kiểm tra phòng khách
</think>
<tool_call>
<function=smart_home.light_get_status>
<parameter=device_id>
light_living
"""
    parsed = parse_lfm3_response(raw)
    assert parsed.parse_format == "xml"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "smart_home.light_get_status"
    assert parsed.tool_calls[0]["arguments"]["device_id"] == "light_living"
