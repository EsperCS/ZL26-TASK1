"""tests/test_lfm3_pipeline.py.

Bộ unit test toàn diện cho LFM3 Pipeline (Parser, Agent ReAct Loop, ToolEnv Integration).
"""

from __future__ import annotations

from typing import Any

from src.agent.lfm3_parser import parse_lfm3_response
from src.agent.voice_agent import LFM3VoiceAgent


def test_parse_lfm3_native_xml() -> None:
    """Kiểm tra parser bóc tách đúng định dạng XML chính thức của Liquid AI."""
    raw_xml = (
        "<tool_call>\n"
        "<function=smart_home.light_set>\n"
        "<parameter=device_id>\n"
        "light_living_room\n"
        "</parameter>\n"
        "<parameter=power>\n"
        "off\n"
        "</parameter>\n"
        "<parameter=brightness>\n"
        "80\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = parse_lfm3_response(raw_xml)
    assert parsed.parse_format == "xml"
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call["name"] == "smart_home.light_set"
    assert call["arguments"]["device_id"] == "light_living_room"
    assert call["arguments"]["power"] == "off"
    assert call["arguments"]["brightness"] == 80


def test_parse_lfm3_multiple_xml_calls() -> None:
    """Kiểm tra parser xử lý nhiều tool calls đồng thời trong XML."""
    raw_xml = (
        "<think>Cần tra cứu danh bạ trước, sau đó gửi tin nhắn.</think>\n"
        "<tool_call>\n"
        "<function=contacts.search>\n"
        "<parameter=query>Sếp Dũng</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<tool_call>\n"
        "<function=sms.send>\n"
        "<parameter=contact_id>c_08</parameter>\n"
        "<parameter=content>Chào anh</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = parse_lfm3_response(raw_xml)
    assert parsed.parse_format == "xml"
    assert parsed.thinking_content == "Cần tra cứu danh bạ trước, sau đó gửi tin nhắn."
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0]["name"] == "contacts.search"
    assert parsed.tool_calls[0]["arguments"] == {"query": "Sếp Dũng"}
    assert parsed.tool_calls[1]["name"] == "sms.send"
    assert parsed.tool_calls[1]["arguments"] == {"contact_id": "c_08", "content": "Chào anh"}


def test_parse_lfm3_markdown_json() -> None:
    """Kiểm tra parser nhận diện khối JSON bọc markdown."""
    raw_json = (
        "```json\n"
        "{\n"
        '  "brief_plan": "Bật điều hòa 26 độ",\n'
        '  "tool_calls": [\n'
        "    {\n"
        '      "name": "smart_home.climate_set",\n'
        '      "arguments": {"device_id": "ac_01", "mode": "cool", "temperature": 26}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )
    parsed = parse_lfm3_response(raw_json)
    assert parsed.parse_format == "json_dict"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "smart_home.climate_set"
    assert parsed.tool_calls[0]["arguments"]["temperature"] == 26


def test_parse_lfm3_chatml() -> None:
    """Kiểm tra parser xử lý ChatML tool_call."""
    raw_chatml = '<tool_call>\n{"name": "weather.get_current", "arguments": {"location": "Hà Nội"}}\n</tool_call>'
    parsed = parse_lfm3_response(raw_chatml)
    assert parsed.parse_format == "chatml"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "weather.get_current"
    assert parsed.tool_calls[0]["arguments"] == {"location": "Hà Nội"}


def test_parse_lfm3_out_of_scope() -> None:
    """Kiểm tra parser nhận diện yêu cầu ngoài phạm vi."""
    raw_text = "Thiết bị chưa được cài đặt ứng dụng ngân hàng nên không thể thực hiện."
    parsed = parse_lfm3_response(raw_text)
    assert parsed.is_out_of_scope is True
    assert len(parsed.tool_calls) == 0


class MockSimulatorEnv:
    """Môi trường giả lập ToolEnv tuân thủ tuyệt đối quy chế."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.validate_called = 0

    def validate(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self.validate_called += 1
        args = arguments or {}
        if name == "smart_home.light_set" and "device_id" not in args:
            return {"ok": False, "error": "thiếu tham số bắt buộc: device_id"}
        return {"ok": True, "error": None}

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        self.calls.append({"name": name, "arguments": args})
        if "search" in name:
            return {"ok": True, "contacts": [{"id": "c_08", "name": "Sếp Dũng"}]}
        return {"ok": True, "done": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return list(self.calls)

    def get_state_hash(self) -> str:
        return "mock_hash_1234567890abcdef"


def test_voice_agent_react_execution() -> None:
    """Kiểm tra LFM3VoiceAgent thực thi luồng ReAct tương tác với ToolEnv."""
    mock_env = MockSimulatorEnv()
    dummy_catalog = {"apps": []}
    agent = LFM3VoiceAgent(tools_catalog=dummy_catalog)

    dummy_sample = {
        "id": "test_001",
        "audio_text": "Tìm Sếp Dũng trong danh bạ",
        "context": {
            "installed_apps": {"contacts": {"tools": ["contacts.search"]}},
        },
        "label": {
            "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}}],
        },
    }

    result = agent.run_sample(
        split="train",
        sample=dummy_sample,
        env=mock_env,
    )

    # 1. Kiểm tra Validate-Before-Call
    assert mock_env.validate_called >= 1, "Bắt buộc phải gọi env.validate trước khi gọi env.call"

    # 2. Kiểm tra cuộc gọi đã được ghi nhận vào ToolEnv
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "contacts.search"

    # 3. Kiểm tra State Hash
    assert result.state_hash == "mock_hash_1234567890abcdef"
    assert result.sample_id == "test_001"
