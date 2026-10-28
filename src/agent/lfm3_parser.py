"""src/agent/lfm3_parser.py.

Bộ giải mã (Parser) chuẩn hóa cho mô hình Liquid Foundation Models (LFM3-1.7B).
Hỗ trợ toàn diện 4 định dạng đầu ra của LFM3:
1. Native LFM3 XML Tool Call:
   <tool_call>
   <function=app.tool>
   <parameter=key>value</parameter>
   </function>
   </tool_call>
2. Markdown JSON block (```json { "tool_calls": [...] } ```).
3. ChatML JSON Tool Call (<tool_call>{"name": ..., "arguments": ...}</tool_call>).
4. Direct JSON Array ([{"name": ..., "arguments": ...}]).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Regex trích xuất định dạng XML chính thức của Liquid AI
CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)

# Regex trích xuất ChatML tool_call
CHATML_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


@dataclass(slots=True)
class LFM3ParsedOutput:
    """Kết quả bóc tách chuẩn hóa từ văn bản sinh ra của LFM3-1.7B."""
    raw_text: str
    thinking_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    brief_plan: str | None = None
    is_out_of_scope: bool = False
    parse_format: str = "none"
    error_message: str | None = None


def parse_lfm3_response(raw_text: str) -> LFM3ParsedOutput:
    """Phân tích và bóc tách toàn diện phản hồi của LFM3-1.7B thành danh sách tool_calls chuẩn."""
    text = (raw_text or "").strip()
    thinking_content = ""

    # 1. Tách khối suy luận <think>...</think> nếu có
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1).strip()
        text = (text[: think_match.start()] + text[think_match.end() :]).strip()
    elif "<think>" in text:
        parts = text.split("<think>", 1)
        text = parts[0].strip()
        thinking_content = parts[1].strip()

    # 2. Định dạng 1: Native LFM3 XML Tool Call (<tool_call><function=...>)
    xml_matches = CALL_RE.findall(text)
    if xml_matches:
        tool_calls: list[dict[str, Any]] = []
        for fn_name, body in xml_matches:
            fn_name = fn_name.strip()
            args: dict[str, Any] = {}
            for param_key, param_val in PARAM_RE.findall(body):
                param_key = param_key.strip()
                param_val = param_val.strip()
                try:
                    # Parse các giá trị số, boolean, json object hoặc list
                    args[param_key] = json.loads(param_val)
                except (json.JSONDecodeError, ValueError):
                    # Nếu là chuỗi text thuần
                    args[param_key] = param_val
            tool_calls.append({"name": fn_name, "arguments": args})

        return LFM3ParsedOutput(
            raw_text=raw_text,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            parse_format="xml",
            is_out_of_scope=(len(tool_calls) == 0),
        )

    # 3. Định dạng 2: ChatML JSON (<tool_call>{...}</tool_call>)
    chatml_matches = CHATML_CALL_RE.findall(text)
    if chatml_matches:
        tool_calls = []
        for json_str in chatml_matches:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls.append({
                        "name": str(parsed["name"]).strip(),
                        "arguments": parsed.get("arguments", {}),
                    })
            except (json.JSONDecodeError, ValueError):
                continue

        if tool_calls:
            return LFM3ParsedOutput(
                raw_text=raw_text,
                thinking_content=thinking_content,
                tool_calls=tool_calls,
                parse_format="chatml",
            )

    # 4. Định dạng 3: Markdown codeblock JSON (```json ... ```) hoặc direct JSON
    json_candidate = text
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_block:
        json_candidate = json_block.group(1).strip()
    elif not json_candidate.startswith(("{", "[")):
        bracket_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", json_candidate)
        if bracket_match:
            json_candidate = bracket_match.group(1).strip()

    try:
        parsed_json = json.loads(json_candidate)
        brief_plan: str | None = None
        tool_calls = []

        if isinstance(parsed_json, dict):
            brief_plan = parsed_json.get("brief_plan")
            raw_calls = parsed_json.get("tool_calls", [])
            if isinstance(raw_calls, list):
                tool_calls = [
                    {"name": c["name"].strip(), "arguments": c.get("arguments", {})}
                    for c in raw_calls
                    if isinstance(c, dict) and "name" in c
                ]
            return LFM3ParsedOutput(
                raw_text=raw_text,
                thinking_content=thinking_content,
                tool_calls=tool_calls,
                brief_plan=brief_plan,
                parse_format="json_dict",
                is_out_of_scope=(len(tool_calls) == 0),
            )
        elif isinstance(parsed_json, list):
            tool_calls = [
                {"name": c["name"].strip(), "arguments": c.get("arguments", {})}
                for c in parsed_json
                if isinstance(c, dict) and "name" in c
            ]
            return LFM3ParsedOutput(
                raw_text=raw_text,
                thinking_content=thinking_content,
                tool_calls=tool_calls,
                parse_format="json_list",
                is_out_of_scope=(len(tool_calls) == 0),
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 5. Fallback: Kiểm tra xem có phải yêu cầu ngoài phạm vi (Out-of-scope) hay không
    out_of_scope_cues = [
        "không có ứng dụng",
        "chưa được cài đặt",
        "ngoài phạm vi",
        "không thể thực hiện",
        "tool_calls: []",
        "\"tool_calls\": []",
    ]
    is_out = any(cue in text.lower() for cue in out_of_scope_cues)

    return LFM3ParsedOutput(
        raw_text=raw_text,
        thinking_content=thinking_content,
        tool_calls=[],
        is_out_of_scope=is_out,
        parse_format="none",
        error_message="Không trích xuất được tool call hợp lệ từ phản hồi",
    )
