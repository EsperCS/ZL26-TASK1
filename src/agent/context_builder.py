"""src/agent/context_builder.py.

Context Engineering & Prompt Construction Module for Task 1 (LifeMate).
Tối ưu hóa ngữ cảnh suy luận cho Small Language Models (< 5B parameters):
1. Bảng quy chiếu lịch tuyệt đối 21 ngày (Pre-computed Calendar Table):
   Tính sẵn 7 ngày quá khứ + 14 ngày tương lai từ `current_time` để SLM tra cứu chính xác ngày tháng.
2. Fuzzy Entity Grounding:
   Đối chiếu thực thể ASR nghe được với `user_profile` và `history` bằng RapidFuzz (khoảng cách âm học/chữ cái).
3. Đóng gói System Prompt & Tool Schemas chuẩn JSON Schema theo format ChatML / Hermes Tool Calling.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from dateutil import parser as date_parser
from rapidfuzz import fuzz

VI_WEEKDAYS = [
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
]


def build_calendar_table(current_time_str: str) -> str:
    """Tự động tính sẵn bảng quy chiếu lịch 21 ngày (7 ngày trước + 14 ngày tới).

    Giúp SLM tra cứu tức thì các mốc thời gian tương đối như:
    - 'hôm nay', 'ngày mai', 'ngày kia', 'hôm qua'
    - 'Thứ Sáu tuần này', 'Thứ Bảy tuần sau', 'Thứ Hai tuần trước'
    """
    try:
        curr_dt = date_parser.isoparse(current_time_str)
    except (ValueError, TypeError):
        # Fallback nếu parse ISO thất bại
        curr_dt = datetime.now()

    curr_date = curr_dt.date()
    curr_weekday = curr_date.weekday()  # 0: Thứ Hai, 6: Chủ Nhật

    # Tìm ngày Thứ Hai của tuần hiện tại
    start_of_current_week = curr_date - timedelta(days=curr_weekday)

    lines = [
        "| Mốc thời gian tương đối | Thứ trong tuần | Ngày chuẩn (YYYY-MM-DD) |",
        "| :--- | :--- | :--- |",
    ]

    # Duyệt từ -7 ngày đến +13 ngày (tổng 21 ngày)
    for offset in range(-7, 14):
        target_date = curr_date + timedelta(days=offset)
        target_weekday = VI_WEEKDAYS[target_date.weekday()]
        iso_str = target_date.strftime("%Y-%m-%d")

        # Xác định nhãn tương đối
        if offset == 0:
            rel_label = "**Hôm nay (current_time)**"
        elif offset == 1:
            rel_label = "Ngày mai"
        elif offset == 2:
            rel_label = "Ngày kia"
        elif offset == -1:
            rel_label = "Hôm qua"
        elif offset == -2:
            rel_label = "Hôm kia"
        else:
            # So sánh theo tuần
            target_week_start = target_date - timedelta(days=target_date.weekday())
            week_diff = (target_week_start - start_of_current_week).days // 7

            if week_diff == 0:
                rel_label = f"{target_weekday} tuần này"
            elif week_diff == 1:
                rel_label = f"{target_weekday} tuần sau"
            elif week_diff == 2:
                rel_label = f"{target_weekday} tuần sau nữa"
            elif week_diff == -1:
                rel_label = f"{target_weekday} tuần trước"
            else:
                rel_label = f"{target_weekday} ({offset:+d} ngày)"

        lines.append(f"| {rel_label} | {target_weekday} | `{iso_str}` |")

    return "\n".join(lines)


def extract_entities_from_profile_and_history(
    user_profile: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> set[str]:
    """Trích xuất toàn bộ các thực thể đã biết từ user_profile và conversation history."""
    known_entities: set[str] = set()

    # 1. Từ user_profile
    if user_profile:
        for val in user_profile.values():
            if isinstance(val, str) and len(val.strip()) > 1:
                # Thêm cả câu và các phân đoạn dấu phẩy
                known_entities.add(val.strip())
                for part in re.split(r"[,–-]", val):
                    part_str = part.strip()
                    if len(part_str) > 2:
                        known_entities.add(part_str)

    # 2. Từ conversation history
    if history:
        for turn in history:
            text = turn.get("text") or turn.get("audio_text") or ""
            # Trích xuất các cụm từ viết hoa (proper nouns)
            capitalized = re.findall(r"\b[A-ZĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zđàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ0-9_]*(?:\s+[A-ZĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zđàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ0-9_]*)*\b", text)
            for cap in capitalized:
                if len(cap.strip()) > 2 and cap.lower() not in {"thứ", "ngày", "tháng", "năm", "hôm"}:
                    known_entities.add(cap.strip())

            # Trích xuất từ arguments của tool_calls trong history
            for tc in turn.get("tool_calls", []):
                args = tc.get("arguments", {})
                if isinstance(args, dict):
                    for v in args.values():
                        if isinstance(v, str) and len(v.strip()) > 2:
                            known_entities.add(v.strip())

    return known_entities


def fuzzy_entity_grounding(
    transcript: str,
    user_profile: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    threshold: float = 78.0,
) -> list[tuple[str, str, float]]:
    """Đối chiếu các từ ngữ trong câu transcript với tập thực thể đã biết bằng RapidFuzz.

    Trả về danh sách (nghe_nhầm, thực_thể_chuẩn, điểm_tương_đồng).
    """
    known_entities = extract_entities_from_profile_and_history(user_profile, history)
    if not known_entities or not transcript:
        return []

    # Tìm các n-grams (độ dài 1-4 từ) viết hoa hoặc từ khóa quan trọng trong transcript
    words = transcript.split()
    candidates: set[str] = set()

    for n in range(1, 5):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i : i + n]).strip(",.?!:;")
            if len(chunk) > 2:
                candidates.add(chunk)

    matches: list[tuple[str, str, float]] = []
    used_canonicals: set[str] = set()

    for cand in candidates:
        for ent in known_entities:
            if ent in used_canonicals or cand.lower() == ent.lower():
                continue

            # Đo độ tương đồng token sort ratio & partial ratio
            score_sort = fuzz.token_sort_ratio(cand, ent)
            score_ratio = fuzz.ratio(cand, ent)
            best_score = max(score_sort, score_ratio)

            if best_score >= threshold and len(cand) >= 3 and len(ent) >= 3:
                # Tránh match các từ quá ngắn hoặc quá lệch độ dài
                len_ratio = min(len(cand), len(ent)) / max(len(cand), len(ent))
                if len_ratio >= 0.5:
                    matches.append((cand, ent, round(float(best_score), 1)))
                    used_canonicals.add(ent)
                    break

    # Sắp xếp theo score giảm dần
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


class ContextBuilder:
    """Bộ tạo ngữ cảnh hoàn chỉnh cho Agent LifeMate."""

    def __init__(
        self,
        tools_catalog: dict[str, Any] | None = None,
        include_calendar_table: bool = True,
        enable_fuzzy_grounding: bool = True,
    ) -> None:
        self.tools_catalog = tools_catalog or {}
        self.include_calendar_table = include_calendar_table
        self.enable_fuzzy_grounding = enable_fuzzy_grounding

    def build_system_prompt(
        self,
        context: dict[str, Any],
        transcript: str = "",
        history: list[dict[str, Any]] | None = None,
        thinking_mode: bool = False,
    ) -> str:
        """Sinh System Prompt chuẩn mực cho SLM kèm bảng lịch và công cụ khả dụng."""
        current_time = context.get("current_time", datetime.now().isoformat())
        timezone_str = context.get("timezone", "Asia/Ho_Chi_Minh")
        user_profile = context.get("user_profile", {})
        installed_apps = context.get("installed_apps", {})

        # 1. Bảng lịch 21 ngày
        calendar_section = ""
        if self.include_calendar_table:
            cal_table = build_calendar_table(current_time)
            calendar_section = (
                f"\n### BẢNG QUY CHIẾU LỊCH 21 NGÀY (Từ current_time: `{current_time}`):\n"
                f"{cal_table}\n"
            )

        # 2. Gợi ý Entity Grounding
        entity_hints_section = ""
        if self.enable_fuzzy_grounding and transcript:
            groundings = fuzzy_entity_grounding(transcript, user_profile, history)
            if groundings:
                hints_text = "\n".join([f"- Nghe trong audio: \"{g[0]}\" ──► Thực thể chuẩn hồ sơ: **\"{g[1]}\"** (Khớp {g[2]}%)" for g in groundings[:5]])
                entity_hints_section = (
                    f"\n### GỢI Ý THỰC THỂ KHỚP HỒ SƠ (ASR-Entity Grounding):\n"
                    f"{hints_text}\n"
                    "Lưu ý: Hãy ưu tiên sử dụng tên thực thể chuẩn hồ sơ khi điền tham số Tool.\n"
                )

        # 3. Danh mục công cụ được phép gọi
        allowed_tools_spec = self._format_allowed_tools(installed_apps)

        thinking_guideline = (
            "### HƯỚNG DẪN ĐỊNH DẠNG (FORMAT GUIDELINES):\n"
            "- BẠN ĐƯỢC PHÉP suy luận từng bước bên trong thẻ `<think>...</think>` trước khi xuất kết quả để phân tích yêu cầu, kiểm tra lịch và sửa lỗi nghe nhầm của ASR.\n"
            "- Sau thẻ `</think>`, bắt buộc kết thúc bằng DUY NHẤT một khối JSON chứa `tool_calls`.\n"
            if thinking_mode
            else "### HƯỚNG DẪN ĐỊNH DẠNG (FORMAT GUIDELINES):\n"
            "- Tuyệt đối KHÔNG suy luận, không dùng thẻ `<think>`, không thêm bất kỳ văn bản nào ngoài khối JSON.\n"
            "- Xuất DUY NHẤT một khối JSON chứa `tool_calls`.\n"
        )

        return (
            "Bạn là Trợ lý cuộc sống thông minh LifeMate (Zalo AI Challenge 2026).\n"
            "Nhiệm vụ của bạn là lắng nghe yêu cầu bằng giọng nói của người dùng tiếng Việt, lập kế hoạch và chuyển thành chuỗi các lệnh gọi công cụ (tool_calls) chính xác.\n"
            "\n"
            "### QUY TẮC BẮT BUỘC (CRITICAL RULES):\n"
            "1. CHỈ ĐƯỢC GỌI CÁC CÔNG CỤ THUỘC ỨNG DỤNG ĐÃ CÀI ĐẶT (Installed Apps) hoặc `sys.ask_missing_info`.\n"
            "2. NẾU YÊU CẦU NẰM NGOÀI PHẠM VI (Out-of-scope) hoặc ứng dụng liên quan chưa cài đặt, trả về danh sách rỗng: `\"tool_calls\": []`.\n"
            "3. THAM SỐ BẮT BUỘC (required): Phải điền đầy đủ. Tuyệt đối không truyền chuỗi rỗng `\"\"` hoặc `null` cho tham số bắt buộc.\n"
            "4. RÀNG BUỘC `x-one-of`: Phải chọn ĐÚNG 1 nhóm tham số được phép trong nhóm (ví dụ: `contact_id` HOẶC `phone_number`, không được truyền cả hai và không được bỏ trống cả hai).\n"
            "5. RÀNG BUỘC `x-ref`: Tuyệt đối không bịa đặt ID (như `event_id`, `device_id`, `contact_id`). Nếu cần thao tác mà chưa có ID, hãy gọi tool tìm kiếm/đọc (`list`, `get`, `search`) trước.\n"
            "6. THIẾU THÔNG TIN: Nếu người dùng yêu cầu hành động nhưng thiếu tham số cốt lõi mà không thể tra cứu, hãy gọi `sys.ask_missing_info`.\n"
            "7. ĐỊNH DẠNG THỜI GIAN: Bắt buộc tuân thủ format (date: `YYYY-MM-DD`, date-time: `YYYY-MM-DDTHH:MM:SS+07:00`, time: `HH:MM`).\n"
            "\n"
            f"{thinking_guideline}"
            "\n### ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT):\n"
            "Bạn phải xuất ra DUY NHẤT một khối JSON hợp lệ theo cấu trúc sau:\n"
            "```json\n"
            "{\n"
            "  \"brief_plan\": \"1 câu tóm tắt kế hoạch ngắn gọn\",\n"
            "  \"tool_calls\": [\n"
            "    {\"name\": \"app_name.tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n"
            "  ]\n"
            "}\n"
            "```\n"
            f"\n### THÔNG TIN BỐI CẢNH NGƯỜI DÙNG:\n"
            f"- Thời gian hiện tại: `{current_time}` (Múi giờ: `{timezone_str}`)\n"
            f"- Hồ sơ người dùng: {user_profile}\n"
            f"{calendar_section}"
            f"{entity_hints_section}"
            f"\n### CÁC CÔNG CỤ KHẢ DỤNG:\n"
            f"{allowed_tools_spec}\n"
        )

    def _format_allowed_tools(self, installed_apps: dict[str, Any]) -> str:
        """Định dạng mô tả schema của các tool thuộc các app đã cài đặt."""
        apps_in_catalog = {a["id"]: a for a in self.tools_catalog.get("apps", [])}
        output_blocks = []

        # Luôn thêm sys.ask_missing_info
        output_blocks.append(
            "Tool: `sys.ask_missing_info` (Hệ thống)\n"
            "  Mô tả: Hỏi lại người dùng khi thiếu thông tin bắt buộc để thực hiện tác vụ.\n"
            "  Tham số bắt buộc: `tool_name` (string, tên tool dự định gọi), `params` (list[string], danh sách tham số còn thiếu)."
        )

        for app_id, app_info in installed_apps.items():
            app_spec = apps_in_catalog.get(app_id)
            if not app_spec:
                continue

            app_title = app_spec.get("name_vi", app_id)
            tool_names_allowed = set(app_info.get("tools", []))

            for tool in app_spec.get("tools", []):
                t_name = tool.get("name")
                if t_name not in tool_names_allowed:
                    continue

                desc = tool.get("description", "")
                params_obj = tool.get("parameters", {})
                props = params_obj.get("properties", {})
                required = params_obj.get("required", [])
                x_one_of = tool.get("x-one-of") or params_obj.get("x-one-of", [])

                param_lines = []
                for p_name, p_spec in props.items():
                    p_type = p_spec.get("type", "any")
                    p_desc = p_spec.get("description", "")
                    p_enum = p_spec.get("enum")
                    p_fmt = p_spec.get("format")
                    p_ref = p_spec.get("x-ref")

                    req_tag = " [BẮT BUỘC]" if p_name in required else " [tùy chọn]"
                    extra = []
                    if p_enum:
                        extra.append(f"chỉ chọn 1 trong {p_enum}")
                    if p_fmt:
                        extra.append(f"format: {p_fmt}")
                    if p_ref:
                        extra.append(f"ref từ: {p_ref}")

                    extra_str = f" ({'; '.join(extra)})" if extra else ""
                    param_lines.append(f"    - `{p_name}` ({p_type}){req_tag}: {p_desc}{extra_str}")

                one_of_str = ""
                if x_one_of:
                    one_of_str = f"\n  Ràng buộc một trong (x-one-of): Cần đúng 1 trong các nhóm {x_one_of}"

                output_blocks.append(
                    f"Tool: `{t_name}` ({app_title})\n"
                    f"  Mô tả: {desc}\n"
                    f"  Tham số:\n"
                    + "\n".join(param_lines)
                    + one_of_str
                )

        return "\n\n".join(output_blocks)

    def build_messages(
        self,
        sample: dict[str, Any],
        transcript_override: str | None = None,
        thinking_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """Xây dựng chuỗi messages chuẩn ChatML cho toàn bộ cuộc hội thoại của sample."""
        transcript = transcript_override or sample.get("audio_text", "")
        context = sample.get("context", {})
        history = sample.get("history", [])

        # 1. System Prompt
        system_content = self.build_system_prompt(
            context=context,
            transcript=transcript,
            history=history,
            thinking_mode=thinking_mode,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # 2. History turns
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("text") or turn.get("audio_text") or ""

            # Nếu là lượt của assistant có tool_calls
            if role == "assistant" and "tool_calls" in turn:
                tc_data = {
                    "tool_calls": turn.get("tool_calls", []),
                    "tool_results": turn.get("tool_results", []),
                }
                messages.append({
                    "role": "assistant",
                    "content": f"{content}\n[Thực thi trước đó]: {tc_data}".strip(),
                })
            else:
                messages.append({"role": role, "content": content})

        # 3. Lượt hỏi hiện tại của người dùng
        messages.append({
            "role": "user",
            "content": f"Yêu cầu bằng giọng nói của người dùng: \"{transcript}\"",
        })

        return messages


def build_prompt_for_sample(
    sample: dict[str, Any],
    tools_catalog: dict[str, Any],
    transcript_override: str | None = None,
) -> list[dict[str, Any]]:
    """Helper thuận tiện xây dựng danh sách messages từ 1 record."""
    builder = ContextBuilder(tools_catalog=tools_catalog)
    return builder.build_messages(sample, transcript_override=transcript_override)
