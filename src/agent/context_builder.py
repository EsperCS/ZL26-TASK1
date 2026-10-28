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

import json
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
        context: dict[str, Any] | None = None,
        transcript: str = "",
        history: list[dict[str, Any]] | None = None,
        thinking_mode: bool = False,
    ) -> str:
        """Mẫu SYSTEM PROMPT đóng vai trò Luật Cứng & Định dạng Đầu ra."""
        thinking_instruction = (
            "Trong phần <think>, hãy kiểm tra nhanh: (1) Tên thực thể/ID trong hồ sơ, "
            "(2) Ngày giờ quy đổi từ bảng lịch, (3) Điều kiện x-one-of/x-ref trong dưới 150 từ, "
            "sau đó đóng thẻ </think> và xuất thẳng JSON."
            if thinking_mode
            else "Tuyệt đối không dùng thẻ <think>, giữ trường `brief_plan` (15–20 từ) ở đầu JSON "
            "làm bước đệm suy luận (Micro Chain-of-Thought), sau đó xuất `tool_calls`."
        )

        return (
            "Bạn là LifeMate — trợ lý ảo điều hành công cụ thông minh. Nhiệm vụ của bạn là phân tích "
            "yêu cầu bằng giọng nói của người dùng (đã qua nhận diện ASR) cùng ngữ cảnh cá nhân để gọi "
            "chính xác chuỗi hàm (tools) tương ứng làm thay đổi trạng thái hệ thống (ToolEnv).\n\n"
            "### QUY TẮC BẮT BUỘC (HARD RULES):\n"
            "1. XỬ LÝ NHIỄU ASR & TÊN RIÊNG (ENTITY GROUNDING):\n"
            "   - Văn bản đầu vào được chuyển từ giọng nói nên có thể sai chính tả hoặc đồng âm (ví dụ: \"sếp Dụng\" -> \"Sếp Dũng\").\n"
            "   - BẮT BUỘC đối chiếu và lấy chính xác 100% tên người, số điện thoại, email, địa điểm hoặc ID sự kiện từ [USER PROFILE], [ENTITY HINTS] hoặc [CONVERSATION HISTORY]. Tuyệt đối không tự bịa (hallucinate) ID hoặc tên không tồn tại.\n\n"
            "2. QUY CHIẾU THỜI GIAN TUYỆT ĐỐI (DATE & TIME RESOLUTION):\n"
            "   - Dựa vào [CURRENT TIME] và [CALENDAR TABLE 21 DAYS] được cung cấp sẵn để quy đổi các từ chỉ thời gian tương đối (\"hôm nay\", \"ngày mai\", \"thứ sáu tuần này\", \"tuần sau\", \"9 rưỡi tối\") sang định dạng chuẩn của schema (ví dụ: YYYY-MM-DD, HH:MM).\n"
            "   - 'sáng' = AM (ví dụ: 'chín giờ sáng' = 09:00:00, TUYỆT ĐỐI KHÔNG nhầm thành 18:00 hay 06:00). 'chiều' / 'tối' / 'đêm' = PM ('sáu giờ chiều' = 18:00:00, 'mười một giờ đêm' = 23:00:00).\n"
            "   - Khẩu ngữ địa phương: 'chiều ni' / 'sáng ni' = chiều nay / sáng nay. Không tự nhẩm tính thứ/ngày ngoài bảng lịch đã cho.\n\n"
            "3. XỬ LÝ KHẨU NGỮ, ĐỔI Ý (SELF-CORRECTION) & ĐIỀU KIỆN:\n"
            "   - Đổi ý: Khi người dùng nói 'làm A, à thôi B đi' hoặc 'chọn X, à thôi Y' -> BỎ QUA A/X, CHỈ THỰC HIỆN B/Y.\n"
            "   - Điều kiện ('nếu... thì...'): Gọi hàm truy vấn/kiểm tra trước (list/get), sau đó thực thi hành động tương ứng.\n\n"
            "4. RÀNG BUỘC THIẾU THÔNG TIN (sys.ask_missing_info):\n"
            "   - Nếu yêu cầu của người dùng thiếu tham số bắt buộc không thể tự suy đoán hoặc tra cứu được: BẮT BUỘC gọi `sys.ask_missing_info` với `tool_name` (tên tool dự định gọi) và `params` (danh sách tham số còn thiếu).\n\n"
            "5. QUY TẮC ĐỌC TRƯỚC - GHI SAU (READ-BEFORE-WRITE MULTI-STEP):\n"
            "   - Khi cần thông tin từ môi trường (ID thuốc, ID sự kiện, ID liên hệ, ID nhà hàng/món ăn): BẮT BUỘC chỉ gọi công cụ ĐỌC (như `.list`, `.search`, `.get`) trước.\n"
            "   - Hệ thống sẽ trả kết quả trong `[OBSERVATION]`. Lượt tiếp theo, bạn mới dùng ID thực tế đó để gọi công cụ ghi. Tuyệt đối KHÔNG tự bịa ID hoặc viết placeholder dạng `<sẽ lấy từ...>`.\n\n"
            "6. TUÂN THỦ NGHIÊM NGẶT RÀNG BUỘC SCHEMA & PHẠM VI ỨNG DỤNG:\n"
            "   - `required`: Phải điền đầy đủ mọi tham số bắt buộc. Tuyệt đối không truyền chuỗi rỗng \"\" hoặc null cho tham số bắt buộc.\n"
            "   - `x-one-of`: Chỉ được phép truyền ĐÚNG MỘT trong các nhóm tham số được liệt kê trong ràng buộc `x-one-of`, tuyệt đối không truyền đồng thời cả hai.\n"
            "   - `x-ref`: Khi một tham số yêu cầu tham chiếu (`x-ref`) tới đối tượng khác (như `contact_id`, `event_id`, `alarm_id`), phải lấy đúng ID từ [USER PROFILE], kết quả trả về của tool trước đó trong [CONVERSATION HISTORY], hoặc gọi hàm tìm kiếm/liệt kê trước khi cập nhật/xóa.\n"
            "   - Yêu cầu ngoài phạm vi: Nếu người dùng yêu cầu hành động thuộc ứng dụng chưa cài đặt trên thiết bị, BẮT BUỘC xuất ra danh sách rỗng: `\"tool_calls\": []`.\n\n"
            "7. THỨ TỰ & ĐỊNH DẠNG ĐẦU RA:\n"
            "   - Chỉ gọi các hàm thực sự cần thiết theo đúng thứ tự logic. Nếu yêu cầu có nhiều ý độc lập, hãy xuất đầy đủ các tool calls tương ứng.\n"
            "   - TUYỆT ĐỐI KHÔNG SUY LUẬN BẰNG TIẾNG ANH (NO English CoT monologue). BẮT BUỘC chỉ xuất ra đúng khối JSON, không giải thích lan man ngoài JSON:\n"
            "```json\n"
            "{\n"
            "  \"brief_plan\": \"<Tóm tắt ngắn gọn: dùng hàm nào, ngày giờ gì quy đổi ra sao>\",\n"
            "  \"tool_calls\": [\n"
            "    {\n"
            "      \"name\": \"<tên_hàm_chính_xác>\",\n"
            "      \"arguments\": {\n"
            "        \"<tên_tham_số>\": \"<giá_trị>\"\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            f"{thinking_instruction}"
        )

    def build_tools_schema(self, installed_apps: dict[str, Any]) -> list[dict[str, Any]]:
        """Trích xuất danh sách tool schemas chuẩn định dạng function calling cho apply_chat_template."""
        apps_in_catalog = {a["id"]: a for a in self.tools_catalog.get("apps", [])}
        tools: list[dict[str, Any]] = []

        # Luôn thêm sys.ask_missing_info
        tools.append({
            "type": "function",
            "function": {
                "name": "sys.ask_missing_info",
                "description": "Hỏi lại người dùng khi thiếu thông tin bắt buộc để thực hiện tác vụ.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "Tên tool dự định gọi"},
                        "params": {"type": "array", "items": {"type": "string"}, "description": "Danh sách tham số còn thiếu"},
                    },
                    "required": ["tool_name", "params"],
                },
            },
        })

        for app_id, app_info in installed_apps.items():
            app_spec = apps_in_catalog.get(app_id)
            if not app_spec:
                continue

            tool_names_allowed = set(app_info.get("tools", []))
            for tool in app_spec.get("tools", []):
                t_name = tool.get("name")
                if t_name not in tool_names_allowed:
                    continue

                tools.append({
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                })

        return tools

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
                    if p_type == "array" and "items" in p_spec:
                        items_spec = p_spec["items"]
                        i_type = items_spec.get("type", "string")
                        if i_type == "object" and "properties" in items_spec:
                            i_props = items_spec.get("properties", {})
                            i_req = items_spec.get("required", [])
                            i_fields = [f"{k}: {v.get('type')}{' [BẮT BUỘC]' if k in i_req else ''}" for k, v in i_props.items()]
                            extra.append(f"danh sách đối tượng [{{{', '.join(i_fields)}}}]")
                        else:
                            extra.append(f"danh sách {i_type}")
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

    def build_user_prompt(
        self,
        sample: dict[str, Any],
        transcript_override: str | None = None,
        alt_transcript: str | None = None,
    ) -> str:
        """Định dạng User Prompt theo các khối [BLOCK] rõ ràng, nạp ngữ cảnh động cho từng mẫu."""
        transcript = transcript_override or sample.get("audio_text", "")
        context = sample.get("context", {})
        current_time = context.get("current_time", datetime.now().isoformat())
        user_profile = context.get("user_profile", {})
        installed_apps = context.get("installed_apps", {})
        history = sample.get("history", [])

        # 1. [CURRENT TIME]
        try:
            curr_dt = date_parser.isoparse(current_time)
            weekday_str = VI_WEEKDAYS[curr_dt.date().weekday()]
            time_display = f"{curr_dt.strftime('%Y-%m-%d %H:%M:%S')} ({weekday_str})"
        except Exception:  # noqa: BLE001
            time_display = str(current_time)

        # 2. [CALENDAR TABLE 21 DAYS (Tra cứu nhanh ngày tháng)]
        calendar_table = build_calendar_table(current_time)

        # 3. [USER PROFILE & DEVICE CONTEXT]
        profile_state: dict[str, Any] = {}
        if user_profile:
            profile_state["user_profile"] = user_profile
        if "location" in context:
            profile_state["gps_location"] = context["location"]
        if "timezone" in context:
            profile_state["timezone"] = context["timezone"]
        profile_state["installed_apps"] = list(installed_apps.keys())
        for key in ["initial_state", "env_state", "state"]:
            if key in context:
                profile_state[key] = context[key]
        user_profile_json = json.dumps(profile_state, ensure_ascii=False, indent=2)

        # 4. [CONVERSATION HISTORY]
        if history:
            # Lấy tối đa 4 lượt gần nhất để tránh tràn context và lan man
            recent_history = history[-4:]
            start_turn = max(1, len(history) - len(recent_history) + 1)
            history_lines = []
            for idx, turn in enumerate(recent_history, start_turn):
                role = turn.get("role", "user")
                text = turn.get("text") or turn.get("audio_text") or ""
                tcs = turn.get("tool_calls", [])
                trs = turn.get("tool_results", [])
                line = f"- Lượt {idx} ({role}): \"{text}\""
                if tcs:
                    line += f"\n  Tool Calls: {json.dumps(tcs, ensure_ascii=False)}"
                if trs:
                    # Rút gọn các mảng kết quả quá dài để tiết kiệm context
                    trimmed_trs = []
                    for r in trs:
                        if isinstance(r, dict):
                            r_copy = dict(r)
                            for k, v in list(r_copy.items()):
                                if isinstance(v, list) and len(v) > 3:
                                    r_copy[k] = v[:3] + [f"...(còn {len(v)-3} mục nữa)"]
                            trimmed_trs.append(r_copy)
                        else:
                            trimmed_trs.append(r)
                    line += f"\n  Tool Results: {json.dumps(trimmed_trs, ensure_ascii=False)}"
                history_lines.append(line)
            history_str = "\n".join(history_lines)
        else:
            history_str = "Không có lịch sử trước đó."

        # 5. [AVAILABLE TOOLS SCHEMA]
        tools_spec = self._format_allowed_tools(installed_apps)

        # 6. [ENTITY HINTS (Gợi ý khớp tên từ Profile/History bằng Fuzzy Match)]
        entity_hints_str = "Không có gợi ý thực thể."
        if self.enable_fuzzy_grounding and transcript:
            groundings = fuzzy_entity_grounding(transcript, user_profile, history)
            if groundings:
                hints = [
                    f"- Nghe trong audio: \"{g[0]}\" ──► Thực thể chuẩn hồ sơ: \"{g[1]}\" (Khớp {g[2]}%)"
                    for g in groundings[:5]
                ]
                entity_hints_str = "\n".join(hints)

        # 7. [USER VOICE COMMAND (ASR Transcript)]
        alt_str = f'\nPhonetic_Alt (2-best): "{alt_transcript}"' if alt_transcript else ""

        return (
            f"[CURRENT TIME]\n{time_display}\n\n"
            f"[CALENDAR TABLE 21 DAYS (Tra cứu nhanh ngày tháng)]\n{calendar_table}\n\n"
            f"[USER PROFILE & INITIAL STATE]\n{user_profile_json}\n\n"
            f"[CONVERSATION HISTORY]\n{history_str}\n\n"
            f"[AVAILABLE TOOLS SCHEMA]\n{tools_spec}\n\n"
            f"[ENTITY HINTS (Gợi ý khớp tên từ Profile/History bằng Fuzzy Match)]\n{entity_hints_str}\n\n"
            f"[USER VOICE COMMAND (ASR Transcript)]\n"
            f'Primary Transcript: "{transcript}"'
            f"{alt_str}\n\n"
            "Hãy phân tích và xuất JSON `tool_calls` chính xác nhất."
        )

    def build_messages(
        self,
        sample: dict[str, Any],
        transcript_override: str | None = None,
        thinking_mode: bool = False,
        alt_transcript: str | None = None,
    ) -> list[dict[str, Any]]:
        """Xây dựng chuỗi messages chuẩn ChatML: 1 lượt system + 1 lượt user theo blocks."""
        system_content = self.build_system_prompt(
            context=sample.get("context", {}),
            transcript=transcript_override or sample.get("audio_text", ""),
            history=sample.get("history", []),
            thinking_mode=thinking_mode,
        )
        user_content = self.build_user_prompt(
            sample=sample,
            transcript_override=transcript_override,
            alt_transcript=alt_transcript,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


def build_self_correction_prompt(
    tool_name: str,
    error_message: str,
) -> str:
    """Prompt sửa lỗi tự động (SELF-CORRECTION PROMPT) dùng khi Validate-Before-Call bắt được lỗi."""
    return (
        "[VALIDATION ERROR]\n"
        "Lệnh gọi hàm vừa rồi của bạn không hợp lệ với lỗi cụ thể sau:\n"
        f"- Hàm `{tool_name}`: {error_message}\n\n"
        "Hãy kiểm tra lại điều kiện `required` / `x-one-of` / `x-ref` và xuất lại khối JSON `tool_calls` đã sửa lỗi."
    )


def build_prompt_for_sample(
    sample: dict[str, Any],
    tools_catalog: dict[str, Any],
    transcript_override: str | None = None,
    thinking_mode: bool = False,
) -> list[dict[str, Any]]:
    """Helper thuận tiện xây dựng danh sách messages từ 1 record."""
    builder = ContextBuilder(tools_catalog=tools_catalog)
    return builder.build_messages(
        sample=sample,
        transcript_override=transcript_override,
        thinking_mode=thinking_mode,
    )
