"""src/eval/zalo_bfcl_evaluator.py.

Zalo-BFCL Evaluator Module (Chuẩn đánh giá kết hợp BFCL v4 + Zalo ToolEnv State Hash):
Đo lường 6 chỉ số định lượng khắt khe cho bài toán LifeMate trên tập kiểm thử (val_500):
1. `State Hash Match (%)`: Tỷ lệ khớp mã băm SHA-256 sau khi Replay trên ToolEnv (Quyết định điểm số).
2. `AST Tool Selection F1 (%)`: Độ chính xác chọn tên tool và số lượng tool cần gọi.
3. `Schema Adherence Rate (%)`: Tỷ lệ tool calls vượt qua bộ kiểm duyệt Pydantic/zaic_tools (required, x-one-of, enum, format).
4. `Argument Exact Match (%)`: Tỷ lệ tham số khớp tuyệt đối (non-fuzzy) và fuzzy Token-F1 > 0.6.
5. `Avg Thinking Tokens & Latency (ms/sample)`: Chi phí token suy luận <think> và độ trễ thực thi.
6. `Total Params vs Active Params`: Kiểm soát trần ngân sách tham số < 6.0B (ASR 0.433B + SLM < 6B).
"""

from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Dynamic path resolution for zaic_tools package
try:
    from zaic_tools.utils import _tokens
    from zaic_tools.validate import catalog_tools, schema_error
except ImportError:
    for candidate in [Path("train"), Path("public_test"), Path("train+public_test/train")]:
        if (candidate / "zaic_tools").is_dir() and str(candidate.resolve()) not in sys.path:
            sys.path.insert(0, str(candidate.resolve()))
    from zaic_tools.utils import _tokens
    from zaic_tools.validate import catalog_tools, schema_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASR_WINNER_PARAMS_B = 0.433  # ZipFormer (0.153B) + Capu (0.278B) + VAD (0.002B)
MAX_SYSTEM_PARAMS_B = 6.000


# ============================================================================
# 1. PYDANTIC SCHEMAS CHO TOOL CALLING ĐẦU RA
# ============================================================================

class ToolCallItem(BaseModel):
    """Mỗi lệnh gọi công cụ riêng lẻ."""
    name: str = Field(description="Tên đầy đủ của tool, ví dụ: 'smart_home.light_set'")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Từ điển tham số của tool")


class AgentModelResponse(BaseModel):
    """Cấu trúc phản hồi đầy đủ của SLM."""
    brief_plan: str | None = Field(default=None, description="Kế hoạch suy luận ngắn (Chain-of-Thought)")
    tool_calls: list[ToolCallItem] = Field(default_factory=list, description="Danh sách các tool calls thực thi")


@dataclass(slots=True)
class ParsedThinkingOutput:
    """Kết quả bóc tách phần suy luận <think> ra khỏi dữ liệu JSON."""
    raw_output: str
    reasoning_content: str
    clean_text: str
    thinking_token_count: int
    brief_plan: str | None
    tool_calls: list[dict[str, Any]]
    parse_success: bool
    error_message: str | None = None


# ============================================================================
# 2. PARSER SUY LUẬN & BÓC TÁCH JSON
# ============================================================================

def parse_thinking_and_tool_calls(raw_text: str) -> ParsedThinkingOutput:
    """Bóc tách phần suy luận bên trong `<think>...</think>` ra khỏi phần JSON tool_calls.

    Đảm bảo khi bật thinking_mode=True, mô hình tự do suy luận mà không làm vỡ parser JSON.
    """
    raw_text = raw_text.strip()
    reasoning_content = ""
    clean_text = raw_text

    # 1. Bóc tách <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        clean_text = (raw_text[: think_match.start()] + raw_text[think_match.end() :]).strip()
    elif "<think>" in raw_text:
        # Trường hợp generation bị cắt ngang chưa đóng </think>
        parts = raw_text.split("<think>", 1)
        clean_text = parts[0].strip()
        reasoning_content = parts[1].strip()

    # Ước lượng số token của thinking (khoảng 1.3 token / từ đối với tiếng Việt/Anh hỗn hợp)
    thinking_token_count = int(len(reasoning_content.split()) * 1.3) if reasoning_content else 0

    # 2. Bóc tách JSON tool_calls từ clean_text
    brief_plan: str | None = None
    tool_calls: list[dict[str, Any]] = []
    parse_success = False
    error_message: str | None = None

    # Tìm khối ```json ... ```
    json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    candidate_json = json_block_match.group(1).strip() if json_block_match else clean_text

    # Nếu không thấy block, thử tìm cặp ngoặc nhọn hoặc vuông ngoài cùng
    if not (candidate_json.startswith(("{", "["))):
        bracket_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate_json)
        if bracket_match:
            candidate_json = bracket_match.group(1).strip()

    try:
        parsed = json.loads(candidate_json)
        if isinstance(parsed, dict):
            brief_plan = parsed.get("brief_plan")
            raw_calls = parsed.get("tool_calls", [])
            if isinstance(raw_calls, list):
                tool_calls = [c for c in raw_calls if isinstance(c, dict) and "name" in c]
                parse_success = True
            else:
                error_message = f"Trường 'tool_calls' không phải là list: {type(raw_calls)}"
        elif isinstance(parsed, list):
            # Mô hình sinh trực tiếp mảng tool_calls [...]
            tool_calls = [c for c in parsed if isinstance(c, dict) and "name" in c]
            parse_success = True
        else:
            error_message = f"JSON root không phải dict hoặc list: {type(parsed)}"
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        error_message = f"Lỗi parse JSON: {e}"

    return ParsedThinkingOutput(
        raw_output=raw_text,
        reasoning_content=reasoning_content,
        clean_text=clean_text,
        thinking_token_count=thinking_token_count,
        brief_plan=brief_plan,
        tool_calls=tool_calls,
        parse_success=parse_success,
        error_message=error_message,
    )


# ============================================================================
# 3. CÁC ĐỘ ĐO BENCHMARK BFCL V4 + ZALO
# ============================================================================

@dataclass(slots=True)
class SampleBFCLMetrics:
    """Kết quả đo 6 chỉ số trên 1 mẫu đơn lẻ."""
    sample_id: str
    state_hash_match: bool
    ast_tool_f1: float
    ast_precision: float
    ast_recall: float
    schema_adherence_rate: float
    schema_valid_calls: int
    schema_total_calls: int
    arg_exact_match_rate: float
    thinking_tokens: int
    latency_ms: float
    error_note: str | None = None


@dataclass(slots=True)
class ZaloBFCLSummary:
    """Bảng tổng hợp chỉ số toàn bộ tập kiểm định (val_500)."""
    model_name: str
    input_mode: str  # 'Oracle Mode' hoặc 'Real ASR Mode'
    thinking_mode: bool
    total_samples: int
    state_hash_match_pct: float
    ast_tool_f1_pct: float
    schema_adherence_pct: float
    arg_exact_match_pct: float
    avg_thinking_tokens: float
    avg_latency_ms: float
    slm_total_params_b: float
    slm_active_params_b: float
    total_system_params_b: float
    budget_compliant: bool
    sample_metrics: list[SampleBFCLMetrics] = field(default_factory=list)


class ZaloBFCLEvaluator:
    """Hệ thống đánh giá toàn diện năng lực SLM theo chuẩn BFCL v4 và Zalo ToolEnv."""

    def __init__(
        self,
        catalog: dict[str, Any],
        slm_total_params_b: float,
        slm_active_params_b: float | None = None,
        asr_params_b: float = ASR_WINNER_PARAMS_B,
    ) -> None:
        self.catalog = catalog
        self.tools_map = catalog_tools(catalog)
        self.slm_total_params_b = slm_total_params_b
        self.slm_active_params_b = slm_active_params_b or slm_total_params_b
        self.asr_params_b = asr_params_b
        self.total_system_params_b = round(self.asr_params_b + self.slm_total_params_b, 3)
        self.budget_compliant = self.total_system_params_b < MAX_SYSTEM_PARAMS_B

    def evaluate_sample(
        self,
        prediction: ParsedThinkingOutput,
        gold_record: dict[str, Any],
        env: Any | None = None,
        latency_ms: float = 0.0,
    ) -> SampleBFCLMetrics:
        """Đo lường 6 chỉ số trên một mẫu kiểm định."""
        sample_id = gold_record.get("id", "unknown")
        gold_label = gold_record.get("label", {})
        gold_calls = gold_label.get("tool_calls", [])
        gold_state_hash = gold_label.get("state_hash")
        installed_apps = gold_record.get("context", {}).get("installed_apps", {})

        # Danh sách tool được phép gọi
        allowed_tools = {"sys.ask_missing_info"}
        for app_info in installed_apps.values():
            allowed_tools.update(app_info.get("tools", []))

        pred_calls = prediction.tool_calls

        # --------------------------------------------------------------------
        # Chỉ số 1: State Hash Match (Replay trên ToolEnv)
        # --------------------------------------------------------------------
        state_match = False
        replayed_hash = None

        if env is not None:
            # Replay pred_calls trên phiên bản clone của ToolEnv
            with contextlib_suppress():
                for call in pred_calls:
                    env.call(call["name"], call.get("arguments", {}))
                replayed_hash = env.get_state_hash()

            if gold_state_hash and replayed_hash:
                state_match = (replayed_hash == gold_state_hash)
            elif not gold_calls and not pred_calls:
                state_match = True
        else:
            # Nếu không truyền env, đối chiếu state_hash nếu pred_calls rỗng giống gold
            if not gold_calls and not pred_calls:
                state_match = True

        # --------------------------------------------------------------------
        # Chỉ số 2: AST Tool Selection F1
        # --------------------------------------------------------------------
        gold_names = [c["name"] for c in gold_calls]
        pred_names = [c["name"] for c in pred_calls]

        if not gold_names and not pred_names:
            ast_f1 = 1.0
            ast_p = 1.0
            ast_r = 1.0
        elif not gold_names or not pred_names:
            ast_f1 = 0.0
            ast_p = 0.0 if pred_names else 1.0
            ast_r = 0.0 if not pred_names else 1.0
        else:
            # Đếm số tool name khớp (1-to-1 multiset matching)
            gn_copy = list(gold_names)
            matched_count = 0
            for pn in pred_names:
                if pn in gn_copy:
                    matched_count += 1
                    gn_copy.remove(pn)

            ast_p = matched_count / len(pred_names)
            ast_r = matched_count / len(gold_names)
            ast_f1 = (2 * ast_p * ast_r / (ast_p + ast_r)) if (ast_p + ast_r) > 0 else 0.0

        # --------------------------------------------------------------------
        # Chỉ số 3: Schema Adherence Rate
        # --------------------------------------------------------------------
        valid_calls_count = 0
        total_calls_count = len(pred_calls)

        for c in pred_calls:
            c_name = c["name"]
            c_args = c.get("arguments", {})
            err = schema_error(c_name, c_args, self.tools_map, allowed_tools)
            if err is None:
                valid_calls_count += 1

        schema_rate = (valid_calls_count / total_calls_count) if total_calls_count > 0 else 1.0

        # --------------------------------------------------------------------
        # Chỉ số 4: Argument Exact Match Rate (trên các call đã khớp tên)
        # --------------------------------------------------------------------
        arg_match_rate = self._compute_arg_match_rate(pred_calls, gold_calls)

        return SampleBFCLMetrics(
            sample_id=sample_id,
            state_hash_match=state_match,
            ast_tool_f1=round(ast_f1, 4),
            ast_precision=round(ast_p, 4),
            ast_recall=round(ast_r, 4),
            schema_adherence_rate=round(schema_rate, 4),
            schema_valid_calls=valid_calls_count,
            schema_total_calls=total_calls_count,
            arg_exact_match_rate=round(arg_match_rate, 4),
            thinking_tokens=prediction.thinking_token_count,
            latency_ms=round(latency_ms, 2),
            error_note=prediction.error_message,
        )

    def _compute_arg_match_rate(
        self,
        pred_calls: list[dict[str, Any]],
        gold_calls: list[dict[str, Any]],
    ) -> float:
        """Đo lường độ khớp tham số (Exact Match cho non-fuzzy và Token-F1 > 0.6 cho fuzzy)."""
        if not gold_calls and not pred_calls:
            return 1.0
        if not gold_calls or not pred_calls:
            return 0.0

        matched_args_count = 0
        gold_available = list(range(len(gold_calls)))

        for pc in pred_calls:
            p_name = pc["name"]
            p_args = pc.get("arguments", {})
            spec = self.tools_map.get(p_name, {})
            param_specs = spec.get("parameters", {}).get("properties", {})

            for g_idx in gold_available:
                gc = gold_calls[g_idx]
                if gc["name"] != p_name:
                    continue

                g_args = gc.get("arguments", {})

                # Kiểm tra từng tham số
                all_args_match = True
                for k, g_val in g_args.items():
                    p_val = p_args.get(k)
                    is_fuzzy = param_specs.get(k, {}).get("x-match") == "fuzzy"

                    if is_fuzzy:
                        # Đo Token-F1 > 0.6
                        s_sub = str(p_val or "")
                        s_gold = str(g_val or "")
                        t_sub = _tokens(s_sub)
                        t_gold = _tokens(s_gold)
                        inter = len(t_sub & t_gold)
                        p = inter / len(t_sub) if t_sub else 0.0
                        r = inter / len(t_gold) if t_gold else 0.0
                        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                        if f1 <= 0.6:
                            all_args_match = False
                            break
                    else:
                        # Non-fuzzy bắt buộc khớp tuyệt đối
                        if p_val != g_val:
                            all_args_match = False
                            break

                if all_args_match:
                    matched_args_count += 1
                    gold_available.remove(g_idx)
                    break

        return matched_args_count / len(gold_calls)

    def summarize(
        self,
        model_name: str,
        input_mode: str,
        thinking_mode: bool,
        sample_metrics: list[SampleBFCLMetrics],
    ) -> ZaloBFCLSummary:
        """Tổng hợp toàn bộ chỉ số trên tập kiểm định val_500."""
        n = max(len(sample_metrics), 1)

        state_matches = sum(1 for m in sample_metrics if m.state_hash_match)
        avg_ast_f1 = sum(m.ast_tool_f1 for m in sample_metrics) / n
        avg_schema_rate = sum(m.schema_adherence_rate for m in sample_metrics) / n
        avg_arg_match = sum(m.arg_exact_match_rate for m in sample_metrics) / n
        avg_thinking_toks = sum(m.thinking_tokens for m in sample_metrics) / n
        avg_latency = sum(m.latency_ms for m in sample_metrics) / n

        return ZaloBFCLSummary(
            model_name=model_name,
            input_mode=input_mode,
            thinking_mode=thinking_mode,
            total_samples=len(sample_metrics),
            state_hash_match_pct=round(100.0 * state_matches / n, 2),
            ast_tool_f1_pct=round(100.0 * avg_ast_f1, 2),
            schema_adherence_pct=round(100.0 * avg_schema_rate, 2),
            arg_exact_match_pct=round(100.0 * avg_arg_match, 2),
            avg_thinking_tokens=round(avg_thinking_toks, 1),
            avg_latency_ms=round(avg_latency, 1),
            slm_total_params_b=self.slm_total_params_b,
            slm_active_params_b=self.slm_active_params_b,
            total_system_params_b=self.total_system_params_b,
            budget_compliant=self.budget_compliant,
            sample_metrics=sample_metrics,
        )


class contextlib_suppress:
    """Helper suppress an toàn không phụ thuộc contextlib ngoài."""

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        return True
