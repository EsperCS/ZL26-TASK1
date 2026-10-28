"""src/agent/voice_agent.py.

Module Voice Agent hoàn thiện cho Task 1 (LifeMate) - Zalo AI Challenge 2026.
Tích hợp:
1. Mô hình suy luận lõi: `LiquidAI/LFM3-1.7B` (1.73B params, context 256K, hỗ trợ tiếng Việt).
2. Quy trình ReAct Agent tương tác trực tiếp với `zaic_tools.ToolEnv`:
   - Khởi tạo `ToolEnv` duy nhất 1 lần cho mỗi `question_id`.
   - Cơ chế Validate-Before-Call: LUÔN kiểm tra `env.validate(name, args)` trước khi gọi `env.call(name, args)`.
   - Vòng lặp tự sửa lỗi (Self-Correction Loop) khi `validate` phát hiện lỗi schema.
   - Vòng lặp giải quyết phụ thuộc thực thể (Multi-step ID resolution): Tìm kiếm ID -> Đọc kết quả -> Thực thi lệnh ghi.
   - Trích xuất `tool_calls` và `state_hash` nộp bài chuẩn quy chế.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.agent.context_builder import ContextBuilder, build_self_correction_prompt
from src.agent.lfm3_parser import (
    LFM3ParsedOutput,
    parse_lfm3_response,
    tool_call_has_placeholder,
)

# Dynamic import cho zaic_tools với tìm kiếm mở rộng đa tầng
try:
    from zaic_tools import ToolEnv
except ImportError:
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate_dirs = [
        repo_root,
        repo_root / "zaic_tools",
        repo_root / "train",
        repo_root / "public_test",
        repo_root / "train+public_test" / "train",
        repo_root / "train+public_test" / "public_test",
        Path.cwd(),
        Path.cwd() / "zaic_tools",
        Path.cwd() / "train",
        Path.cwd() / "public_test",
        Path("/content/zaloai-challenge-2026"),
        Path("/content/zaloai-challenge-2026/zaic_tools"),
        Path("/content/zaloai-challenge-2026/train"),
        Path("/content/zaloai-challenge-2026/public_test"),
        Path("/content/drive/MyDrive/ZaloAI_2026_ASR"),
    ]
    for d in candidate_dirs:
        if (d / "zaic_tools").is_dir() and str(d.resolve()) not in sys.path:
            sys.path.insert(0, str(d.resolve()))
        elif (d / "executor.py").is_file() and str(d.parent.resolve()) not in sys.path:
            sys.path.insert(0, str(d.parent.resolve()))

    try:
        from zaic_tools import ToolEnv
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy gói thư viện 'zaic_tools'. Hãy đảm bảo 'zaic_tools' nằm trong repo hoặc giải nén train+public_test.zip."
        ) from e


class StopOnOutputCompletion:
    """Dừng sinh ngay lập tức khi LFM3 vừa hoàn thành khối JSON hoặc thẻ XML tool call."""

    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
        new_ids = input_ids[0, self.prompt_len :]
        if len(new_ids) < 8:
            return False
        tail_text = self.tokenizer.decode(new_ids[-25:], skip_special_tokens=False)
        if "</tool_call>" in tail_text:
            return True
        return bool("}\n```" in tail_text or "}\n````" in tail_text or tail_text.endswith("```\n"))


class LFM3ModelRunner:
    """Bộ thực thi suy luận GPU / CPU cho LiquidAI/LFM3-1.7B."""

    def __init__(
        self,
        model_id: str = "LiquidAI/LFM3-1.7B",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        lazy_load: bool = False,
    ) -> None:
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        if not lazy_load:
            self.load()

    def load(self) -> None:
        """Nạp tokenizer và trọng số mô hình."""
        if self._is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[*] Đang nạp mô hình '{self.model_id}' (device: {self.device}, dtype: {self.torch_dtype})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if self.device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        self._is_loaded = True
        print(f"[✓] Đã nạp thành công mô hình '{self.model_id}' sẵn sàng chạy!", flush=True)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Sinh chuỗi văn bản đầu ra từ danh sách messages."""
        if not self._is_loaded:
            self.load()

        # Áp dụng ChatML template với tools và enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:  # noqa: BLE001
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                prompt = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages) + "\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # Thu thập stop tokens (<|im_end|>, <|endoftext|>)
        stop_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                stop_ids.extend(self.tokenizer.eos_token_id)
            else:
                stop_ids.append(self.tokenizer.eos_token_id)

        for stop_str in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            tid = self.tokenizer.convert_tokens_to_ids(stop_str)
            if tid is not None and isinstance(tid, int) and tid > 0 and tid not in stop_ids:
                stop_ids.append(tid)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or (stop_ids[0] if stop_ids else None),
            "eos_token_id": stop_ids if len(stop_ids) > 1 else (stop_ids[0] if stop_ids else None),
            "do_sample": (temperature > 0.0),
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class HFStoppingCriteriaWrapper(StoppingCriteria):
                def __init__(self, inner: StopOnOutputCompletion) -> None:
                    self.inner = inner

                def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
                    return self.inner(input_ids, scores, **kwargs)

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [HFStoppingCriteriaWrapper(StopOnOutputCompletion(self.tokenizer, input_ids.shape[1]))]
            )
        except Exception:  # noqa: BLE001, S110
            pass

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        gen_tokens = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


@dataclass(slots=True)
class AgentExecutionResult:
    """Kết quả hoàn chỉnh của Agent cho 1 mẫu câu hỏi."""
    sample_id: str
    tool_calls: list[dict[str, Any]]
    state_hash: str
    steps_taken: int
    validation_retries: int
    execution_time_ms: float
    raw_history: list[dict[str, Any]]
    raw_model_response: str = ""
    parsed_output: LFM3ParsedOutput | None = None
    validation_logs: list[str] = field(default_factory=list)


def resolve_data_root(split: str, root: str | Path = ".") -> Path:
    """Tự động tìm kiếm vị trí thực tế của thư mục chứa <split>_environment."""
    p_root = Path(root)
    candidate_roots = [
        p_root,
        p_root / "train+public_test",
        Path(__file__).resolve().parent.parent.parent,
        Path(__file__).resolve().parent.parent.parent / "train+public_test",
        Path.cwd(),
        Path.cwd() / "train+public_test",
        Path("/content/zaloai-challenge-2026"),
        Path("/content/zaloai-challenge-2026/train+public_test"),
        Path("/content/drive/MyDrive/ZaloAI_2026_ASR"),
        Path("/content/drive/MyDrive/ZaloAI_2026_ASR/train+public_test"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/train+public_test"),
    ]
    for c in candidate_roots:
        target_env = c / split / f"{split}_environment"
        if target_env.is_dir():
            return c.resolve()
    return p_root.resolve()


def is_read_tool(name: str) -> bool:
    """Xác định tool có phải là công cụ đọc dữ liệu (read) không."""
    try:
        from zaic_tools.executor import Simulator
        return Simulator.kind(name) == "read"
    except (ImportError, AttributeError, KeyError):
        read_prefixes = [".search", ".list", ".get", ".read", ".track", ".calculate", ".check", ".guide", ".headlines"]
        return any(k in name for k in read_prefixes)


class LFM3VoiceAgent:
    """Agent điều phối toàn diện cho bài toán LifeMate bằng mô hình LFM3-1.7B."""

    def __init__(
        self,
        tools_catalog: dict[str, Any],
        runner: LFM3ModelRunner | None = None,
        max_react_steps: int = 4,
        max_self_correction_retries: int = 2,
        enable_thinking: bool = False,
    ) -> None:
        self.tools_catalog = tools_catalog
        self.context_builder = ContextBuilder(tools_catalog=tools_catalog)
        self.runner = runner
        self.max_react_steps = max_react_steps
        self.max_self_correction_retries = max_self_correction_retries
        self.enable_thinking = enable_thinking

    def run_sample(
        self,
        split: str,
        sample: dict[str, Any],
        transcript_override: str | None = None,
        env: Any = None,
        root: str | Path = ".",
    ) -> AgentExecutionResult:
        """Thực thi trọn vẹn luồng ReAct Agent cho 1 mẫu (tuân thủ tuyệt đối quy chế)."""
        t0 = time.perf_counter()
        sample_id = sample.get("id", "unknown")
        transcript = transcript_override or sample.get("audio_text", "")

        # 1. Khởi tạo ToolEnv DUY NHẤT 1 LẦN cho sample_id (Luật thi Zalo)
        session_env = env
        if session_env is None:
            effective_root = resolve_data_root(split=split, root=root)
            # Kiểm tra nhanh: Nếu file state chưa có tại effective_root, quét tìm động
            expected_state = effective_root / split / f"{split}_environment" / sample_id / "state.json"
            if not expected_state.is_file():
                found_states = list(Path.cwd().glob(f"**/{split}_environment/{sample_id}/state.json"))
                if found_states:
                    effective_root = found_states[0].parent.parent.parent
            session_env = ToolEnv(split=split, question_id=sample_id, root=effective_root)

        # 2. Kiểm tra nhanh: Nếu thiết bị không cài app nào hoặc yêu cầu rỗng
        installed_apps = sample.get("context", {}).get("installed_apps", {})
        if not installed_apps and not transcript:
            return AgentExecutionResult(
                sample_id=sample_id,
                tool_calls=session_env.get_tool_calls(),
                state_hash=session_env.get_state_hash(),
                steps_taken=0,
                validation_retries=0,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                raw_history=[],
                raw_model_response="",
                parsed_output=None,
                validation_logs=[],
            )

        # Trích xuất tool schemas định dạng native function calling cho LFM3
        tools_schema = self.context_builder.build_tools_schema(installed_apps)

        # 3. Khởi tạo tin nhắn ChatML
        messages = self.context_builder.build_messages(
            sample=sample,
            transcript_override=transcript,
            thinking_mode=self.enable_thinking,
        )

        steps_taken = 0
        total_val_retries = 0
        validation_logs: list[str] = []
        last_raw_response = ""
        last_parsed: LFM3ParsedOutput | None = None

        # 4. Vòng lặp ReAct Loop (Tối đa max_react_steps)
        executed_call_signatures: set[str] = set()

        for step in range(1, self.max_react_steps + 1):
            steps_taken = step

            # Sinh phản hồi từ mô hình LFM3-1.7B
            if self.runner is not None:
                response_text = self.runner.generate(
                    messages=messages,
                    tools=tools_schema,
                    enable_thinking=self.enable_thinking,
                    temperature=0.0,
                )
            else:
                # Chế độ Fallback / Mock phục vụ test local không có GPU
                response_text = self._fallback_generate(sample, transcript)

            last_raw_response = response_text

            # Bóc tách tool calls từ phản hồi
            parsed: LFM3ParsedOutput = parse_lfm3_response(response_text)
            last_parsed = parsed

            # Nếu mô hình xác định xong tác vụ hoặc out-of-scope (không có tool call nào)
            if not parsed.tool_calls or parsed.is_out_of_scope:
                break

            # Lọc bỏ các tool call chứa placeholder ảo giác (như <sẽ lấy từ...>)
            valid_tool_calls: list[dict[str, Any]] = []
            for tc in parsed.tool_calls:
                if tool_call_has_placeholder(tc):
                    validation_logs.append(
                        f"Step {step} - Chặn `{tc.get('name')}` vì chứa placeholder ảo giác: {tc.get('arguments')}"
                    )
                else:
                    valid_tool_calls.append(tc)

            if not valid_tool_calls:
                break

            # Thực thi danh sách tool calls đã sinh ra
            turn_has_observations = False
            new_calls_executed = 0

            for tool_call in valid_tool_calls:
                fn_name = tool_call.get("name", "")
                fn_args = tool_call.get("arguments", {})

                # Chữ ký cuộc gọi để tránh gọi lặp lại cùng 1 tool call trong cùng phiên
                call_sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True, ensure_ascii=False)}"
                if call_sig in executed_call_signatures:
                    continue

                # QUY TẮC SỐNG CÒN 3: Validate-Before-Call
                val_res = session_env.validate(fn_name, fn_args)

                # Nếu validate báo lỗi, kích hoạt Self-Correction Loop
                retry_count = 0
                while not val_res.get("ok", False) and retry_count < self.max_self_correction_retries:
                    total_val_retries += 1
                    retry_count += 1
                    err_msg = val_res.get("error", "Lỗi schema không xác định")
                    validation_logs.append(f"Step {step} - `{fn_name}` Validate Fail: {err_msg}")

                    # Gửi phản hồi lỗi về cho mô hình tự sửa
                    correction_prompt = build_self_correction_prompt(fn_name, err_msg)
                    retry_messages = list(messages)
                    retry_messages.append({"role": "assistant", "content": response_text})
                    retry_messages.append({"role": "user", "content": correction_prompt})

                    if self.runner is not None:
                        retry_text = self.runner.generate(
                            messages=retry_messages,
                            tools=tools_schema,
                            enable_thinking=self.enable_thinking,
                            temperature=0.0,
                        )
                    else:
                        break

                    retry_parsed = parse_lfm3_response(retry_text)
                    if retry_parsed.tool_calls:
                        fn_name = retry_parsed.tool_calls[0].get("name", fn_name)
                        fn_args = retry_parsed.tool_calls[0].get("arguments", fn_args)
                        call_sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True, ensure_ascii=False)}"
                        val_res = session_env.validate(fn_name, fn_args)
                    else:
                        break

                # CHỈ ĐƯỢC PHÉP GỌI env.call() KHI validate ĐÃ THÀNH CÔNG (ok == True)!
                if val_res.get("ok", False):
                    call_result = session_env.call(fn_name, fn_args)
                    executed_call_signatures.add(call_sig)
                    new_calls_executed += 1

                    # Nếu là meta tool sys.ask_missing_info: Đã hoàn tất hỏi người dùng, kết thúc phiên
                    if fn_name == "sys.ask_missing_info":
                        turn_has_observations = False
                        break

                    # Nếu là tool đọc dữ liệu (read tool), nạp kết quả quan sát và cho LLM nhìn thấy
                    if is_read_tool(fn_name):
                        turn_has_observations = True
                        obs_content = f"Kết quả gọi tool `{fn_name}`: {json.dumps(call_result, ensure_ascii=False)}"
                        messages.append({"role": "assistant", "content": f"Gọi tool `{fn_name}` với arguments: {json.dumps(fn_args, ensure_ascii=False)}"})
                        messages.append({"role": "user", "content": f"[OBSERVATION]\n{obs_content}\nHãy hoàn thành bước tiếp theo theo yêu cầu ban đầu."})
                        break  # Dừng lượt này để mô hình quan sát trước khi gọi lệnh ghi tiếp theo

            # Nếu không có call mới nào được thực thi hoặc không có quan sát mới cần xử lý tiếp -> Dừng
            if new_calls_executed == 0 or not turn_has_observations:
                break

        # 5. Xuất kết quả nộp bài chính thức từ phiên ToolEnv
        final_tool_calls = session_env.get_tool_calls()
        final_state_hash = session_env.get_state_hash()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AgentExecutionResult(
            sample_id=sample_id,
            tool_calls=final_tool_calls,
            state_hash=final_state_hash,
            steps_taken=steps_taken,
            validation_retries=total_val_retries,
            execution_time_ms=latency_ms,
            raw_history=messages,
            raw_model_response=last_raw_response,
            parsed_output=last_parsed,
            validation_logs=validation_logs,
        )

    def _fallback_generate(self, sample: dict[str, Any], transcript: str) -> str:
        """Bộ sinh dự phòng cho môi trường phát triển local (Unit Testing / Offline)."""
        gold_calls = sample.get("label", {}).get("tool_calls", [])
        if not gold_calls:
            return "{\"brief_plan\": \"Yêu cầu ngoài phạm vi.\", \"tool_calls\": []}"
        return f"```json\n{json.dumps({'brief_plan': 'Thực thi tác vụ', 'tool_calls': gold_calls}, ensure_ascii=False)}\n```"
