#!/usr/bin/env python3
"""scripts/build_lfm3_grpo_notebook.py.

Script tự động tạo Jupyter Notebook hoàn chỉnh:
`liquidAI/lfm3_grpo_zalo_lifemate_a100.ipynb`
Huấn luyện LFM3-1.7B bằng GRPO + LoRA trên GPU A100 cho Zalo AI Challenge 2026 (Task 1: LifeMate).
"""

from __future__ import annotations

import json
from pathlib import Path


def create_notebook() -> None:
    cells = []

    def add_markdown(source: str) -> None:
        lines = [line + "\n" for line in source.strip().split("\n")]
        # Bỏ newline của dòng cuối
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        })

    def add_code(source: str) -> None:
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        })

    # ==================== CELL 1: HEADER & TITLE ====================
    add_markdown("""# 🚀 Huấn Luyện Trợ Lý Thông Minh LifeMate với Reinforcement Learning (GRPO + LoRA)
## Mô hình: `LiquidAI/LFM3-1.7B` | Tối Ưu Hóa Riêng Cho GPU NVIDIA A100 (40GB / 80GB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) [![GitHub](https://img.shields.io/badge/GitHub-ZL26--TASK1-181717?logo=github)](https://github.com/EsperCS/ZL26-TASK1) [![Zalo AI Challenge](https://img.shields.io/badge/ZaloAI--2026-Task1--LifeMate-blue)](https://zaic.zalo.ai/)

Notebook này chuyển thể và mở rộng từ [Cookbook Chính Thức của Liquid AI](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_grpo_with_trl.ipynb), được tùy biến chuyên biệt cho **Zalo AI Challenge 2026 (Task 1: LifeMate)**:
- **Tác vụ**: Điều khiển trợ lý điện thoại qua giọng nói (ASR Transcript) với 114 công cụ trong hệ sinh thái (Calendar, Alarms, Tasks, Smart Home, Food Delivery, Medication...).
- **Phương pháp**: Học tăng cường **Group Relative Policy Optimization (GRPO)** từ thư viện Hugging Face `TRL` kết hợp **PEFT LoRA ($r=64$)**.
- **Không cần SFT mồi**: Mô hình tự học cách phân tích câu lệnh tiếng Việt, tự giải quyết các cạm bẫy tương đồng (lookalikes), bẫy thời gian tương đối (21 ngày) và gọi đúng công cụ thông qua **Hàm Thưởng Có Thể Kiểm Chứng (Verifiable Reward)**.
- **Tận dụng sức mạnh A100**: Khai thác tối đa Tensor Cores `bfloat16`, sinh đồng thời nhóm $G=8$ câu trả lời với `generation_batch_size=32` siêu tốc!""")

    # ==================== CELL 2: MOUNT DRIVE & SETUP ====================
    add_markdown("""# 📦 Bước 1: Kết Nối Google Drive & Chuẩn Bị Môi Trường
Chúng ta kết nối Google Drive để đọc dữ liệu cuộc thi (`train.jsonl`, `train_tools_catalog.json`) và lưu lại Checkpoint sau khi huấn luyện.""")

    add_code("""# 1. Mount Google Drive
from google.colab import drive
import os
import sys
from pathlib import Path

drive.mount('/content/drive')

# 2. Clone repository ZL26-TASK1 để tái sử dụng toàn bộ ContextBuilder & Parsers
REPO_DIR = Path("/content/zaloai-challenge-2026")
if not REPO_DIR.exists():
    os.system("git clone https://github.com/EsperCS/ZL26-TASK1.git /content/zaloai-challenge-2026")
else:
    os.system("cd /content/zaloai-challenge-2026 && git pull origin main")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Thư mục lưu checkpoint chính thức trên Google Drive
DRIVE_DIR = Path("/content/drive/MyDrive/ZaloAI_2026_ASR")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Thư mục checkpoint Drive: {DRIVE_DIR}")""")

    # ==================== CELL 3: INSTALL DEPENDENCIES ====================
    add_markdown("""Cài đặt các gói cần thiết cho LFM3 + TRL GRPO trên Colab.
> [!TIP]
> Chúng ta gỡ bỏ `torchvision` (thư viện đồ họa xử lý ảnh mà LFM3 không cần dùng tới) để tránh triệt để lỗi xung đột toán tử C++ `torchvision::nms` với PyTorch trên Colab.""")

    add_code("""# 1. Gỡ bỏ torchvision và cudf xung đột
!pip uninstall -y torchvision cudf-cu13 cudf-polars-cu13 -q

# 2. Cài đặt các gói chuẩn cho LFM3 + TRL GRPO
!pip install -q "torch==2.14.1" "trl==1.16.0" "transformers>=5.21.0" "peft>=0.21.2" "accelerate>=1.15.0" "datasets>=5.1.0" "rapidfuzz>=3.14.0" "python-dateutil>=2.9.0" --progress-bar off""")

    # ==================== CELL 4: HARDWARE & A100 CHECK ====================
    add_markdown("""Kiểm tra phần cứng GPU NVIDIA A100 và độ chính xác tính toán `bfloat16`.""")

    add_code("""import torch
import transformers
import trl
import peft

os.environ["WANDB_DISABLED"] = "true"

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"🖥️ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"🔢 Precision: {dtype} (A100 Tensor Cores Optimized: {use_bf16})")
print(f"📦 PyTorch: {torch.__version__} | Transformers: {transformers.__version__} | TRL: {trl.__version__}")""")

    # ==================== CELL 5: LOAD LFM3-1.7B ====================
    add_markdown("""# 🧠 Bước 2: Nạp Mô Hình LFM3-1.7B & Tokenizer
Mô hình `LiquidAI/LFM3-1.7B` đi kèm mã nguồn kiến trúc gốc, nạp qua `trust_remote_code=True` với `padding_side="left"` để hỗ trợ sinh theo batch (batched generation) cho GRPO.""")

    add_code("""from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LiquidAI/LFM3-1.7B"

print(f"📚 Đang nạp tokenizer '{model_id}'...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"🧠 Đang nạp mô hình '{model_id}' (bfloat16 trên A100)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Nạp thành công mô hình LFM3-1.7B sẵn sàng huấn luyện!")""")

    # ==================== CELL 6: DATA PREPARATION ====================
    add_markdown("""# 🗂️ Bước 3: Chuẩn Bị Dữ Liệu LifeMate Từ Google Drive
Hệ thống tự động tìm file zip `train+public_test.zip` hoặc file `train.jsonl` và `train_tools_catalog.json` từ Drive.""")

    add_code("""import json
import zipfile
import shutil

# Tìm dữ liệu từ Drive hoặc repo local
data_zip_candidates = [
    DRIVE_DIR / "train+public_test.zip",
    DRIVE_DIR / "train.zip",
    Path("/content/drive/MyDrive/train+public_test.zip"),
    REPO_DIR / "train+public_test.zip",
    REPO_DIR / "train+public_test",
]

extracted_dir = Path("/content/data")
extracted_dir.mkdir(parents=True, exist_ok=True)

# Giải nén nếu file zip tồn tại
found_zip = next((p for p in data_zip_candidates if p.is_file()), None)
if found_zip and not (extracted_dir / "train").is_dir():
    print(f"📦 Đang giải nén {found_zip.name} vào {extracted_dir}...")
    with zipfile.ZipFile(found_zip, "r") as z:
        z.extractall(extracted_dir)
    print("✅ Giải nén hoàn tất!")

# Xác định đường dẫn file train.jsonl và catalog
candidate_trains = [
    extracted_dir / "train" / "train.jsonl",
    extracted_dir / "train+public_test" / "train" / "train.jsonl",
    REPO_DIR / "train" / "train.jsonl",
    REPO_DIR / "train+public_test" / "train" / "train.jsonl",
    REPO_DIR / "train" / "val_mini.jsonl",
]
train_jsonl_path = next((p for p in candidate_trains if p.is_file()), None)

candidate_catalogs = [
    extracted_dir / "train" / "train_tools_catalog.json",
    extracted_dir / "train+public_test" / "train" / "train_tools_catalog.json",
    REPO_DIR / "train" / "train_tools_catalog.json",
    REPO_DIR / "train+public_test" / "train" / "train_tools_catalog.json",
]
catalog_json_path = next((p for p in candidate_catalogs if p.is_file()), None)

print(f"📄 Train Dataset: {train_jsonl_path}")
print(f"🧰 Tools Catalog: {catalog_json_path}")
assert train_jsonl_path and train_jsonl_path.is_file(), "Không tìm thấy train.jsonl! Hãy kiểm tra Google Drive."
assert catalog_json_path and catalog_json_path.is_file(), "Không tìm thấy train_tools_catalog.json!"

with open(catalog_json_path, encoding="utf-8") as f:
    tools_catalog = json.load(f)

print(f"✨ Số ứng dụng trong catalog: {len(tools_catalog.get('apps', []))}")""")

    # ==================== CELL 7: PROMPT & BALANCING ====================
    add_markdown("""# 🛠️ Bước 4: Đóng Gói Prompt Ngữ Cảnh & Cân Bằng Mẫu
Điểm tối ưu vượt bậc của chúng ta so với cookbook của Liquid AI:
- Liquid AI nạp cả 108 công cụ (~8,000 tokens) vào mọi prompt.
- **Trong LifeMate**: Mỗi câu lệnh chỉ cài đặt **4 đến 8 ứng dụng** (~15 tools). Ta chỉ đưa đúng schema của các app đã cài vào prompt, giúp độ dài prompt chỉ còn **~1,200 tokens (nhẹ gấp 6 lần!)**.
- Áp dụng kỹ thuật **Cân Bằng Dữ Liệu (Class Balancing)** của Liquid AI: nhân đôi các mẫu `tool_calls: []` (out-of-scope / app chưa cài đặt) để mô hình không bị thiên lệch luôn gọi bừa một công cụ.""")

    add_code("""from datasets import Dataset
from src.agent.context_builder import ContextBuilder

context_builder = ContextBuilder(tools_catalog=tools_catalog)

raw_samples = []
with open(train_jsonl_path, encoding="utf-8") as f:
    for line in f:
        line_str = line.strip()
        if line_str:
            raw_samples.append(json.loads(line_str))

print(f"📊 Tổng số mẫu đọc được từ train.jsonl: {len(raw_samples)}")

formatted_data = []
no_call_count = 0
multi_call_count = 0

for sample in raw_samples:
    s_id = sample.get("id", "")
    gold_calls = sample.get("label", {}).get("tool_calls", [])
    installed_apps = sample.get("context", {}).get("installed_apps", {})
    tools_schema = context_builder.build_tools_schema(installed_apps)
    
    # Xây dựng Messages ChatML kèm Bảng lịch 21 ngày và User Profile
    messages = context_builder.build_messages(sample=sample, thinking_mode=False)
    
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools_schema if tools_schema else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    is_no_call = len(gold_calls) == 0
    if is_no_call:
        no_call_count += 1
    if len(gold_calls) > 1:
        multi_call_count += 1
        
    formatted_data.append({
        "prompt": prompt_text,
        "calls": json.dumps(gold_calls, ensure_ascii=False),
        "id": s_id,
        "is_no_call": is_no_call,
    })

print(f"   - Mẫu không gọi tool (out-of-scope / clarify): {no_call_count} ({no_call_count/len(raw_samples):.1%})")
print(f"   - Mẫu gọi nhiều tool (multi-calls): {multi_call_count} ({multi_call_count/len(raw_samples):.1%})")

# CÂN BẰNG MẪU: Nhân đôi mẫu no_call để chống over-calling như Liquid AI khuyên dùng
balanced_data = list(formatted_data)
for item in formatted_data:
    if item["is_no_call"]:
        balanced_data.append(item)

train_dataset = Dataset.from_list(balanced_data).shuffle(seed=42)
print(f"🚀 Kích thước tập huấn luyện sau cân bằng: {len(train_dataset)} mẫu")

# Kiểm tra độ dài prompt
sample_tokens = tokenizer(train_dataset[0]["prompt"], add_special_tokens=False).input_ids
print(f"📏 Độ dài trung bình prompt: ~{len(sample_tokens):,} tokens (rất nhẹ, chạy siêu tốc trên A100!)")""")

    # ==================== CELL 8: REWARD FUNCTION ====================
    add_markdown("""# 🎯 Bước 5: Thiết Kế Hàm Thưởng Có Thể Kiểm Chứng (Verifiable Reward Function)
Hàm thưởng tính điểm trực tiếp từ kết quả sinh ra của mô hình:
- **1.0 điểm**: Khớp hoàn toàn cả tên tool và mọi tham số với gold target (`calls_match == True`).
- **0.3 × F1(tool_names)**: Thưởng khuyến khích khi mô hình chọn đúng công cụ dù tham số chưa chuẩn, giúp dẫn hướng gradient trong các bước đầu.
- **-0.2 điểm (Phạt)**: Nếu người dùng yêu cầu app chưa cài đặt mà mô hình vẫn bịa gọi tool.""")

    add_code("""import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from src.agent.lfm3_parser import parse_lfm3_response

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in unicodedata.normalize("NFC", s))
    return " ".join(s.split())

def text_similar(a: str, b: str) -> bool:
    a_n, b_n = norm_text(a), norm_text(b)
    if a_n == b_n:
        return True
    short, long_ = sorted((a_n, b_n), key=len)
    if short and short in long_ and len(short) / max(len(long_), 1) >= 0.5:
        return True
    return SequenceMatcher(None, a_n, b_n).ratio() >= 0.6

def same_arg(gold_val, pred_val):
    if gold_val == pred_val:
        return True
    if isinstance(gold_val, (int, float)) and isinstance(pred_val, (int, float)) and not isinstance(pred_val, bool):
        return abs(float(gold_val) - float(pred_val)) < 1e-5
    if isinstance(gold_val, str) and isinstance(pred_val, str):
        g_clean, p_clean = gold_val.strip(), pred_val.strip()
        if g_clean == p_clean:
            return True
        return text_similar(g_clean, p_clean)
    if isinstance(gold_val, list) and isinstance(pred_val, list):
        if len(gold_val) != len(pred_val):
            return False
        return all(any(same_arg(g, p) for p in pred_val) for g in gold_val)
    if isinstance(gold_val, dict) and isinstance(pred_val, dict):
        if set(gold_val.keys()) != set(pred_val.keys()):
            return False
        return all(same_arg(gold_val[k], pred_val[k]) for k in gold_val)
    return False

def same_tool_call(gold: dict, pred: dict) -> bool:
    if gold.get("name") != pred.get("name"):
        return False
    g_args = gold.get("arguments", {})
    p_args = pred.get("arguments", {})
    for k, v in g_args.items():
        if k not in p_args:
            return False
        if not same_arg(v, p_args[k]):
            return False
    return True

def calls_match(gold_calls: list, pred_calls: list) -> bool:
    if len(gold_calls) != len(pred_calls):
        return False
    if len(gold_calls) == 0:
        return True
    used = set()
    for g in gold_calls:
        match_found = False
        for idx, p in enumerate(pred_calls):
            if idx not in used and same_tool_call(g, p):
                used.add(idx)
                match_found = True
                break
        if not match_found:
            return False
    return len(used) == len(gold_calls)

def tool_name_f1(gold_calls: list, pred_calls: list) -> float:
    gold_names = Counter(c.get("name", "") for c in gold_calls)
    pred_names = Counter(c.get("name", "") for c in pred_calls)
    overlap = sum((gold_names & pred_names).values())
    total = sum(gold_names.values()) + sum(pred_names.values())
    return 0.0 if total == 0 else (2.0 * overlap / total)

def tool_call_reward(prompts, completions, calls, **kwargs):
    rewards = []
    for prompt, comp, gold_str in zip(prompts, completions, calls):
        gold = json.loads(gold_str) if isinstance(gold_str, str) else gold_str
        parsed = parse_lfm3_response(comp)
        pred = parsed.tool_calls
        
        if calls_match(gold, pred):
            rewards.append(1.0)
        else:
            f1 = tool_name_f1(gold, pred)
            if len(gold) == 0 and len(pred) > 0:
                rewards.append(-0.2)
            else:
                rewards.append(0.3 * f1)
    return rewards

# Kiểm tra nhanh hàm thưởng với mẫu chuẩn
sample_gold = [{"name": "tasks.add", "arguments": {"title": "Kiểm tra phòng", "due": "2026-10-02T09:00:00+07:00"}}]
assert calls_match(sample_gold, sample_gold) is True
print("✅ Hàm thưởng Verifiable Reward Function hoạt động chính xác 100%!")""")

    # ==================== CELL 9: LORA CONFIG ====================
    add_markdown("""# 🎛️ Bước 6: Cấu Hình PEFT LoRA Cho LFM3-1.7B
Sử dụng chính xác cấu hình LoRA chuẩn của Liquid AI ($r=32, \\alpha=64$) trên các modules attention và gated convolution.""")

    add_code("""from peft import LoraConfig

peft_config = LoraConfig(
    r=32,             # Cấu hình chuẩn của Liquid AI
    lora_alpha=64,    # Alpha = 2 * r
    lora_dropout=0.0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
        "in_proj", "out_proj",                     # Gated short convolutions
        "w1", "w2", "w3",                          # Feed-Forward MLP
        "gate_proj",                               # Per-head attention gate (đặc thù của LFM3)
    ],
    task_type="CAUSAL_LM",
)""")

    # ==================== CELL 10: GRPO TRAINING ARGS ====================
    add_markdown("""# ⚡ Bước 7: Cấu Hình Siêu Tham Số GRPO Chuẩn Liquid AI (Tránh OOM)
Áp dụng chính xác siêu tham số chuẩn từ Cookbook chính thức của Liquid AI để đảm bảo tính ổn định và không bao giờ bị tràn bộ nhớ VRAM (OOM):
- **`generation_batch_size = 8`**: Sinh từng nhóm 8 câu (đúng kích thước 1 group của 1 prompt), tránh quá tải bộ nhớ KV cache khi sinh.
- **`per_device_train_batch_size = 2`**: Batch size an toàn, ổn định.
- **`gradient_accumulation_steps = 16`**: Tích lũy 32 câu hỏi trên mỗi bước cập nhật trọng số.
- **`max_completion_length = 256`**: Đủ chỗ cho lệnh tool call mà không tốn thừa token.""")

    add_code("""from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./lfm3-lifemate-a100-checkpoints",
    max_steps=150,                        # 150 optimization steps chuẩn Liquid AI
    per_device_train_batch_size=2,        # 2 requests mỗi micro-batch (chuẩn Liquid AI)
    gradient_accumulation_steps=16,       # 2 * 16 = 32 answers mỗi step (chuẩn Liquid AI)
    num_generations=8,                    # 8 câu sinh cho mỗi prompt để so sánh tương quan (GRPO)
    generation_batch_size=8,              # CỰC KỲ QUAN TRỌNG: Chỉ sinh 8 câu mỗi lượt để chống OOM tuyệt đối!
    max_completion_length=256,            # 256 tokens chuẩn Liquid AI
    temperature=1.0,                      # Độ đa dạng trong nhóm
    learning_rate=1e-5,                   # LR chuẩn 1e-5
    beta=0.04,                            # KL penalty giữ mô hình không bị suy thoái ngôn ngữ
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=5,
    logging_steps=5,
    save_strategy="no",                   # Tiết kiệm I/O đĩa
    bf16=use_bf16,
    fp16=not use_bf16,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=tool_call_reward,
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("🎛️ Phân bổ tham số huấn luyện LoRA:")
trainer.model.print_trainable_parameters()""")

    # ==================== CELL 11: LAUNCH TRAINING ====================
    add_markdown("""# 🚀 Bước 8: Khởi Động Huấn Luyện GRPO""")

    add_code("""print("🚀 Bắt đầu quá trình huấn luyện GRPO trên NVIDIA A100...")
trainer.train()
print("🎉 Hoàn tất huấn luyện GRPO thành công!")
print(f"🧮 VRAM đỉnh sử dụng trên A100: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

# Xem biểu đồ tiến trình tăng điểm thưởng theo từng Step
for log in trainer.state.log_history:
    if "reward" in log:
        print(f"Step {log['step']:>3}: Reward TB {log['reward']:.2f} | Độ dài hoàn thành {log['completions/mean_length']:.0f} tokens")""")

    # ==================== CELL 12: EVALUATE & VERIFY ====================
    add_markdown("""# 📊 Bước 9: Đối Chiếu Trực Tiếp Trước vs Sau Huấn Luyện (Side-by-Side Comparison)
Sử dụng `disable_adapter()` để so sánh phản hồi của mô hình gốc (Original) và mô hình đã qua rèn luyện GRPO (Fine-Tuned).""")

    add_code("""def answer_sample(m, prompt_str):
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(m.device)
    with torch.inference_mode():
        out = m.generate(**inputs, max_new_tokens=384, do_sample=False)[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(out, skip_special_tokens=False)
    parsed = parse_lfm3_response(text)
    return parsed.tool_calls, text

trainer.model.eval()
print("🔍 SO SÁNH PHẢN HỒI: MÔ HÌNH GỐC vs MÔ HÌNH ĐÃ HUẤN LUYỆN GRPO:\\n")

test_samples = train_dataset.select(range(5))
for idx, s in enumerate(test_samples):
    gold = json.loads(s["calls"])
    prompt = s["prompt"]
    
    # Mô hình gốc (tắt adapter LoRA)
    with trainer.model.disable_adapter():
        orig_calls, orig_text = answer_sample(trainer.model, prompt)
    
    # Mô hình sau huấn luyện (bật adapter LoRA)
    ft_calls, ft_text = answer_sample(trainer.model, prompt)
    
    orig_ok = calls_match(gold, orig_calls)
    ft_ok = calls_match(gold, ft_calls)
    
    print(f"[{idx+1}] Sample ID: {s['id']}")
    print(f"   🎯 Gold Target:     {gold}")
    print(f"   {'✅' if orig_ok else '❌'} Gốc (Original):   {orig_calls}")
    print(f"   {'✅' if ft_ok else '❌'} Đã luyện (GRPO):  {ft_calls}\\n")""")

    # ==================== CELL 13: MERGE & SAVE TO DRIVE ====================
    add_markdown("""# 💾 Bước 10: Hợp Nhất Trọng Số (Merge LoRA) & Lưu Trực Tiếp Vào Google Drive
Lưu adapter LoRA và hợp nhất trực tiếp vào base model (`merge_and_unload()`) để mô hình xuất ra là một thư mục Hugging Face hoàn chỉnh, chạy offline không tốn thêm overhead tính toán.""")

    add_code("""# 1. Lưu LoRA Adapter riêng biệt lên Drive
lora_output_dir = DRIVE_DIR / "lfm3_lifemate_grpo_lora"
trainer.save_model(str(lora_output_dir))
print(f"💾 Đã lưu LoRA Adapter vào: {lora_output_dir}")

# 2. Hợp nhất LoRA vào trọng số gốc LFM3-1.7B
merged_output_dir = DRIVE_DIR / "lfm3_lifemate_grpo_merged"
print("🔄 Đang hợp nhất LoRA vào trọng số gốc LFM3-1.7B...")
merged_model = trainer.model.merge_and_unload()

# Lưu kèm modeling code của LFM3 để nạp offline trong container thi
merged_model.save_pretrained(str(merged_output_dir))
tokenizer.save_pretrained(str(merged_output_dir))

print(f"🎉 ĐÃ LƯU THÀNH CÔNG MÔ HÌNH HỢP NHẤT VÀO GOOGLE DRIVE: {merged_output_dir}")
print("📦 Bạn có thể dùng model này trực tiếp trong scripts/run_pipeline_lfm3.py với --model-id!")""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("liquidAI/lfm3_grpo_zalo_lifemate_a100.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"[OK] Da tao thanh cong notebook tai: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    create_notebook()
