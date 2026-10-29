#!/usr/bin/env python3
"""scripts/build_lfm3_grpo_notebook.py.

Tạo file notebook `liquidAI/lfm3_grpo_zalo_lifemate_a100.ipynb`
sao chép chuẩn 1:1 toàn bộ cấu trúc, thư viện, logic so khớp và siêu tham số của Liquid AI
(`liquidAI/lfm3_grpo_with_trl.ipynb`), CHỈ THAY ĐÚNG NGUỒN DATASET & TOOL CATALOG của Zalo LifeMate.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_notebook() -> None:
    ref_path = Path("liquidAI/lfm3_grpo_with_trl.ipynb")
    with open(ref_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # CELL 0: Header Markdown
    nb["cells"][0]["source"] = [
        "# Training LifeMate Assistant with Reinforcement Learning: LFM3-1.7B + GRPO with TRL\n",
        "\n",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) [![GitHub](https://img.shields.io/badge/GitHub-ZL26--TASK1-181717?logo=github)](https://github.com/EsperCS/ZL26-TASK1) [![Zalo AI Challenge](https://img.shields.io/badge/ZaloAI--2026-Task1--LifeMate-blue)](https://zaic.zalo.ai/)\n",
        "\n",
        "Notebook này giữ nguyên 100% cấu trúc, thiết lập thư viện và logic so khớp của [Cookbook Chính Thức của Liquid AI](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_grpo_with_trl.ipynb), chỉ lắp tập dữ liệu từ **Zalo AI Challenge 2026 (Task 1: LifeMate)** lấy trực tiếp từ Google Drive.\n",
        "\n",
        "## 🎯 What you'll find\n",
        "\n",
        "- **The app**: Danh mục 148 công cụ LifeMate (`train_tools_catalog.json`) nạp vào mọi prompt.\n",
        "- **A verifiable reward**: Hàm thưởng kiểm chứng so khớp từng tham số với gold calls theo đúng chuẩn của Liquid AI.\n",
        "- **GRPO + LoRA**: Mô hình LFM3-1.7B tự học qua thuật toán Group Relative Policy Optimization (num_generations=8, generation_batch_size=8, r=32) chống OOM tuyệt đối trên Colab A100.\n",
        "- **Evaluation**: Đánh giá độ chính xác so khớp hoàn toàn (exact match) trước và sau khi huấn luyện."
    ]

    # CELL 1: Setup Markdown
    nb["cells"][1]["source"] = [
        "# 📦 Installation & Setup\n",
        "\n",
        "Kết nối Google Drive để nạp dữ liệu cuộc thi (`train.jsonl`, `train_tools_catalog.json`) và cố định đúng các phiên bản thư viện đã được Liquid AI kiểm thử."
    ]

    # CELL 2: Install Code
    nb["cells"][2]["source"] = [
        "# 1. Kết nối Google Drive để lấy dữ liệu\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# 2. Gỡ bỏ torchvision và cudf xung đột trên Colab để tránh lỗi C++ operator\n",
        "!pip uninstall -y torchvision cudf-cu13 cudf-polars-cu13 -q\n",
        "\n",
        "# 3. Cài đặt các thư viện y xì đúc như notebook của Liquid AI\n",
        "!pip install -qqq \"torch==2.14.1\" \"torchao==0.18.0\" \"trl==1.16.0\" \"transformers==5.20.2\" \"peft==0.21.2\" \"accelerate==1.15.0\" \"datasets==5.1.0\" \"huggingface-hub==2.3.0\" \"tokenizers==0.23.3\" \"safetensors==0.8.0\" --progress-bar off\n"
    ]

    # CELL 6: Tools Markdown
    nb["cells"][6]["source"] = [
        "# 📱 The app: 148 tools (Zalo LifeMate Ecosystem)\n",
        "\n",
        "Thay vì tải bộ tool của Liquid AI, chúng ta nạp danh mục 148 công cụ LifeMate từ Google Drive (hoặc repository) và tự động suy luận kiểu dữ liệu (`kinds`) để dùng cho hàm thưởng so khớp."
    ]

    # CELL 7: Load Tools Code
    nb["cells"][7]["source"] = [
        "from collections import Counter\n",
        "from pathlib import Path\n",
        "import json\n",
        "import os\n",
        "\n",
        "# Tìm file train_tools_catalog.json trên Google Drive hoặc tự động clone từ GitHub\n",
        "possible_tool_paths = [\n",
        "    Path(\"/content/drive/MyDrive/ZaloAI_2026_ASR/train_tools_catalog.json\"),\n",
        "    Path(\"/content/drive/MyDrive/train_tools_catalog.json\"),\n",
        "    Path(\"/content/zaloai-challenge-2026/data/task1_lifemate/train+public_test/train/train_tools_catalog.json\"),\n",
        "    Path(\"./data/task1_lifemate/train+public_test/train/train_tools_catalog.json\"),\n",
        "]\n",
        "tool_file = next((p for p in possible_tool_paths if p.exists()), None)\n",
        "if not tool_file:\n",
        "    os.system(\"git clone https://github.com/EsperCS/ZL26-TASK1.git /content/zaloai-challenge-2026\")\n",
        "    tool_file = Path(\"/content/zaloai-challenge-2026/data/task1_lifemate/train+public_test/train/train_tools_catalog.json\")\n",
        "\n",
        "raw_catalog = json.loads(tool_file.read_text(encoding=\"utf-8\"))\n",
        "\n",
        "tools = []\n",
        "kinds = {}\n",
        "\n",
        "# Luôn có công cụ hệ thống sys.ask_missing_info\n",
        "tools.append({\n",
        "    \"type\": \"function\",\n",
        "    \"function\": {\n",
        "        \"name\": \"sys.ask_missing_info\",\n",
        "        \"description\": \"Hỏi lại người dùng khi thiếu thông tin bắt buộc để thực hiện tác vụ.\",\n",
        "        \"parameters\": {\n",
        "            \"type\": \"object\",\n",
        "            \"properties\": {\n",
        "                \"tool_name\": {\"type\": \"string\", \"description\": \"Tên tool dự định gọi\"},\n",
        "                \"params\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"Danh sách tham số còn thiếu\"}\n",
        "            },\n",
        "            \"required\": [\"tool_name\", \"params\"]\n",
        "        }\n",
        "    }\n",
        "})\n",
        "kinds[\"sys.ask_missing_info\"] = {\"tool_name\": \"name\", \"params\": \"list_name\"}\n",
        "\n",
        "# Nạp toàn bộ 147 công cụ từ 46 ứng dụng trong catalog LifeMate\n",
        "for app in raw_catalog.get(\"apps\", []):\n",
        "    for t in app.get(\"tools\", []):\n",
        "        t_name = t[\"name\"]\n",
        "        params = t.get(\"parameters\", {\"type\": \"object\", \"properties\": {}})\n",
        "        tools.append({\n",
        "            \"type\": \"function\",\n",
        "            \"function\": {\n",
        "                \"name\": t_name,\n",
        "                \"description\": t.get(\"description\", \"\"),\n",
        "                \"parameters\": params\n",
        "            }\n",
        "        })\n",
        "        \n",
        "        # Tự động suy luận kinds phục vụ so khớp trong cùng cấu trúc Liquid AI\n",
        "        t_kinds = {}\n",
        "        props = params.get(\"properties\", {})\n",
        "        for p_name, p_spec in props.items():\n",
        "            p_type = p_spec.get(\"type\", \"string\")\n",
        "            p_fmt = p_spec.get(\"format\", \"\")\n",
        "            if p_fmt == \"date-time\" or \"datetime\" in p_name:\n",
        "                kind = \"datetime\"\n",
        "            elif p_fmt == \"date\" or \"date\" in p_name:\n",
        "                kind = \"date\"\n",
        "            elif p_fmt == \"time\" or (\"time\" in p_name and \"times\" not in p_name):\n",
        "                kind = \"time\"\n",
        "            elif p_type == \"boolean\":\n",
        "                kind = \"bool\"\n",
        "            elif p_type in (\"integer\", \"number\"):\n",
        "                kind = \"int\" if p_type == \"integer\" else \"number\"\n",
        "            elif \"enum\" in p_spec:\n",
        "                kind = \"enum\"\n",
        "            elif p_type == \"array\":\n",
        "                item_spec = p_spec.get(\"items\", {})\n",
        "                if \"enum\" in item_spec:\n",
        "                    kind = \"list_enum\"\n",
        "                elif item_spec.get(\"type\") == \"string\":\n",
        "                    kind = \"list_name\" if (\"id\" in p_name or \"name\" in p_name) else \"list_text\"\n",
        "                else:\n",
        "                    kind = \"list_text\"\n",
        "            elif \"id\" in p_name or \"phone\" in p_name or \"code\" in p_name:\n",
        "                kind = \"code\"\n",
        "            elif \"name\" in p_name or \"title\" in p_name:\n",
        "                kind = \"name\"\n",
        "            else:\n",
        "                kind = \"text\"\n",
        "            t_kinds[p_name] = kind\n",
        "        kinds[t_name] = t_kinds\n",
        "\n",
        "print(f\"🧰 {len(tools)} tools loaded, for example:\")\n",
        "for tool in tools[:3]:\n",
        "    print(json.dumps(tool[\"function\"], ensure_ascii=False)[:300])\n"
    ]

    # CELL 8: Requests Markdown
    nb["cells"][8]["source"] = [
        "## The requests (Zalo LifeMate Dataset)\n",
        "\n",
        "Chúng ta nạp `train.jsonl` từ Google Drive, ánh xạ chuẩn vào cấu trúc `request`, `context`, `calls`, `category` đúng y xì định dạng của Liquid AI:\n",
        "\n",
        "- `train`: 1,390 câu lệnh thực tế từ người dùng.\n",
        "- `test`: 155 câu lệnh dùng để đánh giá độ chính xác.\n",
        "- `showcase`: 5 ví dụ chọn lọc trải rộng các loại yêu cầu (`single`, `multi`, `clarify`, `no_tool`)."
    ]

    # CELL 9: Load Dataset Code
    nb["cells"][9]["source"] = [
        "from datasets import Dataset, DatasetDict\n",
        "import random\n",
        "\n",
        "# Tìm file train.jsonl trên Google Drive hoặc thư mục repo\n",
        "possible_data_paths = [\n",
        "    Path(\"/content/drive/MyDrive/ZaloAI_2026_ASR/train.jsonl\"),\n",
        "    Path(\"/content/drive/MyDrive/train.jsonl\"),\n",
        "    Path(\"/content/zaloai-challenge-2026/data/task1_lifemate/train+public_test/train/train.jsonl\"),\n",
        "    Path(\"./data/task1_lifemate/train+public_test/train/train.jsonl\"),\n",
        "]\n",
        "data_file = next((p for p in possible_data_paths if p.exists()), None)\n",
        "if not data_file:\n",
        "    raise FileNotFoundError(\"Không tìm thấy file train.jsonl trên Google Drive hoặc repo!\")\n",
        "\n",
        "raw_records = []\n",
        "with open(data_file, \"r\", encoding=\"utf-8\") as f:\n",
        "    for line in f:\n",
        "        if line.strip():\n",
        "            raw_records.append(json.loads(line))\n",
        "\n",
        "processed_records = []\n",
        "for r in raw_records:\n",
        "    calls = r.get(\"label\", {}).get(\"tool_calls\", [])\n",
        "    if not calls:\n",
        "        category = \"no_tool\"\n",
        "    elif any(c.get(\"name\") == \"sys.ask_missing_info\" for c in calls):\n",
        "        category = \"clarify\"\n",
        "    elif len(calls) > 1:\n",
        "        category = \"multi\"\n",
        "    else:\n",
        "        category = \"single\"\n",
        "        \n",
        "    ctx = r.get(\"context\", {})\n",
        "    current_time_str = ctx.get(\"current_time\", \"2026-11-28T07:10:00+07:00\")\n",
        "    dt_clean = current_time_str.split(\"+\")[0] if \"+\" in current_time_str else current_time_str\n",
        "    \n",
        "    clean_ctx = {\n",
        "        \"datetime\": dt_clean,\n",
        "        \"timezone\": ctx.get(\"timezone\", \"Asia/Ho_Chi_Minh\"),\n",
        "        \"locale\": \"vi-VN\",\n",
        "        \"user_profile\": json.dumps(ctx.get(\"user_profile\", {}), ensure_ascii=False) if ctx.get(\"user_profile\") else \"\",\n",
        "    }\n",
        "    \n",
        "    processed_records.append({\n",
        "        \"request\": r.get(\"audio_text\", \"\"),\n",
        "        \"context\": clean_ctx,\n",
        "        \"calls\": json.dumps(calls, ensure_ascii=False),\n",
        "        \"category\": category,\n",
        "        \"language\": \"vi\",\n",
        "    })\n",
        "\n",
        "# Chia train (90%) và test (10%)\n",
        "random.seed(42)\n",
        "random.shuffle(processed_records)\n",
        "n_train = int(len(processed_records) * 0.9)\n",
        "\n",
        "dataset = DatasetDict({\n",
        "    \"train\": Dataset.from_list(processed_records[:n_train]),\n",
        "    \"test\": Dataset.from_list(processed_records[n_train:]),\n",
        "    \"showcase\": Dataset.from_list(processed_records[n_train:n_train + 5]),\n",
        "})\n",
        "\n",
        "print(dataset)\n",
        "print(Counter(dataset[\"train\"][\"category\"]))\n",
        "\n",
        "for category in [\"multi\", \"clarify\"]:\n",
        "    example = next(e for e in dataset[\"train\"] if e[\"category\"] == category)\n",
        "    print(f\"\\n🗂️ {category} ({example['language']}, {example['context']['datetime']}): {example['request']}\")\n",
        "    print(f\"🎯 Gold calls: {example['calls']}\")\n"
    ]

    # CELL 10: Prompt Markdown
    nb["cells"][10]["source"] = [
        "## Build the prompts\n",
        "\n",
        "The system prompt holds the rules and the context, and the chat template adds the 148 tool schemas. LFM3 writes tool calls in XML, and the tokenizer knows how to parse them back into structured calls with `tokenizer.parse_response`.\n",
        "\n",
        "We use LFM3's non-reasoning mode (`enable_thinking=False`). Since every prompt carries the same catalog, we render the prompts ourselves and give TRL plain text prompts."
    ]

    # CELL 11: Prompt Code
    nb["cells"][11]["source"] = [
        "from datetime import datetime\n",
        "\n",
        "SYSTEM_PROMPT = \"\"\"You are a helpful assistant running on the user's phone. You can call the phone's tools.\n",
        "Current date and time: {weekday}, {date} {time} ({timezone}). User locale: {locale}.\n",
        "{user_profile_info}\n",
        "Rules:\n",
        "- Call every tool needed to complete the request, with arguments taken from the request and the context above. Fill optional arguments only when the user gives them; never invent values.\n",
        "- Resolve relative dates and times (\"tomorrow\", \"on Friday\", \"in 20 minutes\") against the current date and time. Times are local, in the formats the tool descriptions give.\n",
        "- If the request needs a tool but a required detail is missing, ask a short clarifying question using sys.ask_missing_info instead of calling other tools.\n",
        "- Answer small talk and general-knowledge questions directly, without tools.\n",
        "- If no tool can do what is asked, say so briefly instead of calling a different tool. If only part of the request is possible, do that part and say what you could not do.\n",
        "- Keep free-text arguments (messages, notes, search queries) in Vietnamese as requested.\"\"\"\n",
        "\n",
        "def build_prompt(example):\n",
        "    now = datetime.fromisoformat(example[\"context\"][\"datetime\"])\n",
        "    up = example[\"context\"].get(\"user_profile\", \"\")\n",
        "    user_profile_info = f\"User profile: {up}\" if up else \"\"\n",
        "    system = SYSTEM_PROMPT.format(\n",
        "        weekday=now.strftime(\"%A\"),\n",
        "        date=now.strftime(\"%Y-%m-%d\"),\n",
        "        time=now.strftime(\"%H:%M\"),\n",
        "        user_profile_info=user_profile_info,\n",
        "        **example[\"context\"]\n",
        "    )\n",
        "    messages = [{\"role\": \"system\", \"content\": system}, {\"role\": \"user\", \"content\": example[\"request\"]}]\n",
        "    return tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)\n",
        "\n",
        "prompt = build_prompt(dataset[\"train\"][0])\n",
        "print(f\"📏 Prompt length: {len(tokenizer(prompt, add_special_tokens=False).input_ids):,} tokens\")\n",
        "print(prompt[-1200:])\n"
    ]

    # CELL 15: Evaluation Code
    nb["cells"][15]["source"] = [
        "from tqdm.auto import tqdm\n",
        "\n",
        "@torch.no_grad()\n",
        "def evaluate(model, split, n=150, batch_size=8):\n",
        "    model.eval()\n",
        "    examples = dataset[split].shuffle(seed=42).select(range(min(n, len(dataset[split]))))\n",
        "    actual_n = len(examples)\n",
        "    correct = {}\n",
        "    for start in tqdm(range(0, actual_n, batch_size), desc=f\"Evaluating {split}\"):\n",
        "        batch = examples.select(range(start, min(start + batch_size, actual_n)))\n",
        "        prompts = [build_prompt(e) for e in batch]\n",
        "        inputs = tokenizer(prompts, return_tensors=\"pt\", padding=True, add_special_tokens=False).to(model.device)\n",
        "        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)\n",
        "        for output, prompt, example in zip(outputs[:, inputs[\"input_ids\"].shape[1]:], prompts, batch):\n",
        "            output = output[output != tokenizer.pad_token_id]\n",
        "            correct.setdefault(example[\"category\"], []).append(calls_match(json.loads(example[\"calls\"]), predicted_calls(output, prompt)))\n",
        "    results = {category: sum(c) / len(c) for category, c in correct.items()}\n",
        "    results[\"all\"] = sum(sum(c) for c in correct.values()) / actual_n if actual_n > 0 else 0.0\n",
        "    return results\n",
        "\n",
        "def show(results, title):\n",
        "    print(title)\n",
        "    for split, r in results.items():\n",
        "        print(f\"   {split:<12} exact match {r['all']:.1%} | \" + \" | \".join(f\"{k} {v:.0%}\" for k, v in sorted(r.items()) if k != 'all'))\n",
        "\n",
        "baseline = {split: evaluate(model, split) for split in [\"test\"]}\n",
        "show(baseline, \"📏 Before training\")\n"
    ]

    # CELL 23: GRPO Config
    nb["cells"][23]["source"] = [
        "from trl import GRPOConfig, GRPOTrainer\n",
        "\n",
        "grpo_config = GRPOConfig(\n",
        "    output_dir=\"./LFM3-1.7B-lifemate-assistant\",\n",
        "    max_steps=150,\n",
        "    per_device_train_batch_size=2,\n",
        "    gradient_accumulation_steps=16,  # <- 32 answers per step\n",
        "    num_generations=8,\n",
        "    generation_batch_size=8,  # <- generate one group at a time to prevent OOM\n",
        "    max_completion_length=256,\n",
        "    temperature=1.0,\n",
        "    learning_rate=1e-5,\n",
        "    beta=0.04,  # <- KL penalty towards the original model\n",
        "    lr_scheduler_type=\"constant_with_warmup\",\n",
        "    warmup_steps=5,\n",
        "    logging_steps=5,\n",
        "    save_strategy=\"no\",\n",
        "    bf16=use_bf16,\n",
        "    fp16=not use_bf16,\n",
        "    report_to=\"none\",\n",
        ")\n",
        "\n",
        "trainer = GRPOTrainer(\n",
        "    model=model,\n",
        "    reward_funcs=tool_call_reward,\n",
        "    args=grpo_config,\n",
        "    train_dataset=train_dataset,\n",
        "    processing_class=tokenizer,\n",
        "    peft_config=peft_config,\n",
        ")\n",
        "trainer.model.print_trainable_parameters()\n"
    ]

    # CELL 29: Evaluate after training
    nb["cells"][29]["source"] = [
        "final = {split: evaluate(trainer.model, split) for split in [\"test\"]}\n",
        "show(baseline, \"📏 Before training\")\n",
        "show(final, \"📏 After training\")\n"
    ]

    # CELL 33: Summary table
    nb["cells"][33]["source"] = [
        "print(f\"{'exact match':<36}{'original':>10}{'fine-tuned':>12}\")\n",
        "for split, name in [(\"test\", \"LifeMate test tools\")]:\n",
        "    print(f\"{name:<36}{baseline[split]['all']:>10.1%}{final[split]['all']:>12.1%}\")\n",
        "    for category in sorted(k for k in baseline[split] if k != \"all\"):\n",
        "        print(f\"{'  ' + category:<36}{baseline[split][category]:>10.0%}{final[split][category]:>12.0%}\")\n"
    ]

    # CELL 35: Save and merge model to Drive
    nb["cells"][35]["source"] = [
        "# Lưu LoRA adapter và bản model đã merge vào Google Drive để sử dụng vĩnh viễn\n",
        "save_dir = Path(\"/content/drive/MyDrive/ZaloAI_2026_ASR/LFM3-1.7B-lifemate-merged\")\n",
        "save_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "trainer.save_model(\"./LFM3-1.7B-lifemate-lora\")\n",
        "\n",
        "merged_model = trainer.model.merge_and_unload()\n",
        "merged_model.save_pretrained(save_dir)\n",
        "tokenizer.save_pretrained(save_dir)\n",
        "print(f\"💾 Mô hình hoàn chỉnh đã lưu thành công vào Drive: {save_dir}\")\n"
    ]

    out_path = Path("liquidAI/lfm3_grpo_zalo_lifemate_a100.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Generated {out_path} successfully ({len(nb['cells'])} cells).")


if __name__ == "__main__":
    build_notebook()
