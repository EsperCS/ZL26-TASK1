"""Self-verification script for the 3 test scenarios requested by the user.

Zalo AI Challenge 2026.
Scenarios:
1. Feed original label.tool_calls of train.jsonl into evaluator -> Score must be 100.0.
2. Insert 1 redundant tool call (not changing state_hash, e.g. contacts.search)
   -> Verify U increases by 1 and F1 decreases strictly according to the formula.
3. Slightly modify text phrasing in content of sms.send (e.g. "mai chạy bộ dời sang 9h nhé")
   -> Verify token-F1 > 0.6 works and matches correctly.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# Ensure UTF-8 console output across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.evaluator import compute_token_f1, evaluate_sample


def run_scenario_1(sample: dict) -> None:
    print("\n" + "=" * 65)
    print("SCENARIO 1: ĐƯA NGUYÊN LABEL.TOOL_CALLS VÀO HÀM CHẤM")
    print("=" * 65)
    gold_calls = sample["label"]["tool_calls"]
    gold_final_state = sample["label"]["final_state"]
    qid = sample["id"]

    res = evaluate_sample(
        sub_tool_calls=gold_calls,
        gold_tool_calls=gold_calls,
        question_id=qid,
        split="train",
        replay=True,
        gold_final_state=gold_final_state,
    )

    print(f"Sample ID               : {res.question_id}")
    print(f"Total Gold Calls (|G|)  : {res.g_count}")
    print(f"Total Submitted Calls   : {res.k_count}")
    print(f"State Hash Matched      : {res.state_match}")
    print(f"Matched Calls (M)       : {res.m}")
    print(f"Unmatched Calls (U)     : {res.u}")
    print(f"Precision (P)           : {res.precision:.4f}")
    print(f"Recall (R)              : {res.recall:.4f}")
    print(f"F1 Score                : {res.f1:.4f}")
    print(f"Sample Score            : {res.score:.4f} (Kỳ vọng: 1.0)")
    assert res.score == 1.0, f"Scenario 1 failed! Expected 1.0, got {res.score}"
    assert res.state_match, "Scenario 1 failed! State hash did not match."
    print(">>> KẾT QUẢ: ĐẠT 100.0% (THÀNH CÔNG)")


def run_scenario_2(sample: dict) -> None:
    print("\n" + "=" * 65)
    print("SCENARIO 2: CHÈN THÊM 1 LỆNH GỌI DƯ THỪA (contacts.search)")
    print("=" * 65)
    gold_calls = sample["label"]["tool_calls"]
    gold_final_state = sample["label"]["final_state"]
    qid = sample["id"]

    # Insert redundant read-only tool call: contacts.search
    sub_calls = copy.deepcopy(gold_calls)
    redundant_call = {"name": "contacts.search", "arguments": {"query": "Người quen lạ"}}
    sub_calls.append(redundant_call)

    res = evaluate_sample(
        sub_tool_calls=sub_calls,
        gold_tool_calls=gold_calls,
        question_id=qid,
        split="train",
        replay=True,
        gold_final_state=gold_final_state,
    )

    m = len(gold_calls)
    u_expected = 1
    p_expected = m / (m + u_expected)
    r_expected = 1.0
    f1_expected = (2.0 * p_expected * r_expected) / (p_expected + r_expected)

    print(f"Sample ID               : {res.question_id}")
    print(f"Total Gold Calls (|G|)  : {res.g_count}")
    print(f"Total Submitted Calls   : {res.k_count} (Chèn thêm 1 call)")
    print(f"State Hash Matched      : {res.state_match} (Read-only không đổi state_hash)")
    print(f"Matched Calls (M)       : {res.m} (Kỳ vọng: {m})")
    print(f"Unmatched Calls (U)     : {res.u} (Kỳ vọng tăng lên: {u_expected})")
    print(f"Precision (P = M/(M+U)) : {res.precision:.6f} (Kỳ vọng: {p_expected:.6f})")
    print(f"Recall (R = M/|G|)      : {res.recall:.6f} (Kỳ vọng: {r_expected:.6f})")
    print(f"F1 Score (2PR/(P+R))    : {res.f1:.6f} (Kỳ vọng: {f1_expected:.6f})")
    print(f"Sample Score            : {res.score:.6f}")
    print(f"Unmatched Indices (sub) : {res.unmatched_sub_indices}")

    assert res.state_match, "Scenario 2 failed! State hash altered unexpectedly."
    assert res.u == u_expected, f"Scenario 2 failed! U expected {u_expected}, got {res.u}"
    assert abs(res.precision - p_expected) < 1e-6, "Scenario 2 failed! Precision mismatch"
    assert abs(res.f1 - f1_expected) < 1e-6, "Scenario 2 failed! F1 formula mismatch"
    assert abs(res.score - f1_expected) < 1e-6, "Scenario 2 failed! Score mismatch"
    print(">>> KẾT QUẢ: U TĂNG LÊN 1, F1 GIẢM CHÍNH XÁC THEO CÔNG THỨC (THÀNH CÔNG)")


def run_scenario_3(sample: dict) -> None:
    print("\n" + "=" * 65)
    print("SCENARIO 3: SỬA NHẸ CÂU CHỮ TRƯỜNG CONTENT CỦA sms.send")
    print("=" * 65)
    gold_calls = sample["label"]["tool_calls"]
    gold_final_state = sample["label"]["final_state"]
    qid = sample["id"]

    gold_sms = next(c for c in gold_calls if c["name"] == "sms.send")
    gold_content = gold_sms["arguments"]["content"]
    sub_content = "mai chạy bộ dời sang 9h nhé"

    # Compute raw token F1
    token_f1 = compute_token_f1(sub_content, gold_content)
    print(f"Gold content            : '{gold_content}'")
    print(f"Submitted content       : '{sub_content}'")
    print(f"Token-F1 score          : {token_f1:.4f} (Ngưỡng yêu cầu: > 0.6)")

    # Create sub_calls with modified content
    sub_calls = copy.deepcopy(gold_calls)
    for c in sub_calls:
        if c["name"] == "sms.send":
            c["arguments"]["content"] = sub_content

    res = evaluate_sample(
        sub_tool_calls=sub_calls,
        gold_tool_calls=gold_calls,
        question_id=qid,
        split="train",
        replay=True,
        gold_final_state=gold_final_state,
    )

    print(f"Sample ID               : {res.question_id}")
    print(f"State Hash Matched      : {res.state_match} (Fuzzy param không đổi state_hash)")
    print(f"Matched Calls (M)       : {res.m} (Tất cả tool đều khớp)")
    print(f"Unmatched Calls (U)     : {res.u}")
    print(f"Precision (P)           : {res.precision:.4f}")
    print(f"Recall (R)              : {res.recall:.4f}")
    print(f"F1 Score                : {res.f1:.4f}")
    print(f"Sample Score            : {res.score:.4f} (Kỳ vọng: 1.0)")

    sms_match = next((m for m in res.matches if m.tool_name == "sms.send"), None)
    assert sms_match is not None, "Scenario 3 failed! sms.send call was not matched."
    print(f"sms.send Match Scores   : {sms_match.fuzzy_f1_scores}")

    assert token_f1 > 0.6, f"Scenario 3 failed! Token-F1 was {token_f1} <= 0.6"
    assert res.state_match, "Scenario 3 failed! State hash did not match."
    assert res.score == 1.0, f"Scenario 3 failed! Expected 1.0, got {res.score}"
    print(">>> KẾT QUẢ: TOKEN-F1 > 0.6 HOẠT ĐỘNG CHUẨN XÁC, ĐIỂM 1.0 (THÀNH CÔNG)")


def main() -> None:
    train_path = Path("train/train.jsonl")
    if not train_path.is_file():
        train_path = Path("data/task1_lifemate/train+public_test/train/train.jsonl")

    with open(train_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # train_0001 has sms.send with 'mai chạy bộ dời sang 09:00 nhé'
    sample = records[0]

    run_scenario_1(sample)
    run_scenario_2(sample)
    run_scenario_3(sample)

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ 3 KỊCH BẢN ĐÃ ĐƯỢC XÁC THỰC THÀNH CÔNG 100%!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
