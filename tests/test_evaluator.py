"""Unit tests for the official LifeMate evaluator and stratification module.

Zalo AI Challenge 2026.
Tests the exact BTC metric:
- Replay on ToolEnv and state_hash verification
- Out-of-scope (|G| == 0) scoring
- Bipartite matching (order-independent)
- Non-fuzzy exact matching & Fuzzy Token-F1 > 0.6 threshold
- Precision, Recall, and F1 calculations
- End-to-end dataset evaluation & slice metrics
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.evaluator import (
    compute_token_f1,
    evaluate_dataset,
    evaluate_sample,
    match_tool_call,
    max_bipartite_matching,
)

# ==============================================================================
# 1. Token-F1 Unit Tests
# ==============================================================================


def test_token_f1_exact_match() -> None:
    text = "Nhắc tôi đi uống cà phê lúc 8h sáng"
    assert compute_token_f1(text, text) == 1.0


def test_token_f1_empty_cases() -> None:
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1(None, None) == 1.0
    assert compute_token_f1("hello", "") == 0.0
    assert compute_token_f1("", "hello") == 0.0
    assert compute_token_f1(None, "hello") == 0.0


def test_token_f1_unicode_nfc_and_folding() -> None:
    # Text with tone variations / lowercase / spaces
    gold = "Phòng Khách - Đèn Trùm"
    sub = "phong khach - den trum"
    f1 = compute_token_f1(sub, gold)
    assert f1 == 1.0


def test_token_f1_threshold_behavior() -> None:
    # 5 gold tokens: 'nhắc', 'họp', 'nhóm', 'chiều', 'nay'
    gold = "nhắc họp nhóm chiều nay"

    # 4 of 5 overlap: P = 4/4 = 1.0, R = 4/5 = 0.8 -> F1 = 0.888 > 0.6
    sub_high = "nhắc họp nhóm chiều"
    assert compute_token_f1(sub_high, gold) > 0.6

    # 1 of 5 overlap: F1 < 0.6
    sub_low = "nhắc mua bánh mì"
    assert compute_token_f1(sub_low, gold) <= 0.6


def test_token_f1_disjoint_returns_zero() -> None:
    assert compute_token_f1("hà nội mùa thu", "sài gòn hoa lệ") == 0.0


# ==============================================================================
# 2. Tool Call Matching Tests
# ==============================================================================


def test_match_tool_call_different_name() -> None:
    sub = {"name": "alarm.create", "arguments": {"time": "07:00"}}
    gold = {"name": "alarm.update", "arguments": {"time": "07:00"}}
    is_match, _ = match_tool_call(sub, gold, fuzzy_params_for_tool={"label"})
    assert not is_match


def test_match_tool_call_non_fuzzy_exact_match() -> None:
    sub = {
        "name": "calendar.update_event",
        "arguments": {"event_id": "evt_01", "start": "2026-11-29T09:00:00+07:00"},
    }
    gold = {
        "name": "calendar.update_event",
        "arguments": {"event_id": "evt_01", "start": "2026-11-29T09:00:00+07:00"},
    }
    is_match, _ = match_tool_call(sub, gold, fuzzy_params_for_tool={"title", "location"})
    assert is_match


def test_match_tool_call_non_fuzzy_mismatch() -> None:
    sub = {
        "name": "calendar.update_event",
        "arguments": {"event_id": "evt_02", "start": "2026-11-29T09:00:00+07:00"},
    }
    gold = {
        "name": "calendar.update_event",
        "arguments": {"event_id": "evt_01", "start": "2026-11-29T09:00:00+07:00"},
    }
    is_match, _ = match_tool_call(sub, gold, fuzzy_params_for_tool={"title", "location"})
    assert not is_match


def test_match_tool_call_fuzzy_parameter_threshold() -> None:
    fuzzy_params = {"content"}
    gold = {
        "name": "sms.send",
        "arguments": {"contact_id": "c_08", "content": "mai chạy bộ dời sang 09:00 nhé"},
    }

    # High overlap > 0.6
    sub_pass = {
        "name": "sms.send",
        "arguments": {"contact_id": "c_08", "content": "mai chay bo doi sang 9h nhe"},
    }
    is_match, scores = match_tool_call(sub_pass, gold, fuzzy_params)
    assert is_match
    assert scores["content"] > 0.6

    # Low overlap <= 0.6
    sub_fail = {
        "name": "sms.send",
        "arguments": {"contact_id": "c_08", "content": "mai đi đá bóng không"},
    }
    is_match, _ = match_tool_call(sub_fail, gold, fuzzy_params)
    assert not is_match


def test_match_tool_call_multiple_fuzzy_params() -> None:
    fuzzy_params = {"title", "location"}
    gold = {
        "name": "calendar.create_event",
        "arguments": {
            "title": "Họp phòng kỹ thuật quý 4",
            "location": "Phòng họp 302 tòa nhà Keangnam",
            "start": "2026-11-29T10:00:00+07:00",
        },
    }

    # Both fuzzy params pass
    sub_pass = {
        "name": "calendar.create_event",
        "arguments": {
            "title": "Họp phòng kỹ thuật Q4",
            "location": "Phòng 302 Keangnam",
            "start": "2026-11-29T10:00:00+07:00",
        },
    }
    is_match, scores = match_tool_call(sub_pass, gold, fuzzy_params)
    assert is_match
    assert scores["title"] > 0.6
    assert scores["location"] > 0.6

    # One fuzzy param fails
    sub_fail = {
        "name": "calendar.create_event",
        "arguments": {
            "title": "Họp phòng kỹ thuật Q4",
            "location": "Quán cafe Highland",
            "start": "2026-11-29T10:00:00+07:00",
        },
    }
    is_match, _ = match_tool_call(sub_fail, gold, fuzzy_params)
    assert not is_match


def test_match_sys_ask_missing_info() -> None:
    sub = {
        "name": "sys.ask_missing_info",
        "arguments": {"tool_name": "sms.send", "params": ["contact_id"]},
    }
    gold = {
        "name": "sys.ask_missing_info",
        "arguments": {"tool_name": "sms.send", "params": ["contact_id"]},
    }
    is_match, _ = match_tool_call(sub, gold, fuzzy_params_for_tool=set())
    assert is_match


# ==============================================================================
# 3. Maximum Bipartite Matching Tests
# ==============================================================================


def test_bipartite_matching_order_independent() -> None:
    gold_calls = [
        {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
        {"name": "calendar.list_events", "arguments": {}},
    ]
    # Reverse order in submission
    sub_calls = [
        {"name": "calendar.list_events", "arguments": {}},
        {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
    ]
    fuzzy_map: dict[str, set[str]] = {}

    matched, unmatched_sub, unmatched_gold = max_bipartite_matching(
        sub_calls, gold_calls, fuzzy_map
    )
    assert len(matched) == 2
    assert len(unmatched_sub) == 0
    assert len(unmatched_gold) == 0


# ==============================================================================
# 4. evaluate_sample Offline & Formula Tests
# ==============================================================================


def test_eval_sample_out_of_scope_correct() -> None:
    res = evaluate_sample(
        sub_tool_calls=[],
        gold_tool_calls=[],
        question_id="test_oos",
        sub_state_hash="hash_a",
        gold_state_hash="hash_a",
        replay=False,
    )
    assert res.score == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0


def test_eval_sample_out_of_scope_incorrect() -> None:
    res = evaluate_sample(
        sub_tool_calls=[{"name": "alarm.list", "arguments": {}}],
        gold_tool_calls=[],
        question_id="test_oos",
        sub_state_hash="hash_a",
        gold_state_hash="hash_a",
        replay=False,
    )
    assert res.score == 0.0
    assert res.precision == 0.0
    assert res.f1 == 0.0


def test_eval_sample_state_hash_mismatch_gives_zero() -> None:
    calls = [{"name": "alarm.list", "arguments": {}}]
    res = evaluate_sample(
        sub_tool_calls=calls,
        gold_tool_calls=calls,
        question_id="test_mismatch",
        sub_state_hash="hash_incorrect",
        gold_state_hash="hash_gold",
        replay=False,
    )
    assert not res.state_match
    assert res.score == 0.0


def test_eval_sample_precision_recall_formula() -> None:
    gold = [
        {"name": "tool_a", "arguments": {}},
        {"name": "tool_b", "arguments": {}},
        {"name": "tool_c", "arguments": {}},
    ]
    # Matched 2 calls, 1 unmatched submitted call -> K = 3, M = 2, U = 1
    sub = [
        {"name": "tool_a", "arguments": {}},
        {"name": "tool_b", "arguments": {}},
        {"name": "tool_extra", "arguments": {}},
    ]
    res = evaluate_sample(
        sub_tool_calls=sub,
        gold_tool_calls=gold,
        question_id="test_prf",
        sub_state_hash="hash_ok",
        gold_state_hash="hash_ok",
        replay=False,
    )
    assert res.state_match
    assert res.m == 2
    assert res.u == 1
    assert pytest.approx(res.precision, 0.001) == 2 / 3  # M / (M + U)
    assert pytest.approx(res.recall, 0.001) == 2 / 3     # M / |G|
    expected_f1 = (2 * (2 / 3) * (2 / 3)) / ((2 / 3) + (2 / 3))
    assert pytest.approx(res.score, 0.001) == expected_f1


# ==============================================================================
# 5. Real ToolEnv Replay Integration Test
# ==============================================================================


def test_eval_sample_with_real_toolenv_replay() -> None:
    # Use real sample from val_mini.jsonl
    val_mini_path = Path("train/val_mini.jsonl")
    if not val_mini_path.is_file():
        val_mini_path = Path("data/val_mini.jsonl")
    assert val_mini_path.is_file(), "val_mini.jsonl must exist"

    with open(val_mini_path, encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    qid = first_line["id"]
    gold_calls = first_line["label"]["tool_calls"]
    gold_final_state = first_line["label"].get("final_state")
    gold_hash = first_line["label"].get("state_hash")

    # Evaluate gold calls replaying on real ToolEnv: must yield 1.0 perfect score
    res = evaluate_sample(
        sub_tool_calls=gold_calls,
        gold_tool_calls=gold_calls,
        question_id=qid,
        sub_state_hash=gold_hash,
        gold_state_hash=gold_hash,
        split="train",
        replay=True,
        gold_final_state=gold_final_state,
    )
    assert res.state_match
    assert res.score == 1.0
    assert res.replayed_state_hash if hasattr(res, "replayed_state_hash") else res.sub_state_hash


# ==============================================================================
# 6. Dataset-Level Evaluation Tests
# ==============================================================================


def test_evaluate_dataset_end_to_end(tmp_path: Path) -> None:
    gold_records = [
        # 1. Out-of-scope (score 1.0)
        {"id": "s_oos", "label": {"tool_calls": [], "state_hash": "h_oos"}},
        # 2. Perfect match (|G| = 1, score 1.0)
        {"id": "s_single", "label": {"tool_calls": [{"name": "app.read", "arguments": {}}], "state_hash": "h_s"}},
        # 3. Partial match (score ~0.8)
        {
            "id": "s_multi",
            "label": {
                "tool_calls": [
                    {"name": "tool_1", "arguments": {}},
                    {"name": "tool_2", "arguments": {}},
                    {"name": "tool_3", "arguments": {}},
                ],
                "state_hash": "h_m",
            },
        },
    ]

    sub_records = [
        {"id": "s_oos", "tool_calls": [], "state_hash": "h_oos"},
        {"id": "s_single", "tool_calls": [{"name": "app.read", "arguments": {}}], "state_hash": "h_s"},
        {
            "id": "s_multi",
            "tool_calls": [
                {"name": "tool_1", "arguments": {}},
                {"name": "tool_2", "arguments": {}},
            ],
            "state_hash": "h_m",
        },
    ]

    gold_file = tmp_path / "gold.jsonl"
    sub_file = tmp_path / "sub.jsonl"

    with open(gold_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in gold_records)
    with open(sub_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in sub_records)

    report = evaluate_dataset(
        submissions=sub_file,
        ground_truth=gold_file,
        replay=False,
    )

    assert report.total_samples == 3
    assert report.out_of_scope_acc == 1.0
    assert report.state_match_rate == 1.0
    assert 0.0 < report.final_score <= 100.0


# ==============================================================================
# 7. Stratified Dataset Split Integrity Tests
# ==============================================================================


def test_stratified_split_integrity() -> None:
    train_path = Path("train/train_split.jsonl")
    val_path = Path("train/val.jsonl")
    mini_path = Path("train/val_mini.jsonl")

    assert train_path.is_file(), "train_split.jsonl missing"
    assert val_path.is_file(), "val.jsonl missing"
    assert mini_path.is_file(), "val_mini.jsonl missing"

    with open(train_path, encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f if line.strip()]
    with open(val_path, encoding="utf-8") as f:
        val_lines = [json.loads(line) for line in f if line.strip()]
    with open(mini_path, encoding="utf-8") as f:
        mini_lines = [json.loads(line) for line in f if line.strip()]

    # 1. Total counts
    assert len(train_lines) == 1236
    assert len(val_lines) == 309
    assert len(mini_lines) == 30

    # 2. No leakage between train and val
    train_ids = {r["id"] for r in train_lines}
    val_ids = {r["id"] for r in val_lines}
    assert len(train_ids & val_ids) == 0, "Data leakage between train and val!"

    # 3. Mini is strict subset of val
    mini_ids = {r["id"] for r in mini_lines}
    assert mini_ids.issubset(val_ids), "val_mini must be a subset of val!"

    # 4. Each split has state_hash attached
    assert all("state_hash" in r.get("label", {}) for r in train_lines[:20])
    assert all("state_hash" in r.get("label", {}) for r in val_lines[:20])
    assert all("state_hash" in r.get("label", {}) for r in mini_lines)


# ==============================================================================
# 8. User-Specified 3 Verification Scenarios
# ==============================================================================


def test_scenario_1_original_gold_scores_100() -> None:
    """Scenario i: Feeding original label.tool_calls must yield 100.0 score."""
    train_path = Path("train/train.jsonl")
    if not train_path.is_file():
        train_path = Path("data/task1_lifemate/train+public_test/train/train.jsonl")

    with open(train_path, encoding="utf-8") as f:
        sample = json.loads(f.readline())

    gold_calls = sample["label"]["tool_calls"]
    gold_final = sample["label"]["final_state"]

    res = evaluate_sample(
        sub_tool_calls=gold_calls,
        gold_tool_calls=gold_calls,
        question_id=sample["id"],
        split="train",
        replay=True,
        gold_final_state=gold_final,
    )

    assert res.state_match
    assert res.score == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.u == 0


def test_scenario_2_redundant_call_increases_u_and_drops_f1() -> None:
    """Scenario ii: Adding a redundant call (contacts.search) increases U by 1 and drops F1."""
    train_path = Path("train/train.jsonl")
    if not train_path.is_file():
        train_path = Path("data/task1_lifemate/train+public_test/train/train.jsonl")

    with open(train_path, encoding="utf-8") as f:
        sample = json.loads(f.readline())  # train_0001 (5 gold calls)

    gold_calls = sample["label"]["tool_calls"]
    gold_final = sample["label"]["final_state"]
    m_gold = len(gold_calls)

    # Insert redundant read-only tool
    sub_calls = copy.deepcopy(gold_calls)
    sub_calls.append({"name": "contacts.search", "arguments": {"query": "Người quen lạ"}})

    res = evaluate_sample(
        sub_tool_calls=sub_calls,
        gold_tool_calls=gold_calls,
        question_id=sample["id"],
        split="train",
        replay=True,
        gold_final_state=gold_final,
    )

    # State hash must still match (read-only call)
    assert res.state_match
    # U increases to 1
    assert res.u == 1
    assert res.m == m_gold
    # Precision drops to M / (M + 1)
    expected_p = m_gold / (m_gold + 1)
    assert pytest.approx(res.precision, 1e-5) == expected_p
    # Recall remains 1.0
    assert pytest.approx(res.recall, 1e-5) == 1.0
    # F1 drops to 2P/(P+1)
    expected_f1 = (2.0 * expected_p) / (expected_p + 1.0)
    assert pytest.approx(res.f1, 1e-5) == expected_f1
    assert pytest.approx(res.score, 1e-5) == expected_f1
    assert 5 in res.unmatched_sub_indices


def test_scenario_3_fuzzy_text_variation_matches() -> None:
    """Scenario iii: Slight phrasing variation in sms.send content (F1 > 0.6) still matches and scores 1.0."""
    train_path = Path("train/train.jsonl")
    if not train_path.is_file():
        train_path = Path("data/task1_lifemate/train+public_test/train/train.jsonl")

    with open(train_path, encoding="utf-8") as f:
        sample = json.loads(f.readline())  # train_0001: sms.send 'mai chạy bộ dời sang 09:00 nhé'

    gold_calls = sample["label"]["tool_calls"]
    gold_final = sample["label"]["final_state"]

    sub_calls = copy.deepcopy(gold_calls)
    # Modify sms.send content to "mai chạy bộ dời sang 9h nhé"
    for c in sub_calls:
        if c["name"] == "sms.send":
            c["arguments"]["content"] = "mai chạy bộ dời sang 9h nhé"

    res = evaluate_sample(
        sub_tool_calls=sub_calls,
        gold_tool_calls=gold_calls,
        question_id=sample["id"],
        split="train",
        replay=True,
        gold_final_state=gold_final,
    )

    # State hash must match because content is a fuzzy free-text field
    assert res.state_match
    # All 5 calls match because token-F1 > 0.6
    assert res.m == len(gold_calls)
    assert res.u == 0
    assert res.score == 1.0
