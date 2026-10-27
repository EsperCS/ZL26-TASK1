"""Official Scorer and Evaluator Module for Zalo AI Challenge 2026 (Task 1: LifeMate).

Implements the exact evaluation metric defined in docs/ZALO_CHALLENGE_RAW.md (Section 3.1):
1. State Match: Replays submitted tool_calls on ToolEnv to verify that replayed state_hash
   matches gold state_hash exactly. If state_hash does not match, score is 0 immediately.
2. Out-of-scope samples (|G| == 0):
   - Score = 1.0 if submitted tool_calls == []
   - Score = 0.0 otherwise
3. Tool-call Match (|G| > 0):
   - One submitted call matches a gold call if:
     * tool name matches exactly.
     * all non-fuzzy arguments match exactly.
     * all fuzzy arguments achieve token-F1 > 0.6 (strictly greater than 0.6)
       after lowercase and Unicode NFC normalization.
   - Maximum 1-to-1 bipartite matching between submitted calls and gold calls
     (order-independent).
   - Precision: P = M / (M + U), where M is matched calls, U is unmatched submitted calls.
   - Recall:    R = M / |G|, where |G| is total gold calls.
   - F1 Score:  F1 = 2PR / (P + R) if P + R > 0 else 0.0.
   - Sample Score = 0 if hash mismatch or M == 0 else F1.
4. Final Leaderboard Score:
   Final Score = (100 / N) * sum(Score_i) for i = 1 to N.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dynamic path resolution for zaic_tools package
try:
    from zaic_tools import ToolEnv, catalog_path
    from zaic_tools.utils import _norm, _tokens, state_hash
    from zaic_tools.validate import catalog_tools
except ImportError:
    for candidate in [Path("train"), Path("public_test"), Path("train+public_test/train")]:
        if (candidate / "zaic_tools").is_dir() and str(candidate.resolve()) not in sys.path:
            sys.path.insert(0, str(candidate.resolve()))
    from zaic_tools import ToolEnv, catalog_path
    from zaic_tools.utils import _norm, _tokens, state_hash
    from zaic_tools.validate import catalog_tools

# Fallback free text keys if catalog is not available
FALLBACK_FREE_TEXT_KEYS = {
    "address", "body", "content", "dropoff", "expression", "label",
    "location", "name", "near", "note", "pickup", "subject", "title",
}


@dataclass(slots=True)
class MatchedCallPair:
    """Details of a single matched submitted and gold tool call."""

    sub_index: int
    gold_index: int
    tool_name: str
    fuzzy_f1_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SampleEvaluation:
    """Evaluation scorecard for a single sample."""

    question_id: str
    score: float
    state_match: bool
    sub_state_hash: str
    gold_state_hash: str
    precision: float
    recall: float
    f1: float
    m: int
    u: int
    g_count: int
    k_count: int
    matches: list[MatchedCallPair] = field(default_factory=list)
    unmatched_sub_indices: list[int] = field(default_factory=list)
    unmatched_gold_indices: list[int] = field(default_factory=list)
    error_message: str | None = None


@dataclass(slots=True)
class DatasetEvaluation:
    """Aggregated evaluation scorecard across an entire evaluation split."""

    total_samples: int
    final_score: float  # 0 to 100
    mean_score: float   # 0 to 1.0
    state_match_rate: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    out_of_scope_acc: float
    ask_missing_acc: float
    multi_step_acc: float
    single_step_acc: float
    sample_evaluations: list[SampleEvaluation] = field(default_factory=list)


def load_catalog_file(cp: Path) -> dict[str, Any] | None:
    """Safely load JSON catalog file using context manager."""
    if not cp.is_file():
        return None
    with open(cp, encoding="utf-8") as f:
        return json.load(f)


def extract_fuzzy_parameters(catalog: dict[str, Any] | None) -> dict[str, set[str]]:
    """Extract tool_name -> set of parameter names marked with 'x-match': 'fuzzy' from catalog."""
    fuzzy_map: dict[str, set[str]] = {}
    if not catalog:
        return fuzzy_map

    tools = catalog_tools(catalog)
    for tool_name, spec in tools.items():
        props = (spec.get("parameters") or {}).get("properties") or {}
        f_params = {k for k, p in props.items() if p.get("x-match") == "fuzzy"}
        if f_params:
            fuzzy_map[tool_name] = f_params
    return fuzzy_map


def compute_token_f1(text_sub: Any, text_gold: Any) -> float:
    """Compute token-level F1 between two text values.

    Normalization:
    1. Unicode NFC normalization and lowercased.
    2. Extract word tokens with regex [^\\W_]+.
    3. Also evaluate diacritic-folded tokens (_tokens from zaic_tools) for fairness.
    Returns the maximum F1 between pure NFC and folded tokens.
    """
    if text_sub is None and text_gold is None:
        return 1.0
    if text_sub is None or text_gold is None:
        return 0.0

    s_sub_raw = str(text_sub).strip()
    s_gold_raw = str(text_gold).strip()

    if not s_sub_raw and not s_gold_raw:
        return 1.0
    if not s_sub_raw or not s_gold_raw:
        return 0.0

    # NFC normalized tokens
    nfc_sub = unicodedata.normalize("NFC", s_sub_raw.lower())
    nfc_gold = unicodedata.normalize("NFC", s_gold_raw.lower())

    tok_nfc_sub = set(re.findall(r"[^\W_]+", nfc_sub))
    tok_nfc_gold = set(re.findall(r"[^\W_]+", nfc_gold))

    def _set_f1(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        if intersection == 0:
            return 0.0
        p = intersection / len(a)
        r = intersection / len(b)
        return (2.0 * p * r) / (p + r)

    f1_nfc = _set_f1(tok_nfc_sub, tok_nfc_gold)

    # Folded tokens (Vietnamese diacritics removed)
    tok_fold_sub = _tokens(s_sub_raw)
    tok_fold_gold = _tokens(s_gold_raw)
    f1_fold = _set_f1(tok_fold_sub, tok_fold_gold)

    return max(f1_nfc, f1_fold)


def are_non_fuzzy_equal(val_sub: Any, val_gold: Any) -> bool:
    """Compare non-fuzzy arguments for exact equality."""
    if val_sub == val_gold:
        return True

    # Allow order independence for list of primitives (e.g. ['contact_id', 'content'])
    if isinstance(val_sub, list) and isinstance(val_gold, list):
        if len(val_sub) != len(val_gold):
            return False
        try:
            return sorted(val_sub) == sorted(val_gold)
        except TypeError:
            return val_sub == val_gold

    # Allow string equivalence with normalized whitespace
    if isinstance(val_sub, str) and isinstance(val_gold, str):
        return _norm(val_sub) == _norm(val_gold)

    return False


def match_tool_call(
    sub_call: dict[str, Any],
    gold_call: dict[str, Any],
    fuzzy_params_for_tool: set[str],
) -> tuple[bool, dict[str, float]]:
    """Determine whether a submitted tool call matches a gold tool call.

    Criteria:
    1. Tool name matches exactly.
    2. All non-fuzzy parameters match exactly.
    3. All fuzzy parameters achieve token-F1 > 0.6.
    Returns: (is_match, dict of parameter -> token_f1 for fuzzy parameters).
    """
    sub_name = sub_call.get("name")
    gold_name = gold_call.get("name")
    if sub_name != gold_name:
        return False, {}

    sub_args = sub_call.get("arguments") or {}
    gold_args = gold_call.get("arguments") or {}

    # 1. Non-fuzzy arguments comparison
    all_non_fuzzy_keys = (set(sub_args.keys()) | set(gold_args.keys())) - fuzzy_params_for_tool
    empty_values = (None, "", [], {})

    for k in all_non_fuzzy_keys:
        v_sub = sub_args.get(k)
        v_gold = gold_args.get(k)

        # Missing in gold but given non-empty in sub -> mismatch
        if k not in gold_args and v_sub not in empty_values:
            return False, {}
        # Missing in sub but required/present in gold -> mismatch
        if k not in sub_args and v_gold not in empty_values:
            return False, {}
        # Both present -> check exact equality
        if k in sub_args and k in gold_args and not are_non_fuzzy_equal(v_sub, v_gold):
            return False, {}

    # 2. Fuzzy arguments comparison
    fuzzy_scores: dict[str, float] = {}
    for k in fuzzy_params_for_tool:
        v_sub = sub_args.get(k)
        v_gold = gold_args.get(k)

        sub_is_empty = v_sub in empty_values
        gold_is_empty = v_gold in empty_values

        if sub_is_empty and gold_is_empty:
            continue
        if sub_is_empty or gold_is_empty:
            # One provided fuzzy content while the other is empty
            return False, {}

        f1 = compute_token_f1(v_sub, v_gold)
        fuzzy_scores[k] = f1
        if f1 <= 0.6:  # Strict inequality required by regulations
            return False, fuzzy_scores

    return True, fuzzy_scores


def max_bipartite_matching(
    sub_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    fuzzy_map: dict[str, set[str]],
) -> tuple[list[MatchedCallPair], list[int], list[int]]:
    """Find maximum 1-to-1 bipartite matching between submitted and gold calls.

    Uses augmenting paths (Kuhn's algorithm) with tie-breaking preference
    for matches with higher fuzzy similarity.
    """
    n_sub = len(sub_calls)
    n_gold = len(gold_calls)

    if n_sub == 0 or n_gold == 0:
        return [], list(range(n_sub)), list(range(n_gold))

    # Build adjacency list: adj[i] = list of (j, avg_f1, fuzzy_scores) where sub_calls[i] matches gold_calls[j]
    adj: list[list[tuple[int, float, dict[str, float]]]] = [[] for _ in range(n_sub)]

    for i, s_call in enumerate(sub_calls):
        tool_name = s_call.get("name", "")
        f_params = fuzzy_map.get(tool_name, set())
        for j, g_call in enumerate(gold_calls):
            is_match, scores = match_tool_call(s_call, g_call, f_params)
            if is_match:
                avg_f1 = sum(scores.values()) / len(scores) if scores else 1.0
                adj[i].append((j, avg_f1, scores))

    # Sort candidates by descending similarity score to prioritize highest-quality matches
    for i in range(n_sub):
        adj[i].sort(key=lambda item: item[1], reverse=True)

    # Kuhn's algorithm for maximum bipartite matching
    match_gold_to_sub = [-1] * n_gold
    gold_match_metadata: dict[int, dict[str, float]] = {}

    def dfs(curr_u: int, visited_nodes: list[bool]) -> bool:
        for v, _, scores in adj[curr_u]:
            if not visited_nodes[v]:
                visited_nodes[v] = True
                if match_gold_to_sub[v] < 0 or dfs(match_gold_to_sub[v], visited_nodes):
                    match_gold_to_sub[v] = curr_u
                    gold_match_metadata[v] = scores
                    return True
        return False

    for u in range(n_sub):
        visited = [False] * n_gold
        dfs(u, visited)

    # Collect matched pairs
    matched_pairs: list[MatchedCallPair] = []
    matched_sub_indices = set()
    matched_gold_indices = set()

    for v, u in enumerate(match_gold_to_sub):
        if u >= 0:
            matched_pairs.append(
                MatchedCallPair(
                    sub_index=u,
                    gold_index=v,
                    tool_name=sub_calls[u].get("name", ""),
                    fuzzy_f1_scores=gold_match_metadata.get(v, {}),
                )
            )
            matched_sub_indices.add(u)
            matched_gold_indices.add(v)

    unmatched_sub = [i for i in range(n_sub) if i not in matched_sub_indices]
    unmatched_gold = [j for j in range(n_gold) if j not in matched_gold_indices]

    return matched_pairs, unmatched_sub, unmatched_gold


def compute_sample_score(
    p: float,
    r: float,
    m: int,
    g_count: int,
    k_count: int,
    state_match: bool,
) -> float:
    """Compute exact sample score according to ZAC regulations."""
    if g_count == 0:
        return 1.0 if k_count == 0 else 0.0

    if not state_match or m == 0:
        return 0.0

    if p + r == 0.0:
        return 0.0

    return (2.0 * p * r) / (p + r)


def evaluate_sample(
    sub_tool_calls: list[dict[str, Any]],
    gold_tool_calls: list[dict[str, Any]],
    question_id: str = "sample_test",
    sub_state_hash: str | None = None,
    gold_state_hash: str | None = None,
    split: str = "train",
    root: Path | None = None,
    replay: bool = True,
    catalog: dict[str, Any] | None = None,
    gold_final_state: dict[str, Any] | None = None,
) -> SampleEvaluation:
    """Evaluate a single sample against ground truth according to BTC formula."""
    # 1. Prepare tools catalog & fuzzy map
    if catalog is None:
        cp = catalog_path(split, root)
        catalog = load_catalog_file(cp)
    fuzzy_map = extract_fuzzy_parameters(catalog)

    # 2. State hash verification & ToolEnv replay
    replayed_sub_hash = sub_state_hash
    replayed_gold_hash = gold_state_hash
    state_match = True
    err_msg: str | None = None

    if replay and question_id:
        try:
            # Replay submitted tool_calls on ToolEnv
            env = ToolEnv(split=split, question_id=question_id, root=root)
            for call in sub_tool_calls:
                c_name = call.get("name", "")
                c_args = call.get("arguments") or {}
                env.call(c_name, c_args)
            replayed_sub_hash = env.get_state_hash()

            # Obtain gold state hash
            if not replayed_gold_hash:
                if gold_final_state is not None:
                    # Fast computation from initial state + final_state
                    init_env = ToolEnv(split=split, question_id=question_id, root=root)
                    combined_state = init_env.get_state()
                    for app, changed_st in gold_final_state.items():
                        combined_state[app] = changed_st
                    replayed_gold_hash = state_hash(combined_state)
                else:
                    # Replay gold calls
                    gold_env = ToolEnv(split=split, question_id=question_id, root=root)
                    for call in gold_tool_calls:
                        gold_env.call(call.get("name", ""), call.get("arguments") or {})
                    replayed_gold_hash = gold_env.get_state_hash()

            # Verify: replayed state hash must match gold state hash
            state_match = (replayed_sub_hash == replayed_gold_hash)

            # If participant provided sub_state_hash, it must also match replayed hash
            if sub_state_hash and sub_state_hash != replayed_sub_hash:
                state_match = False
                err_msg = (
                    f"Submission state_hash {sub_state_hash[:8]}... differs from "
                    f"replay state_hash {replayed_sub_hash[:8]}..."
                )

        except (KeyError, ValueError, FileNotFoundError, TypeError, AttributeError, RuntimeError) as e:
            state_match = False
            err_msg = f"ToolEnv replay failed: {type(e).__name__}: {e}"
            replayed_sub_hash = replayed_sub_hash or "error"
            replayed_gold_hash = replayed_gold_hash or "error"
    else:
        # Offline or mock evaluation
        replayed_sub_hash = sub_state_hash or ""
        replayed_gold_hash = gold_state_hash or ""
        if sub_state_hash is not None and gold_state_hash is not None:
            state_match = (sub_state_hash == gold_state_hash)

    # 3. Tool call bipartite matching
    k_count = len(sub_tool_calls)
    g_count = len(gold_tool_calls)

    matched_pairs, unmatched_sub, unmatched_gold = max_bipartite_matching(
        sub_calls=sub_tool_calls,
        gold_calls=gold_tool_calls,
        fuzzy_map=fuzzy_map,
    )

    m = len(matched_pairs)
    u = k_count - m

    if g_count == 0:
        if k_count == 0:
            precision, recall, f1, score = 1.0, 1.0, 1.0, 1.0
        else:
            precision, recall, f1, score = 0.0, 0.0, 0.0, 0.0
    else:
        precision = m / (m + u) if (m + u) > 0 else 0.0
        recall = m / g_count
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        score = compute_sample_score(
            p=precision,
            r=recall,
            m=m,
            g_count=g_count,
            k_count=k_count,
            state_match=state_match,
        )

    return SampleEvaluation(
        question_id=question_id,
        score=score,
        state_match=state_match,
        sub_state_hash=replayed_sub_hash or "",
        gold_state_hash=replayed_gold_hash or "",
        precision=precision,
        recall=recall,
        f1=f1,
        m=m,
        u=u,
        g_count=g_count,
        k_count=k_count,
        matches=matched_pairs,
        unmatched_sub_indices=unmatched_sub,
        unmatched_gold_indices=unmatched_gold,
        error_message=err_msg,
    )


def evaluate_dataset(
    submissions: list[dict[str, Any]] | Path,
    ground_truth: list[dict[str, Any]] | Path,
    split: str = "train",
    root: Path | None = None,
    replay: bool = True,
    catalog: dict[str, Any] | None = None,
) -> DatasetEvaluation:
    """Evaluate full participant submissions against ground truth dataset."""
    # Load files if paths provided
    if isinstance(submissions, (str, Path)):
        p_sub = Path(submissions)
        if not p_sub.is_file():
            raise FileNotFoundError(
                f"Tệp submission không tồn tại: '{p_sub}'. "
                "Bạn cần chạy Agent/Pipeline để xuất file kết quả nộp bài trước khi chấm điểm!"
            )
        with open(p_sub, encoding="utf-8") as f:
            sub_list = [json.loads(line) for line in f if line.strip()]
    else:
        sub_list = submissions

    if isinstance(ground_truth, (str, Path)):
        p_gold = Path(ground_truth)
        if not p_gold.is_file():
            raise FileNotFoundError(
                f"Tệp ground truth không tồn tại: '{p_gold}'. Vui lòng kiểm tra lại đường dẫn!"
            )
        with open(p_gold, encoding="utf-8") as f:
            gold_list = [json.loads(line) for line in f if line.strip()]
    else:
        gold_list = ground_truth

    sub_by_id = {row["id"]: row for row in sub_list}
    gold_by_id = {row["id"]: row for row in gold_list}

    if catalog is None:
        cp = catalog_path(split, root)
        catalog = load_catalog_file(cp)

    sample_evals: list[SampleEvaluation] = []
    oos_correct = oos_total = 0
    ask_correct = ask_total = 0
    multi_correct = multi_total = 0
    single_correct = single_total = 0

    for qid, gold_record in gold_by_id.items():
        sub_record = sub_by_id.get(qid, {"id": qid, "tool_calls": [], "state_hash": None})

        sub_calls = sub_record.get("tool_calls") or []
        sub_hash = sub_record.get("state_hash")

        gold_label = gold_record.get("label", {})
        gold_calls = gold_label.get("tool_calls") or []
        gold_hash = gold_label.get("state_hash")
        gold_final_state = gold_label.get("final_state")

        eval_res = evaluate_sample(
            sub_tool_calls=sub_calls,
            gold_tool_calls=gold_calls,
            question_id=qid,
            sub_state_hash=sub_hash,
            gold_state_hash=gold_hash,
            split=split,
            root=root,
            replay=replay,
            catalog=catalog,
            gold_final_state=gold_final_state,
        )
        sample_evals.append(eval_res)

        # Categorical slice tracking
        is_oos = len(gold_calls) == 0
        has_ask = any(c.get("name") == "sys.ask_missing_info" for c in gold_calls)
        is_multi = len(gold_calls) > 1

        if is_oos:
            oos_total += 1
            if eval_res.score == 1.0:
                oos_correct += 1
        else:
            if has_ask:
                ask_total += 1
                if eval_res.score == 1.0:
                    ask_correct += 1
            if is_multi:
                multi_total += 1
                if eval_res.score == 1.0:
                    multi_correct += 1
            elif len(gold_calls) == 1 and not has_ask:
                single_total += 1
                if eval_res.score == 1.0:
                    single_correct += 1

    n = len(sample_evals)
    if n == 0:
        return DatasetEvaluation(
            total_samples=0,
            final_score=0.0,
            mean_score=0.0,
            state_match_rate=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            out_of_scope_acc=0.0,
            ask_missing_acc=0.0,
            multi_step_acc=0.0,
            single_step_acc=0.0,
        )

    mean_score = sum(s.score for s in sample_evals) / n
    final_score = (100.0 / n) * sum(s.score for s in sample_evals)
    state_match_rate = sum(1 for s in sample_evals if s.state_match) / n
    mean_p = sum(s.precision for s in sample_evals) / n
    mean_r = sum(s.recall for s in sample_evals) / n
    mean_f1 = sum(s.f1 for s in sample_evals) / n

    return DatasetEvaluation(
        total_samples=n,
        final_score=final_score,
        mean_score=mean_score,
        state_match_rate=state_match_rate,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_f1=mean_f1,
        out_of_scope_acc=oos_correct / oos_total if oos_total > 0 else 1.0,
        ask_missing_acc=ask_correct / ask_total if ask_total > 0 else 1.0,
        multi_step_acc=multi_correct / multi_total if multi_total > 0 else 1.0,
        single_step_acc=single_correct / single_total if single_total > 0 else 1.0,
        sample_evaluations=sample_evals,
    )


def print_evaluation_report(report: DatasetEvaluation) -> None:
    """Print beautifully formatted evaluation report to stdout."""
    print("\n" + "=" * 65)
    print("===  ZALO AI CHALLENGE 2026 - LIFEMATE OFFICIAL SCORECARD  ===")
    print("=" * 65)
    print(f"Total Evaluated Samples : {report.total_samples:,d}")
    print(f"FINAL COMPETITION SCORE : {report.final_score:.4f} / 100.0")
    print(f"Mean Sample Score (0-1) : {report.mean_score:.4f}")
    print(f"State Hash Match Rate   : {report.state_match_rate * 100:.2f}%")
    print("-" * 65)
    print(f"Macro Precision (P)     : {report.mean_precision:.4f}")
    print(f"Macro Recall (R)        : {report.mean_recall:.4f}")
    print(f"Macro F1 Score          : {report.mean_f1:.4f}")
    print("-" * 65)
    print("Slice Accuracy Breakdown:")
    print(f" * Out-of-scope ([])    : {report.out_of_scope_acc * 100:.2f}%")
    print(f" * sys.ask_missing_info : {report.ask_missing_acc * 100:.2f}%")
    print(f" * Multi-step (> 1 call): {report.multi_step_acc * 100:.2f}%")
    print(f" * Single-step (1 call) : {report.single_step_acc * 100:.2f}%")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Evaluator for Zalo AI Challenge 2026 (Task 1: LifeMate)."
    )
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Path to participant submission.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to ground truth dataset (val.jsonl or train.jsonl)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name for ToolEnv environment (default: train)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Data root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip ToolEnv replay and rely solely on submitted state_hash",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full evaluation report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(
        submissions=args.submission,
        ground_truth=args.gold,
        split=args.split,
        root=args.root,
        replay=not args.no_replay,
    )
    print_evaluation_report(report)

    if args.output_json:
        data = {
            "total_samples": report.total_samples,
            "final_score": report.final_score,
            "mean_score": report.mean_score,
            "state_match_rate": report.state_match_rate,
            "mean_precision": report.mean_precision,
            "mean_recall": report.mean_recall,
            "mean_f1": report.mean_f1,
            "slices": {
                "out_of_scope_acc": report.out_of_scope_acc,
                "ask_missing_acc": report.ask_missing_acc,
                "multi_step_acc": report.multi_step_acc,
                "single_step_acc": report.single_step_acc,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
