"""Stratified Train / Validation Split Generator for Task 1: LifeMate.

Zalo AI Challenge 2026.
Splits train.jsonl into:
- 80% Train (train_split.jsonl)
- 20% Validation (val.jsonl)
- 30-sample fast smoke test (val_mini.jsonl)

Guarantees exact stratified representation across:
1. Out-of-scope samples (|tool_calls| == 0)
2. sys.ask_missing_info samples (single-step and multi-step)
3. Multi-step samples (|tool_calls| > 1)
4. Normal single-step samples
Precomputes and attaches gold state_hash to each record's label for instant, deterministic scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure UTF-8 console output across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class StratumSpec:
    """Definition and metadata for a dataset stratification category."""

    key: str
    description: str


STRATA = [
    StratumSpec("out_of_scope", "Out-of-scope ([] empty calls)"),
    StratumSpec("ask_missing_multi", "sys.ask_missing_info (Multi-step)"),
    StratumSpec("ask_missing_single", "sys.ask_missing_info (Single-step)"),
    StratumSpec("normal_multi", "Normal tools (Multi-step, len > 1)"),
    StratumSpec("normal_single", "Normal tools (Single-step, len == 1)"),
]


def classify_sample(record: dict[str, Any]) -> str:
    """Classify a train record into one of 5 mutually exclusive strata."""
    calls = record.get("label", {}).get("tool_calls", [])
    if not calls:
        return "out_of_scope"

    has_ask = any(c.get("name") == "sys.ask_missing_info" for c in calls)
    is_multi = len(calls) > 1

    if has_ask:
        return "ask_missing_multi" if is_multi else "ask_missing_single"
    return "normal_multi" if is_multi else "normal_single"


def locate_file(candidates: list[Path], description: str) -> Path:
    """Find the first existing file among candidate paths."""
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not locate {description}. Checked:\n"
        + "\n".join(f" - {p.resolve()}" for p in candidates)
    )


def compute_gold_state_hash(
    record: dict[str, Any],
    split: str = "train",
    root: Path | None = None,
) -> str:
    """Compute canonical state_hash for a gold record using zaic_tools ToolEnv."""
    try:
        from zaic_tools import ToolEnv
        from zaic_tools.utils import state_hash
    except ImportError:
        # Dynamic fallback if zaic_tools is in train/
        for cand in [Path("train"), Path("public_test"), Path("train+public_test/train")]:
            if (cand / "zaic_tools").is_dir() and str(cand.resolve()) not in sys.path:
                sys.path.insert(0, str(cand.resolve()))
        from zaic_tools import ToolEnv
        from zaic_tools.utils import state_hash

    qid = record["id"]
    env = ToolEnv(split=split, question_id=qid, root=root)
    state = env.get_state()
    for app, changed_state in record.get("label", {}).get("final_state", {}).items():
        state[app] = changed_state
    return state_hash(state)


def create_stratified_splits(
    records: list[dict[str, Any]],
    val_ratio: float = 0.20,
    mini_size: int = 30,
    seed: int = 42,
    compute_hash: bool = True,
    split_name: str = "train",
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Perform deterministic stratified split into train, val, and val_mini sets."""
    rng = random.Random(seed)

    # 1. Group records by stratum
    stratum_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = classify_sample(rec)
        stratum_buckets[key].append(rec)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    val_mini_records: list[dict[str, Any]] = []

    total_count = len(records)
    print(f"\n[Stratification] Total records: {total_count:,d} across {len(stratum_buckets)} strata (seed={seed})")

    # Allocate mini counts proportional to stratum sizes
    stratum_mini_counts: dict[str, int] = {}
    allocated_mini = 0
    stratum_order = [s.key for s in STRATA]

    for key in stratum_order:
        count = len(stratum_buckets[key])
        n_mini = round(mini_size * (count / total_count))
        stratum_mini_counts[key] = n_mini
        allocated_mini += n_mini

    # Adjust rounding discrepancy for val_mini
    diff = mini_size - allocated_mini
    if diff != 0:
        # Adjust largest stratum
        largest_key = max(stratum_order, key=lambda k: len(stratum_buckets[k]))
        stratum_mini_counts[largest_key] += diff

    print("\n| Stratum Key | Description | Total | Train (80%) | Val (20%) | Val-Mini (30) |")
    print("|---|---|---:|---:|---:|---:|")

    for spec in STRATA:
        key = spec.key
        items = list(stratum_buckets[key])
        rng.shuffle(items)

        n_total = len(items)
        n_val = round(n_total * val_ratio)
        n_train = n_total - n_val

        train_subset = items[:n_train]
        val_subset = items[n_train:]

        n_mini = stratum_mini_counts[key]
        mini_subset = val_subset[:n_mini]

        train_records.extend(train_subset)
        val_records.extend(val_subset)
        val_mini_records.extend(mini_subset)

        print(
            f"| `{key}` | {spec.description} | {n_total} ({n_total/total_count*100:.1f}%) "
            f"| {len(train_subset)} ({len(train_subset)/n_total*100:.1f}%) "
            f"| {len(val_subset)} ({len(val_subset)/n_total*100:.1f}%) "
            f"| {len(mini_subset)} ({len(mini_subset)/mini_size*100:.1f}%) |"
        )

    # Shuffle combined sets deterministically for balanced sequential reading
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(val_mini_records)

    # 2. Attach state_hash to labels if requested
    if compute_hash:
        print("\n[State Hash] Precomputing gold state_hash for all splits...")
        hash_cache: dict[str, str] = {}
        all_unique = {rec["id"]: rec for rec in train_records + val_records}
        for i, (qid, rec) in enumerate(all_unique.items(), start=1):
            if "state_hash" not in rec.get("label", {}):
                h = compute_gold_state_hash(rec, split=split_name, root=root)
                rec.setdefault("label", {})["state_hash"] = h
                hash_cache[qid] = h
            if i % 300 == 0 or i == len(all_unique):
                print(f"  Processed {i}/{len(all_unique)} samples...")

    return train_records, val_records, val_mini_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Save records to a JSONL file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    print(f"  -> Saved {len(records):>5,d} records to {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified train / val / val_mini splits for Task 1: LifeMate."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input train.jsonl (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Target directory to write output jsonl splits (default: data/)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20 for 80/20 split)",
    )
    parser.add_argument(
        "--mini-size",
        type=int,
        default=30,
        help="Number of samples in val_mini fast smoke test (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip precomputing gold state_hash in label",
    )
    parser.add_argument(
        "--copy-to-train",
        action="store_true",
        default=True,
        help="Also copy split files to train/ directory (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Locate input train.jsonl
    candidates = [
        args.input,
        Path("train/train.jsonl"),
        Path("data/task1_lifemate/train+public_test/train/train.jsonl"),
        Path("train+public_test/train/train.jsonl"),
    ]
    train_path = locate_file([p for p in candidates if p is not None], "train.jsonl")
    print(f"[Loading] Source file: {train_path}")

    with open(train_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    train_rec, val_rec, val_mini_rec = create_stratified_splits(
        records=records,
        val_ratio=args.val_ratio,
        mini_size=args.mini_size,
        seed=args.seed,
        compute_hash=not args.no_hash,
    )

    out_dir = args.output_dir
    print(f"\n[Writing Output] Writing splits to {out_dir.resolve()}...")
    train_out = out_dir / "train_split.jsonl"
    val_out = out_dir / "val.jsonl"
    val_mini_out = out_dir / "val_mini.jsonl"

    write_jsonl(train_out, train_rec)
    write_jsonl(val_out, val_rec)
    write_jsonl(val_mini_out, val_mini_rec)

    if args.copy_to_train and Path("train").is_dir():
        print("\n[Mirroring Output] Mirroring splits to train/ folder for convenience...")
        write_jsonl(Path("train/train_split.jsonl"), train_rec)
        write_jsonl(Path("train/val.jsonl"), val_rec)
        write_jsonl(Path("train/val_mini.jsonl"), val_mini_rec)

    print("\n✅ Stratified split generation completed successfully!")
    print(f"   Train split: {len(train_rec):,d} samples ({len(train_rec)/len(records)*100:.1f}%)")
    print(f"   Val split:   {len(val_rec):,d} samples ({len(val_rec)/len(records)*100:.1f}%)")
    print(f"   Mini split:  {len(val_mini_rec):,d} samples")


if __name__ == "__main__":
    main()
