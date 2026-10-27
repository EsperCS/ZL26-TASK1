"""Exploratory Data Analysis and Data Quality Validation for Task 1: LifeMate.

Zalo AI Challenge 2026.
Enforces Data Quality Frameworks (Completeness, Uniqueness, Validity, Consistency)
and modern strictly-typed Python 3.12+ engineering practices.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(slots=True, kw_only=True)
class AudioStats:
    """Summary statistics for audio durations in seconds."""

    count: int
    min_sec: float
    max_sec: float
    mean_sec: float
    median_sec: float
    std_sec: float
    p25_sec: float
    p75_sec: float
    p90_sec: float
    p95_sec: float
    p99_sec: float
    buckets: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class ExecutionBreakdown:
    """Breakdown of sample execution behaviors in train set."""

    total_samples: int
    normal_tools: int
    ask_missing_info: int
    ask_missing_solo: int
    ask_missing_multistep: int
    out_of_scope: int
    history_any: int  # len(history) > 0
    history_gt2: int  # len(history) > 2 (deep multi-turn)
    history_distribution: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class ToolCallStats:
    """Statistics of tool_calls counts per sample."""

    total_samples: int
    min_calls: int
    max_calls: int
    mean_calls: float
    median_calls: float
    std_calls: float
    frequency: dict[int, int] = field(default_factory=dict)
    active_mean_calls: float  # Mean for samples with tool_calls > 0


@dataclass(slots=True, kw_only=True)
class AppCatalogEntry:
    """Information about an application in the tools catalog."""

    app_id: str
    name_vi: str
    group: str
    description: str
    tools: list[str]
    installed_in_train: int = 0
    called_in_train: int = 0


@dataclass(slots=True, kw_only=True)
class QualityCheckResult:
    """Data quality test outcome for validation scorecard."""

    dimension: str
    check_name: str
    passed: bool
    details: str


def compute_audio_stats(durations: list[float]) -> AudioStats:
    """Compute comprehensive descriptive statistics and bucket histogram for durations."""
    if not durations:
        return AudioStats(
            count=0,
            min_sec=0.0,
            max_sec=0.0,
            mean_sec=0.0,
            median_sec=0.0,
            std_sec=0.0,
            p25_sec=0.0,
            p75_sec=0.0,
            p90_sec=0.0,
            p95_sec=0.0,
            p99_sec=0.0,
            buckets={},
        )

    dur_sorted = sorted(durations)
    n = len(dur_sorted)

    def percentile(p: float) -> float:
        """Linear interpolation percentile for exact reporting."""
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return dur_sorted[int(k)]
        d0 = dur_sorted[int(f)] * (c - k)
        d1 = dur_sorted[int(c)] * (k - f)
        return d0 + d1

    buckets: dict[str, int] = {
        "< 3s": 0,
        "3s - 5s": 0,
        "5s - 8s": 0,
        "8s - 10s": 0,
        "10s - 15s": 0,
        ">= 15s": 0,
    }

    for d in dur_sorted:
        if d < 3.0:
            buckets["< 3s"] += 1
        elif d < 5.0:
            buckets["3s - 5s"] += 1
        elif d < 8.0:
            buckets["5s - 8s"] += 1
        elif d < 10.0:
            buckets["8s - 10s"] += 1
        elif d < 15.0:
            buckets["10s - 15s"] += 1
        else:
            buckets[">= 15s"] += 1

    return AudioStats(
        count=n,
        min_sec=dur_sorted[0],
        max_sec=dur_sorted[-1],
        mean_sec=statistics.mean(dur_sorted),
        median_sec=statistics.median(dur_sorted),
        std_sec=statistics.stdev(dur_sorted) if n > 1 else 0.0,
        p25_sec=percentile(25.0),
        p75_sec=percentile(75.0),
        p90_sec=percentile(90.0),
        p95_sec=percentile(95.0),
        p99_sec=percentile(99.0),
        buckets=buckets,
    )


def read_single_wav_duration(path: Path) -> float | None:
    """Read WAV duration in seconds from file header safely."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError, EOFError):
        return None


def read_audio_durations_parallel(
    paths: list[Path], max_workers: int = 16
) -> list[float]:
    """Parallelized WAV duration extractor via ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(read_single_wav_duration, paths))
    return [r for r in results if r is not None]


def locate_file(candidates: list[Path], description: str = "Required file") -> Path:
    """Resolve file from candidate paths or raise FileNotFoundError."""
    for cand in candidates:
        if cand.is_file():
            return cand
    attempted = "\n  - ".join(str(c.resolve()) for c in candidates)
    msg = f"Could not locate {description}. Tried:\n  - {attempted}"
    raise FileNotFoundError(msg)


def analyze_train_samples(
    train_records: list[dict],
) -> tuple[ExecutionBreakdown, ToolCallStats, dict[str, int], dict[str, int]]:
    """Perform in-depth analysis on train set samples."""
    total = len(train_records)
    normal_tools = 0
    ask_missing_info = 0
    ask_missing_solo = 0
    ask_missing_multistep = 0
    out_of_scope = 0

    history_any = 0
    history_gt2 = 0
    history_dist: dict[int, int] = {}

    tool_counts: list[int] = []
    tool_dist: dict[int, int] = {}

    installed_apps_count: dict[str, int] = {}
    called_tools_count: dict[str, int] = {}

    for item in train_records:
        history = item.get("history", [])
        h_len = len(history)
        history_dist[h_len] = history_dist.get(h_len, 0) + 1
        if h_len > 0:
            history_any += 1
        if h_len > 2:
            history_gt2 += 1

        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        if isinstance(installed_apps, dict):
            for app_name in installed_apps:
                installed_apps_count[app_name] = (
                    installed_apps_count.get(app_name, 0) + 1
                )

        label = item.get("label", {})
        tool_calls = label.get("tool_calls", [])
        tc_len = len(tool_calls)
        tool_counts.append(tc_len)
        tool_dist[tc_len] = tool_dist.get(tc_len, 0) + 1

        tool_names = [tc.get("name", "") for tc in tool_calls]
        for tname in tool_names:
            called_tools_count[tname] = called_tools_count.get(tname, 0) + 1

        if tc_len == 0:
            out_of_scope += 1
        elif any("sys.ask_missing_info" in t for t in tool_names):
            ask_missing_info += 1
            if tc_len == 1:
                ask_missing_solo += 1
            else:
                ask_missing_multistep += 1
        else:
            normal_tools += 1

    active_counts = [c for c in tool_counts if c > 0]
    active_mean = statistics.mean(active_counts) if active_counts else 0.0

    breakdown = ExecutionBreakdown(
        total_samples=total,
        normal_tools=normal_tools,
        ask_missing_info=ask_missing_info,
        ask_missing_solo=ask_missing_solo,
        ask_missing_multistep=ask_missing_multistep,
        out_of_scope=out_of_scope,
        history_any=history_any,
        history_gt2=history_gt2,
        history_distribution=dict(sorted(history_dist.items())),
    )

    tc_stats = ToolCallStats(
        total_samples=total,
        min_calls=min(tool_counts) if tool_counts else 0,
        max_calls=max(tool_counts) if tool_counts else 0,
        mean_calls=statistics.mean(tool_counts) if tool_counts else 0.0,
        median_calls=statistics.median(tool_counts) if tool_counts else 0.0,
        std_calls=statistics.stdev(tool_counts) if len(tool_counts) > 1 else 0.0,
        frequency=dict(sorted(tool_dist.items())),
        active_mean_calls=active_mean,
    )

    return breakdown, tc_stats, installed_apps_count, called_tools_count


def load_tools_catalog(catalog_path: Path) -> list[AppCatalogEntry]:
    """Parse and load tools catalog JSON into structured AppCatalogEntry objects."""
    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)

    apps_raw = data.get("apps", [])
    catalog_entries: list[AppCatalogEntry] = []

    for app in apps_raw:
        app_id = app.get("id", "")
        name_vi = app.get("name_vi", "")
        group = app.get("group", "")
        description = app.get("description", "")
        tools = [t.get("name", "") for t in app.get("tools", [])]

        catalog_entries.append(
            AppCatalogEntry(
                app_id=app_id,
                name_vi=name_vi,
                group=group,
                description=description,
                tools=tools,
            )
        )
    return catalog_entries


def perform_data_quality_checks(
    train_records: list[dict],
    test_records: list[dict],
    catalog_entries: list[AppCatalogEntry],
    train_dir: Path,
    test_dir: Path,
) -> list[QualityCheckResult]:
    """Execute Data Quality Framework checks across 4 core dimensions."""
    results: list[QualityCheckResult] = []

    # 1. COMPLETENESS
    # Check ID completeness
    missing_train_id = sum(1 for r in train_records if not r.get("id"))
    missing_test_id = sum(1 for r in test_records if not r.get("id"))
    results.append(
        QualityCheckResult(
            dimension="Completeness",
            check_name="Sample ID non-null",
            passed=(missing_train_id == 0 and missing_test_id == 0),
            details=f"Train missing ID: {missing_train_id}, Test missing ID: {missing_test_id}",
        )
    )

    # Check Audio field completeness
    missing_train_audio = sum(1 for r in train_records if not r.get("audio"))
    missing_test_audio = sum(1 for r in test_records if not r.get("audio"))
    results.append(
        QualityCheckResult(
            dimension="Completeness",
            check_name="Audio path present in manifest",
            passed=(missing_train_audio == 0 and missing_test_audio == 0),
            details=f"Train missing audio: {missing_train_audio}, Test missing: {missing_test_audio}",
        )
    )

    # Check Physical Audio file existence
    missing_train_audio_files = 0
    for r in train_records:
        rel = r.get("audio", "")
        if not (train_dir / rel).is_file():
            missing_train_audio_files += 1

    missing_test_audio_files = 0
    for r in test_records:
        rel = r.get("audio", "")
        if not (test_dir / rel).is_file():
            missing_test_audio_files += 1

    results.append(
        QualityCheckResult(
            dimension="Completeness",
            check_name="Physical .wav files exist on disk",
            passed=(missing_train_audio_files == 0 and missing_test_audio_files == 0),
            details=(
                f"Train missing files: {missing_train_audio_files}/{len(train_records)}, "
                f"Test missing files: {missing_test_audio_files}/{len(test_records)}"
            ),
        )
    )

    # Check Train Label completeness
    missing_labels = sum(1 for r in train_records if "label" not in r)
    results.append(
        QualityCheckResult(
            dimension="Completeness",
            check_name="Train labels present",
            passed=(missing_labels == 0),
            details=f"Missing label blocks: {missing_labels}/{len(train_records)}",
        )
    )

    # 2. UNIQUENESS
    train_ids = [r["id"] for r in train_records if "id" in r]
    test_ids = [r["id"] for r in test_records if "id" in r]
    dup_train = len(train_ids) - len(set(train_ids))
    dup_test = len(test_ids) - len(set(test_ids))
    overlap = len(set(train_ids).intersection(set(test_ids)))

    results.append(
        QualityCheckResult(
            dimension="Uniqueness",
            check_name="Sample ID uniqueness within splits",
            passed=(dup_train == 0 and dup_test == 0),
            details=f"Train duplicate IDs: {dup_train}, Test duplicate IDs: {dup_test}",
        )
    )

    results.append(
        QualityCheckResult(
            dimension="Uniqueness",
            check_name="Zero ID collision between Train and Test",
            passed=(overlap == 0),
            details=f"Overlapping IDs: {overlap}",
        )
    )

    # 3. CONSISTENCY
    catalog_apps_set = {entry.app_id for entry in catalog_entries}
    catalog_tools_set = {tool for entry in catalog_entries for tool in entry.tools}

    installed_unknown_apps: set[str] = set()
    for r in train_records:
        installed = r.get("context", {}).get("installed_apps", {})
        if isinstance(installed, dict):
            for app_name in installed:
                if app_name not in catalog_apps_set:
                    installed_unknown_apps.add(app_name)

    results.append(
        QualityCheckResult(
            dimension="Consistency",
            check_name="Installed apps defined in catalog",
            passed=(len(installed_unknown_apps) == 0),
            details=f"Unknown installed apps in train: {sorted(installed_unknown_apps) or 'None'}",
        )
    )

    unrecognized_tools: set[str] = set()
    for r in train_records:
        for tc in r.get("label", {}).get("tool_calls", []):
            tname = tc.get("name", "")
            if tname not in catalog_tools_set:
                unrecognized_tools.add(tname)

    results.append(
        QualityCheckResult(
            dimension="Consistency",
            check_name="Label tool_calls defined in catalog",
            passed=(len(unrecognized_tools) == 0),
            details=f"Unrecognized tool calls: {sorted(unrecognized_tools) or 'None'}",
        )
    )

    # 4. VALIDITY
    # Check sys.ask_missing_info schema validity
    invalid_ask_params = 0
    for r in train_records:
        for tc in r.get("label", {}).get("tool_calls", []):
            if tc.get("name") == "sys.ask_missing_info":
                args = tc.get("arguments", {})
                if (
                    not isinstance(args, dict)
                    or "tool_name" not in args
                    or "params" not in args
                ):
                    invalid_ask_params += 1

    results.append(
        QualityCheckResult(
            dimension="Validity",
            check_name="sys.ask_missing_info payload conformance",
            passed=(invalid_ask_params == 0),
            details=f"Malformed sys.ask_missing_info calls: {invalid_ask_params}",
        )
    )

    return results


def render_ascii_bar(count: int, total: int, width: int = 24) -> str:
    """Render a visual ASCII progress / distribution bar."""
    if total <= 0:
        return "[" + " " * width + "]"
    filled = round((count / total) * width)
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + " " * (width - filled) + "]"


def print_eda_report(
    train_records: list[dict],
    test_records: list[dict],
    train_audio_stats: AudioStats,
    train_history_audio_stats: AudioStats,
    test_audio_stats: AudioStats,
    test_history_audio_stats: AudioStats,
    breakdown: ExecutionBreakdown,
    tc_stats: ToolCallStats,
    catalog_entries: list[AppCatalogEntry],
    quality_results: list[QualityCheckResult],
) -> None:
    """Print the complete, beautifully formatted EDA & Data Quality Report."""
    w = 88
    separator = "=" * w
    sub_sep = "-" * w

    print("\n" + separator)
    print(" ZALO AI CHALLENGE 2026 - TASK 1 (LIFEMATE) DATA ANALYSIS REPORT")
    print(" Standards: data-quality-frameworks | python-pro (Python 3.12+)")
    print(separator)

    # SECTION 1
    print("\n" + "1. TỔNG SỐ MẪU VÀ PHÂN BỐ ĐỘ DÀI FILE .WAV")
    print(sub_sep)
    n_train = len(train_records)
    n_test = len(test_records)
    n_total = n_train + n_test

    print(
        f" * Tổng số mẫu huấn luyện (train.jsonl)      : {n_train:>6,d} ({n_train / n_total * 100:.2f}%)"
    )
    print(
        f" * Tổng số mẫu kiểm thử   (public_test.jsonl) : {n_test:>6,d} ({n_test / n_total * 100:.2f}%)"
    )
    print(f" * Tổng toàn bộ mẫu       (Total Manifest)    : {n_total:>6,d} (100.00%)")
    print()

    # Detailed Audio Duration Table
    print(" [Bảng Thống Kê Chi Tiết Độ Dài File .WAV (Đơn vị: Giây)]")
    hdr = f"{'Audio Partition':<22} | {'Count':>6} | {'Min':>5} | {'Mean':>6} | {'Median':>6} | {'Max':>6} | {'Std':>5} | {'P25':>5} | {'P75':>5} | {'P95':>5}"
    print(hdr)
    print("-" * len(hdr))

    partitions = [
        ("Train (Primary)", train_audio_stats),
        ("Train (History)", train_history_audio_stats),
        ("Test  (Primary)", test_audio_stats),
        ("Test  (History)", test_history_audio_stats),
    ]

    for label, stat in partitions:
        print(
            f"{label:<22} | {stat.count:>6d} | {stat.min_sec:>5.2f} | "
            f"{stat.mean_sec:>6.2f} | {stat.median_sec:>6.2f} | {stat.max_sec:>6.2f} | "
            f"{stat.std_sec:>5.2f} | {stat.p25_sec:>5.2f} | {stat.p75_sec:>5.2f} | {stat.p95_sec:>5.2f}"
        )
    print("-" * len(hdr))

    # Histogram of Primary Audio Durations
    print("\n [Phân Bố Độ Dài Audio Lượt Chính (Primary Request Audio)]")
    print(
        f"{'Khoảng thời gian':<14} | {'Train':>7} | {'%':>6} | {'Test':>6} | {'%':>6} | {'Biểu đồ phân phối Train'}"
    )
    print("-" * 75)
    for bucket in train_audio_stats.buckets:
        c_tr = train_audio_stats.buckets.get(bucket, 0)
        p_tr = (
            (c_tr / train_audio_stats.count * 100) if train_audio_stats.count else 0.0
        )
        c_te = test_audio_stats.buckets.get(bucket, 0)
        p_te = (c_te / test_audio_stats.count * 100) if test_audio_stats.count else 0.0
        bar = render_ascii_bar(c_tr, train_audio_stats.count, width=20)
        print(
            f"{bucket:<14} | {c_tr:>7d} | {p_tr:>5.1f}% | {c_te:>6d} | {p_te:>5.1f}% | {bar}"
        )

    # SECTION 2
    print("\n" + separator)
    print("2. TỶ LỆ % CÁC MẪU THEO NHÓM HÀNH ĐỘNG VÀ LỊCH SỬ HỘI THOẠI")
    print(sub_sep)
    print(" (a) Gọi tool bình thường (Normal Tool Execution)")
    print(
        f"     -> Số lượng: {breakdown.normal_tools:>5d} mẫu | Tỷ lệ: {breakdown.normal_tools / n_train * 100:>6.2f}%"
    )
    print()
    print(" (b) Gọi sys.ask_missing_info (Chủ động hỏi lại thông tin thiếu)")
    print(
        f"     -> Số lượng: {breakdown.ask_missing_info:>5d} mẫu | Tỷ lệ: {breakdown.ask_missing_info / n_train * 100:>6.2f}%"
    )
    print(
        f"        * Chỉ gọi hỏi thông tin thiếu (Solo ask) : {breakdown.ask_missing_solo:>4d} mẫu ({breakdown.ask_missing_solo / n_train * 100:.2f}%)"
    )
    print(
        f"        * Đọc dữ liệu trước rồi hỏi (Multi-step): {breakdown.ask_missing_multistep:>4d} mẫu ({breakdown.ask_missing_multistep / n_train * 100:.2f}%)"
    )
    print()
    print(" (c) Out-of-scope (Không thể xử lý -> tool_calls == [])")
    print(
        f"     -> Số lượng: {breakdown.out_of_scope:>5d} mẫu | Tỷ lệ: {breakdown.out_of_scope / n_train * 100:>6.2f}%"
    )
    print()
    print(
        f" [*] Tổng kiểm tra 3 nhóm độc lập (a + b + c): {breakdown.normal_tools + breakdown.ask_missing_info + breakdown.out_of_scope} / {n_train} (100.00%)"
    )
    print()
    print(" (d) Có lịch sử hội thoại (Conversation History Analysis)")
    print(
        f"     -> Mẫu có lịch sử thoại (history >= 1 lượt / len > 0): {breakdown.history_any:>5d} mẫu | {breakdown.history_any / n_train * 100:>6.2f}%"
    )
    print(
        f"     -> Mẫu nhiều lượt thoại sâu (multi-turn / len > 2) : {breakdown.history_gt2:>5d} mẫu | {breakdown.history_gt2 / n_train * 100:>6.2f}%"
    )
    print(
        f"     -> Mẫu đơn lượt (single-turn / len == 0)            : {n_train - breakdown.history_any:>5d} mẫu | {(n_train - breakdown.history_any) / n_train * 100:>6.2f}%"
    )
    print("\n     [Bảng Phân Bố Độ Dài History (Số messages trong lịch sử)]")
    print(
        f"     {'Số lượt thoại (Messages)':<26} | {'Số mẫu':>7} | {'Tỷ lệ %':>8} | {'Biểu đồ phân phối'}"
    )
    print("     " + "-" * 65)
    for h_cnt, s_cnt in breakdown.history_distribution.items():
        h_pct = s_cnt / n_train * 100
        h_bar = render_ascii_bar(s_cnt, n_train, width=18)
        turn_desc = (
            f"{h_cnt} msgs ({h_cnt // 2} turn pairs)"
            if h_cnt > 0
            else "0 msgs (single-turn)"
        )
        print(f"     {turn_desc:<26} | {s_cnt:>7d} | {h_pct:>7.2f}% | {h_bar}")

    # SECTION 3
    print("\n" + separator)
    print("3. PHÂN BỐ SỐ LƯỢNG TOOL_CALLS VÀ DANH SÁCH ỨNG DỤNG / CATALOG TOOLS")
    print(sub_sep)
    print(" [Phân Bố Số Lượng Tool Calls Trên Mỗi Mẫu]")
    print(f" * Min tool_calls/mẫu  : {tc_stats.min_calls}")
    print(f" * Max tool_calls/mẫu  : {tc_stats.max_calls}")
    print(
        f" * Trung bình (Toàn bộ): {tc_stats.mean_calls:.3f} calls/mẫu (Độ lệch chuẩn: {tc_stats.std_calls:.2f})"
    )
    print(
        f" * Trung bình (Active) : {tc_stats.active_mean_calls:.3f} calls/mẫu (chỉ tính mẫu có thực thi tool)"
    )
    print(f" * Trung vị (Median)   : {tc_stats.median_calls:.1f} calls/mẫu")
    print()
    print(
        f"{'Số tool calls':<15} | {'Số mẫu':>8} | {'Tỷ lệ %':>8} | {'Biểu đồ tần suất'}"
    )
    print("-" * 55)
    for calls, freq in tc_stats.frequency.items():
        pct = freq / n_train * 100
        bar = render_ascii_bar(freq, n_train, width=18)
        print(f"{calls:<15} | {freq:>8d} | {pct:>7.2f}% | {bar}")

    # Catalog App List
    total_catalog_tools = sum(len(e.tools) for e in catalog_entries)
    print(
        f"\n [Danh Sách Toàn Bộ Các Ứng Dụng Trong Catalog ({len(catalog_entries)} Apps, {total_catalog_tools} Tools)]"
    )
    app_hdr = f"{'STT':<3} | {'App ID':<20} | {'Tên Ứng Dụng':<22} | {'Nhóm':<15} | {'#Tools':>6} | {'Cài Đặt':>7} | {'Gọi Train':>9}"
    print(app_hdr)
    print("-" * len(app_hdr))

    for idx, app in enumerate(catalog_entries, start=1):
        name_trunc = (app.name_vi[:20] + "..") if len(app.name_vi) > 22 else app.name_vi
        group_trunc = (app.group[:13] + "..") if len(app.group) > 15 else app.group
        print(
            f"{idx:<3d} | {app.app_id:<20} | {name_trunc:<22} | {group_trunc:<15} | "
            f"{len(app.tools):>6d} | {app.installed_in_train:>7d} | {app.called_in_train:>9d}"
        )
    print("-" * len(app_hdr))

    # SECTION 4
    print("\n" + separator)
    print("4. DATA QUALITY SCORECARD (BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU)")
    print(sub_sep)
    q_hdr = f"{'Dimension':<15} | {'Quality Rule Check':<42} | {'Status':<6} | {'Chi Tiết Kết Quả'}"
    print(q_hdr)
    print("-" * 96)
    passed_all = True
    for q in quality_results:
        status_str = "PASS" if q.passed else "FAIL"
        if not q.passed:
            passed_all = False
        print(f"{q.dimension:<15} | {q.check_name:<42} | {status_str:<6} | {q.details}")
    print("-" * 96)
    if passed_all:
        print(
            " -> ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU: TẤT CẢ CÁC TIÊU CHÍ ĐỀU ĐẠT TIÊU CHUẨN HOÀN TOÀN (100% PASS)!\n"
        )
    else:
        print(
            " -> CẢNH BÁO: Phát hiện vi phạm tiêu chuẩn chất lượng dữ liệu! Vui lòng kiểm tra các mục FAIL bên trên.\n"
        )
    print(separator + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments with sensible default paths."""
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis and Quality Audit for Task 1: LifeMate."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Path to train.jsonl (auto-detected if omitted)",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="Path to public_test.jsonl (auto-detected if omitted)",
    )
    parser.add_argument(
        "--tools-catalog",
        type=Path,
        default=None,
        help="Path to train_tools_catalog.json (auto-detected if omitted)",
    )
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=16,
        help="Number of worker threads for parallel audio header reading (default: 16)",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path to save terminal report to text file",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint for EDA Task 1."""
    args = parse_arguments()

    # 1. Locate files
    train_candidates = [
        Path("train/train.jsonl"),
        Path("data/task1_lifemate/train+public_test/train/train.jsonl"),
        Path("train+public_test/train/train.jsonl"),
    ]
    test_candidates = [
        Path("public_test/public_test.jsonl"),
        Path("data/task1_lifemate/train+public_test/public_test/public_test.jsonl"),
        Path("train+public_test/public_test/public_test.jsonl"),
    ]
    catalog_candidates = [
        Path("train/train_tools_catalog.json"),
        Path("data/task1_lifemate/train+public_test/train/train_tools_catalog.json"),
        Path("train+public_test/train/train_tools_catalog.json"),
    ]

    train_path = args.train_jsonl or locate_file(train_candidates, "train.jsonl")
    test_path = args.test_jsonl or locate_file(test_candidates, "public_test.jsonl")
    catalog_path = args.tools_catalog or locate_file(
        catalog_candidates, "train_tools_catalog.json"
    )

    # 2. Load JSON / JSONL
    with open(train_path, encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]

    with open(test_path, encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    catalog_entries = load_tools_catalog(catalog_path)

    # 3. Analyze samples & tool calls
    breakdown, tc_stats, installed_counts, called_counts = analyze_train_samples(
        train_records
    )

    # Enrich catalog entries with dataset metrics
    for entry in catalog_entries:
        entry.installed_in_train = installed_counts.get(entry.app_id, 0)
        # Sum calls for tools belonging to this app
        entry.called_in_train = sum(called_counts.get(t, 0) for t in entry.tools)

    # 4. Resolve and extract audio durations in parallel
    train_dir = train_path.parent
    test_dir = test_path.parent

    train_primary_audio = [
        train_dir / r["audio"] for r in train_records if "audio" in r
    ]
    train_history_audio = [
        train_dir / h["audio"]
        for r in train_records
        for h in r.get("history", [])
        if h.get("role") == "user" and "audio" in h
    ]

    test_primary_audio = [test_dir / r["audio"] for r in test_records if "audio" in r]
    test_history_audio = [
        test_dir / h["audio"]
        for r in test_records
        for h in r.get("history", [])
        if h.get("role") == "user" and "audio" in h
    ]

    train_primary_durs = read_audio_durations_parallel(
        train_primary_audio, max_workers=args.audio_workers
    )
    train_history_durs = read_audio_durations_parallel(
        train_history_audio, max_workers=args.audio_workers
    )
    test_primary_durs = read_audio_durations_parallel(
        test_primary_audio, max_workers=args.audio_workers
    )
    test_history_durs = read_audio_durations_parallel(
        test_history_audio, max_workers=args.audio_workers
    )

    train_audio_stats = compute_audio_stats(train_primary_durs)
    train_hist_audio_stats = compute_audio_stats(train_history_durs)
    test_audio_stats = compute_audio_stats(test_primary_durs)
    test_hist_audio_stats = compute_audio_stats(test_history_durs)

    # 5. Data Quality Checks
    quality_results = perform_data_quality_checks(
        train_records=train_records,
        test_records=test_records,
        catalog_entries=catalog_entries,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    # 6. If requested to save output to file
    if args.save_report:
        import io

        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_eda_report(
                train_records=train_records,
                test_records=test_records,
                train_audio_stats=train_audio_stats,
                train_history_audio_stats=train_hist_audio_stats,
                test_audio_stats=test_audio_stats,
                test_history_audio_stats=test_hist_audio_stats,
                breakdown=breakdown,
                tc_stats=tc_stats,
                catalog_entries=catalog_entries,
                quality_results=quality_results,
            )
            report_text = buf.getvalue()
        finally:
            sys.stdout = old_stdout

        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(report_text, encoding="utf-8")
        print(f"Report saved successfully to: {args.save_report}")
        print(report_text)
    else:
        # Print directly to stdout
        print_eda_report(
            train_records=train_records,
            test_records=test_records,
            train_audio_stats=train_audio_stats,
            train_history_audio_stats=train_hist_audio_stats,
            test_audio_stats=test_audio_stats,
            test_history_audio_stats=test_hist_audio_stats,
            breakdown=breakdown,
            tc_stats=tc_stats,
            catalog_entries=catalog_entries,
            quality_results=quality_results,
        )


if __name__ == "__main__":
    main()
