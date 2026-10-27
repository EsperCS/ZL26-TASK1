"""Unit tests for EDA and Data Quality script for Task 1: LifeMate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eda_task1 import (
    AppCatalogEntry,
    analyze_train_samples,
    compute_audio_stats,
    load_tools_catalog,
    perform_data_quality_checks,
    render_ascii_bar,
)


def test_compute_audio_stats_empty() -> None:
    """Ensure empty duration list produces zeroed AudioStats."""
    stats = compute_audio_stats([])
    assert stats.count == 0
    assert stats.min_sec == 0.0
    assert stats.max_sec == 0.0
    assert stats.mean_sec == 0.0


def test_compute_audio_stats_values() -> None:
    """Ensure stats calculation matches expected mathematical values."""
    durations = [2.0, 4.0, 6.0, 8.0, 10.0]
    stats = compute_audio_stats(durations)
    assert stats.count == 5
    assert stats.min_sec == 2.0
    assert stats.max_sec == 10.0
    assert stats.mean_sec == 6.0
    assert stats.median_sec == 6.0
    assert stats.buckets["< 3s"] == 1
    assert stats.buckets["3s - 5s"] == 1
    assert stats.buckets["5s - 8s"] == 1
    assert stats.buckets["8s - 10s"] == 1
    assert stats.buckets["10s - 15s"] == 1


def test_render_ascii_bar() -> None:
    """Test ASCII bar chart renderer."""
    bar_full = render_ascii_bar(10, 10, width=10)
    assert bar_full == "[##########]"

    bar_empty = render_ascii_bar(0, 10, width=10)
    assert bar_empty == "[          ]"

    bar_half = render_ascii_bar(5, 10, width=10)
    assert bar_half == "[#####     ]"


def test_analyze_train_samples_classification() -> None:
    """Test sample classification across normal, ask_missing, out-of-scope, and history."""
    samples = [
        # 1. Normal tool execution (2 calls)
        {
            "id": "s1",
            "history": [],
            "context": {"installed_apps": {"weather": {}}},
            "label": {
                "tool_calls": [
                    {"name": "weather.get_current"},
                    {"name": "weather.get_forecast"},
                ]
            },
        },
        # 2. Solo ask_missing_info (1 call)
        {
            "id": "s2",
            "history": [
                {"role": "user", "audio": "a1.wav"},
                {"role": "assistant", "text": "t1"},
            ],
            "context": {"installed_apps": {"calendar": {}}},
            "label": {
                "tool_calls": [
                    {
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "x", "params": []},
                    }
                ]
            },
        },
        # 3. Multistep with ask_missing_info (2 calls)
        {
            "id": "s3",
            "history": [
                {"role": "user", "audio": "a1.wav"},
                {"role": "assistant", "text": "t1"},
                {"role": "user", "audio": "a2.wav"},
                {"role": "assistant", "text": "t2"},
            ],
            "context": {"installed_apps": {"contacts": {}}},
            "label": {
                "tool_calls": [
                    {"name": "contacts.search"},
                    {
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "x", "params": []},
                    },
                ]
            },
        },
        # 4. Out-of-scope (0 calls)
        {
            "id": "s4",
            "history": [],
            "context": {"installed_apps": {}},
            "label": {"tool_calls": []},
        },
    ]

    breakdown, tc_stats, installed, called = analyze_train_samples(samples)

    assert breakdown.total_samples == 4
    assert breakdown.normal_tools == 1
    assert breakdown.ask_missing_info == 2
    assert breakdown.ask_missing_solo == 1
    assert breakdown.ask_missing_multistep == 1
    assert breakdown.out_of_scope == 1
    assert breakdown.history_any == 2
    assert breakdown.history_gt2 == 1  # sample 3 has 4 messages > 2

    assert tc_stats.min_calls == 0
    assert tc_stats.max_calls == 2
    assert tc_stats.frequency[0] == 1
    assert tc_stats.frequency[1] == 1
    assert tc_stats.frequency[2] == 2

    assert installed["weather"] == 1
    assert called["weather.get_current"] == 1
    assert called["contacts.search"] == 1


def test_load_tools_catalog(tmp_path: Path) -> None:
    """Test loading and parsing tools catalog JSON."""
    catalog_data = {
        "apps": [
            {
                "id": "app_test",
                "name_vi": "Ứng dụng thử nghiệm",
                "group": "utility",
                "description": "Mô tả test",
                "tools": [{"name": "app_test.do_action"}],
            }
        ]
    }
    cat_file = tmp_path / "catalog.json"
    cat_file.write_text(json.dumps(catalog_data), encoding="utf-8")

    entries = load_tools_catalog(cat_file)
    assert len(entries) == 1
    assert entries[0].app_id == "app_test"
    assert entries[0].name_vi == "Ứng dụng thử nghiệm"
    assert entries[0].tools == ["app_test.do_action"]


def test_data_quality_checks_detects_flaws(tmp_path: Path) -> None:
    """Test that data quality framework catches schema flaws."""
    # Create empty dirs
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    test_dir = tmp_path / "test"
    test_dir.mkdir()

    # Flawed samples
    train_samples = [
        {
            "id": "dup_01",
            "audio": "audio/missing.wav",
            "context": {"installed_apps": {"unknown_app": {}}},
            "label": {"tool_calls": [{"name": "unknown_app.tool1"}]},
        },
        {
            "id": "dup_01",  # Duplicate ID
            "audio": "audio/missing.wav",
            "context": {},
            # Missing label block
        },
    ]

    test_samples = [
        {
            "id": "dup_01",  # Cross-split collision
            "audio": "audio/test_missing.wav",
        }
    ]

    catalog_entries = [
        AppCatalogEntry(
            app_id="known_app",
            name_vi="Known",
            group="core",
            description="",
            tools=["known_app.tool"],
        )
    ]

    results = perform_data_quality_checks(
        train_records=train_samples,
        test_records=test_samples,
        catalog_entries=catalog_entries,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    failures = {r.check_name: r.passed for r in results}
    assert failures["Sample ID uniqueness within splits"] is False
    assert failures["Zero ID collision between Train and Test"] is False
    assert failures["Physical .wav files exist on disk"] is False
    assert failures["Train labels present"] is False
    assert failures["Installed apps defined in catalog"] is False
    assert failures["Label tool_calls defined in catalog"] is False
