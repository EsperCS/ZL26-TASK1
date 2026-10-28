"""Evaluation package for Zalo AI Challenge 2026 (Task 1: LifeMate)."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluator import (
        DatasetEvaluation,
        SampleEvaluation,
        compute_token_f1,
        evaluate_dataset,
        evaluate_sample,
        match_tool_call,
    )
    from .zalo_bfcl_evaluator import (
        ParsedThinkingOutput,
        SampleBFCLMetrics,
        ZaloBFCLEvaluator,
        ZaloBFCLSummary,
        parse_thinking_and_tool_calls,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "DatasetEvaluation",
        "SampleEvaluation",
        "compute_token_f1",
        "evaluate_dataset",
        "evaluate_sample",
        "match_tool_call",
    }:
        from . import evaluator as _ev

        return getattr(_ev, name)
    if name in {
        "ParsedThinkingOutput",
        "SampleBFCLMetrics",
        "ZaloBFCLEvaluator",
        "ZaloBFCLSummary",
        "parse_thinking_and_tool_calls",
    }:
        from . import zalo_bfcl_evaluator as _zbfcl

        return getattr(_zbfcl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetEvaluation",
    "ParsedThinkingOutput",
    "SampleBFCLMetrics",
    "SampleEvaluation",
    "ZaloBFCLEvaluator",
    "ZaloBFCLSummary",
    "compute_token_f1",
    "evaluate_dataset",
    "evaluate_sample",
    "match_tool_call",
    "parse_thinking_and_tool_calls",
]
