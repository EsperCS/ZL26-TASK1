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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetEvaluation",
    "SampleEvaluation",
    "compute_token_f1",
    "evaluate_dataset",
    "evaluate_sample",
    "match_tool_call",
]
