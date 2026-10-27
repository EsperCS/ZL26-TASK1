"""
src/eval/asr_metrics.py

Module đánh giá đa chiều cho hệ thống Automatic Speech Recognition (ASR):
1. Normalized WER & CER (%): Đánh giá thuần chất lượng âm học (Acoustic Model).
2. Raw WER & CER (%): Đánh giá chất lượng văn bản hoàn chỉnh sau khôi phục hoa/thường và dấu câu.
3. Exact Sentence Match (%): Tỷ lệ khớp tuyệt đối 100% cả ở dạng Normalized và Raw.
4. Number/Digit Recall (%): Tỷ lệ nhận diện chính xác các con số (phục vụ tham số cho ToolEnv).
5. Capitalized Entity Recall (%): Tỷ lệ khôi phục viết hoa tên riêng (người, địa danh, dịch vụ).
6. Latency & RTF: Độ trễ trung bình (ms/sample) và Real-Time Factor (thời gian chạy / thời lượng âm thanh).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import jiwer


def normalize_text_for_acoustic_eval(text: str) -> str:
    """
    Chuẩn hóa văn bản cho đánh giá âm học thuần túy:
    - Unicode NFC normalization.
    - Chuyển về chữ thường (.lower()).
    - Xóa toàn bộ ký tự dấu câu và ký tự đặc biệt, chỉ giữ chữ và số.
    - Chuẩn hóa khoảng trắng dư thừa.
    """
    if not text:
        return ""
    # NFC
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Xóa dấu câu: thay bất kỳ ký tự không phải chữ/số/khoảng trắng bằng space
    text = re.sub(r"[^\w\s]", " ", text)
    # Gộp khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(text: str) -> list[str]:
    """Trích xuất tất cả các chuỗi chữ số liên tiếp từ văn bản."""
    return re.findall(r"\d+", text)


def extract_capitalized_entities(text: str) -> list[str]:
    """
    Trích xuất các thực thể viết hoa (Proper Nouns) trong câu:
    - Bỏ qua từ đầu tiên của câu (sau dấu chấm, dấu hỏi, dấu chấm than, hoặc đầu đoạn).
    - Lấy các từ viết hoa chữ cái đầu hoặc viết hoa toàn bộ (như Sếp, Dũng, SJC, Momo, Lan, Hà Nội).
    """
    if not text:
        return []

    # Tách văn bản thành các câu/mệnh đề
    sentences = re.split(r"[.?!;\n]+", text)
    entities: list[str] = []

    for sentence in sentences:
        words = [w.strip() for w in re.split(r"[^\w]+", sentence) if w.strip()]
        if len(words) <= 1:
            continue
        # Bỏ qua từ đầu câu (word 0), xét các từ tiếp theo
        for word in words[1:]:
            if word[0].isupper() and not word.isdigit():
                entities.append(word)

    return entities


@dataclass(slots=True, kw_only=True)
class ASRCorpusMetrics:
    total_samples: int
    normalized_wer: float
    normalized_cer: float
    raw_wer: float
    raw_cer: float
    exact_match_norm_pct: float
    exact_match_raw_pct: float
    number_recall_pct: float
    total_gold_numbers: int
    matched_gold_numbers: int
    capitalized_recall_pct: float
    total_gold_entities: int
    matched_gold_entities: int
    total_audio_duration_sec: float
    total_inference_time_sec: float
    avg_latency_ms: float
    real_time_factor: float


def compute_corpus_metrics(
    gold_texts: Sequence[str],
    pred_raw_texts: Sequence[str],
    audio_durations_sec: Sequence[float],
    inference_times_sec: Sequence[float],
) -> ASRCorpusMetrics:
    """
    Tính toán toàn diện 6 tiêu chuẩn metric định lượng corpus-level.
    """
    n = len(gold_texts)
    if n == 0:
        raise ValueError("Danh sách mẫu đánh giá không được để trống.")

    # 1. Chuẩn bị danh sách chuỗi chuẩn hóa
    gold_norms = [normalize_text_for_acoustic_eval(t) for t in gold_texts]
    pred_norms = [normalize_text_for_acoustic_eval(t) for t in pred_raw_texts]

    # 2. Normalized WER & CER
    # Để tránh lỗi chuỗi rỗng của jiwer
    clean_gold_norms = [g if g else "<empty>" for g in gold_norms]
    clean_pred_norms = [p if p else "<empty>" for p in pred_norms]
    norm_wer = float(jiwer.wer(reference=clean_gold_norms, hypothesis=clean_pred_norms)) * 100.0
    norm_cer = float(jiwer.cer(reference=clean_gold_norms, hypothesis=clean_pred_norms)) * 100.0

    # 3. Raw WER & CER
    clean_gold_raws = [g.strip() if g.strip() else "<empty>" for g in gold_texts]
    clean_pred_raws = [p.strip() if p.strip() else "<empty>" for p in pred_raw_texts]
    raw_wer = float(jiwer.wer(reference=clean_gold_raws, hypothesis=clean_pred_raws)) * 100.0
    raw_cer = float(jiwer.cer(reference=clean_gold_raws, hypothesis=clean_pred_raws)) * 100.0

    # 4. Exact Sentence Match (%)
    exact_norm_count = sum(1 for g, p in zip(gold_norms, pred_norms, strict=True) if g == p)
    exact_raw_count = sum(1 for g, p in zip(clean_gold_raws, clean_pred_raws, strict=True) if g == p)
    exact_norm_pct = (exact_norm_count / n) * 100.0
    exact_raw_pct = (exact_raw_count / n) * 100.0

    # 5. Number / Digit Recall (%)
    total_gold_numbers = 0
    matched_gold_numbers = 0
    for gold_raw, pred_raw in zip(gold_texts, pred_raw_texts, strict=True):
        gold_nums = extract_numbers(gold_raw)
        pred_nums = extract_numbers(pred_raw)
        if not gold_nums:
            continue
        total_gold_numbers += len(gold_nums)
        pred_cnt = Counter(pred_nums)
        for num in gold_nums:
            if pred_cnt[num] > 0:
                matched_gold_numbers += 1
                pred_cnt[num] -= 1

    number_recall = (
        (matched_gold_numbers / total_gold_numbers * 100.0) if total_gold_numbers > 0 else 100.0
    )

    # 6. Capitalized Entity Recall (%)
    total_gold_entities = 0
    matched_gold_entities = 0
    for gold_raw, pred_raw in zip(gold_texts, pred_raw_texts, strict=True):
        gold_ents = extract_capitalized_entities(gold_raw)
        pred_ents = extract_capitalized_entities(pred_raw)
        if not gold_ents:
            continue
        total_gold_entities += len(gold_ents)
        pred_cnt = Counter(pred_ents)
        for ent in gold_ents:
            if pred_cnt[ent] > 0:
                matched_gold_entities += 1
                pred_cnt[ent] -= 1

    cap_recall = (
        (matched_gold_entities / total_gold_entities * 100.0) if total_gold_entities > 0 else 100.0
    )

    # 7. Speed & RTF
    total_audio_sec = float(sum(audio_durations_sec))
    total_infer_sec = float(sum(inference_times_sec))
    avg_latency_ms = (total_infer_sec / n) * 1000.0
    rtf = (total_infer_sec / total_audio_sec) if total_audio_sec > 0 else 0.0

    return ASRCorpusMetrics(
        total_samples=n,
        normalized_wer=norm_wer,
        normalized_cer=norm_cer,
        raw_wer=raw_wer,
        raw_cer=raw_cer,
        exact_match_norm_pct=exact_norm_pct,
        exact_match_raw_pct=exact_raw_pct,
        number_recall_pct=number_recall,
        total_gold_numbers=total_gold_numbers,
        matched_gold_numbers=matched_gold_numbers,
        capitalized_recall_pct=cap_recall,
        total_gold_entities=total_gold_entities,
        matched_gold_entities=matched_gold_entities,
        total_audio_duration_sec=total_audio_sec,
        total_inference_time_sec=total_infer_sec,
        avg_latency_ms=avg_latency_ms,
        real_time_factor=rtf,
    )
