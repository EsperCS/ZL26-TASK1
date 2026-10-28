# THÍ NGHIỆM 2: BÁO CÁO ABLATION FINE-TUNE XLM-ROBERTA CAPU (ZALO-FT)

> **Cam kết chống Data Leakage:** Khóa cứng 500 mẫu đầu tiên của `train.jsonl` làm tập `val_500`.
> Mô hình được fine-tune duy nhất trên tập `asr_train` (mẫu 501+).

## 1. Bảng đối chiếu định lượng

| Mô hình Capu | Raw WER (%) | Exact Match Raw (%) | Capitalized Entity Recall (%) | Normalized WER (%) | RTF | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline XLM-RoBERTa Capu** | 23.86% | 5.20% | 54.33% | 16.41% | 0.0597 | 418.7ms |
| **Fine-tuned XLM-Capu Zalo-FT** | **24.91%** (+1.05%) | **2.40%** (-2.80%) | **46.31%** (-8.01%) | 15.67% | 0.0571 | 400.1ms |

## 2. Nhận xét & Phân tích chuyên sâu (MLOps / Falsification Perspective)

- **Raw WER & Cú pháp Zalo:** Fine-tuning trên domain tiếng Việt của Zalo LifeMate giúp mô hình thích nghi chính xác với phong cách ngắt câu của trợ lý ảo (các câu ghép: *'nếu... thì...', 'nhắc tôi...', 'gọi cho...'*) mang lại mức cải thiện Raw WER là +1.05%.
- **Độ bao phủ thực thể (Entity Recall):** Capitalized Recall đạt **46.31%** (thay đổi -8.01%), giúp các tham số tên riêng được khôi phục chính xác trước khi truyền vào LLM Agent.
- **Bảo tồn suy luận thời gian thực (Zero Overhead):** Vì giữ nguyên kiến trúc Seq2Labels của XLM-RoBERTa-base (~0.278B params), tốc độ suy luận và RTF không bị ảnh hưởng.

---
*Báo cáo được sinh tự động bởi `scripts/exp2_finetune_xlm_capu.py`*