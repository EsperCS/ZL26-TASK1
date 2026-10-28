# BẢNG XẾP HẠNG SLM SOTA 2026 — TASK 1: LIFEMATE (VAL_500)
> **Tiêu chuẩn kiểm thử:** Berkeley Function Calling Leaderboard (BFCL v4) + Zalo ToolEnv State Hash  
> **Thời gian đánh giá:** 2026-10-28 17:59:15  
> **Ngân sách tham số cho phép:** Tổng hệ thống (ASR ~0.433B + SLM) < 6.0B  

| Hạng | Ứng viên SLM | Chế độ Input | Thinking | State Hash (%) | Tool F1 (%) | Schema Adherence (%) | Arg Exact Match (%) | Avg Think Tokens | Latency (ms) | Tổng Params (B) | Đạt Chuẩn Ngân Sách |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Qwen/Qwen2.5-3B-Instruct (Thinking)** | Oracle Mode |  Bật | **0.00%** | 100.00% | 100.00% | 100.00% | 79 | 0.1ms | 3.52B |  PASS |
| **#2** | **Qwen/Qwen2.5-3B-Instruct (Thinking)** | Real ASR Mode |  Bật | **0.00%** | 100.00% | 100.00% | 100.00% | 80 | 0.1ms | 3.52B |  PASS |

### Ghi chú 6 Tiêu chuẩn định lượng:
1. **State Hash Match (%)**: Tiêu chí sống còn của Zalo AI Challenge 2026 (lệch 1 ký tự = 0 điểm).
2. **AST Tool Selection F1 (%)**: Đo lường độ chính xác trong việc lựa chọn đúng tên hàm và số lượng hàm.
3. **Schema Adherence (%)**: Tỷ lệ các tool calls tuân thủ 100% Pydantic Schema (`required`, `x-one-of`, `enum`, `format`).
4. **Argument Exact Match (%)**: Đo lường độ khớp tham số (Exact Match cho non-fuzzy, Token-F1 > 0.6 cho fuzzy).
5. **Avg Think Tokens & Latency**: Chi phí token phát sinh khi bật cơ chế CoT `<think>` và thời gian phản hồi.
6. **Tổng Params (B)**: Tổng tham số ASR (~0.433B) + SLM Total Params bắt buộc phải < 6.0B.