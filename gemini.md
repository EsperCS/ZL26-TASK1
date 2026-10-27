# PROJECT RULES - ZALO AI CHALLENGE 2026 (TASK 1: LIFEMATE)

## 1. Tech Stack & Môi trường
- Python 3.12+, quản lý gói bằng `uv`, kiểm tra code bằng `ruff` và `pytest`.
- Tổng tham số toàn bộ mô hình (ASR + LLM) BẮT BUỘC < 6 tỷ (6B). Riêng LLM < 5 tỷ (5B), context <= 16,384 tokens.
- Cố định toàn bộ random seeds (Python, NumPy, PyTorch) và luôn dùng `temperature = 0.0` (greedy decoding) để đảm bảo tính tái lập 100%.

## 2. Luật Cấm Tuyệt Đối (CRITICAL DON'Ts)
1. TUYỆT ĐỐI CẤM đọc trực tiếp file `state.json` từ ổ cứng dưới mọi hình thức (vi phạm sẽ bị loại khỏi cuộc thi). Mọi tương tác dữ liệu phải đi qua `zaic_tools.ToolEnv`.
2. CẤM khởi tạo lại `ToolEnv` nhiều hơn 1 lần cho mỗi `question_id`.
3. CẤM gọi trực tiếp `env.call(name, args)` khi chưa kiểm tra qua `env.validate(name, args)`. Nếu `env.validate()` báo lỗi (`ok == False`), phải trả lỗi về cho Agent sửa, không được đẩy vào `env.call()`.
4. CẤM gọi bất kỳ API internet nào trong luồng inference (container chạy ở chế độ offline `--network none`).
5. CẤM tự ý sửa file chấm điểm `src/eval/evaluator.py` sau khi đã chốt.
