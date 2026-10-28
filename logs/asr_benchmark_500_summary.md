# BÁO CÁO ĐÁNH GIÁ SO SÁNH ASR 4 TẦNG (500 MẪU)
**Dự án:** Zalo AI Challenge 2026 - Task 1 (Lifemate)  
**Thời gian thực hiện:** 2026-10-27 17:29:04  
**Tập dữ liệu:** `train.jsonl` (500 mẫu) | Tổng thời lượng audio: 3504.0s (58.4 phút)  
**Cấu hình phần cứng & Thiết bị:** `CUDA`

---

## 1. Bảng So Sánh Toàn Diện 6 Tiêu Chuẩn Định Lượng

| Tiêu chí đánh giá | Gipformer-68M (~68M params) | ZipFormer-150M (~153M params) | Chênh lệch (Zip - Gip) | Mô hình tốt hơn |
| :--- | :---: | :---: | :---: | :---: |
| **Normalized WER (%)** (Acoustic thuần) | **18.01%** | **16.41%** | -1.60% | ZipFormer |
| **Normalized CER (%)** | **11.56%** | **10.28%** | -1.28% | ZipFormer |
| **Raw WER (%)** (Sau Capu + ITN) | **25.53%** | **23.86%** | -1.67% | ZipFormer |
| **Raw CER (%)** (Sau Capu + ITN) | **13.50%** | **12.22%** | -1.28% | ZipFormer |
| **Exact Sentence Match (Normalized)** | **9.2%** | **10.4%** | +1.2% | ZipFormer |
| **Exact Sentence Match (Raw)** | **4.0%** | **5.2%** | +1.2% | ZipFormer |
| **Number / Digit Recall (%)** | **68.2%** (193/283) | **61.5%** (174/283) | -6.7% | Gipformer |
| **Capitalized Entity Recall (%)** | **54.2%** (338/624) | **54.3%** (339/624) | +0.2% | ZipFormer |
| **Độ trễ trung bình (ms/sample)** | **379.5 ms** | **424.6 ms** | +45.2 ms | Gipformer (Nhanh hơn) |
| **Real-Time Factor (RTF)** | **0.0541** | **0.0606** | +0.0064 | Gipformer |

---

## 2. Phân Tích Kỹ Thuật Chuyên Sâu (Falsify Protocol)

1. **Hiệu năng Âm học (Acoustic Performance):**
   - Gipformer-68M có Normalized WER là `18.01%`.
   - ZipFormer-150M có Normalized WER là `16.41%`.
2. **Vai trò của XLM-RoBERTa Capu & Vietnamese ITN:**
   - Khôi phục chính xác các thực thể viết hoa (Tên riêng như Lan, Dũng, SJC, Ba, Vy...) đạt `54.3%`.
   - Khôi phục số lượng và ngày giờ đạt Number Recall `68.2%`.
3. **Ngân sách tham số & Tốc độ thực thi:**
   - Gipformer Pipeline: ~348M tham số, RTF = `0.0541`.
   - ZipFormer Pipeline: ~433M tham số, RTF = `0.0606`.
   - Cả 2 phương án đều bảo toàn > 5.5B tham số cho LLM và đáp ứng trần < 6B của đề thi.
