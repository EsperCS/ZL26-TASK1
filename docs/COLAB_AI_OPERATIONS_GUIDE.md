# SỔ TAY VẬN HÀNH COLAB & TÀI NGUYÊN DRIVE (DÀNH CHO AI & KỸ SƯ)
> **Dự án:** Zalo AI Challenge 2026 — Task 1: LifeMate (Trợ lý cuộc sống thông minh qua giọng nói)  
> **Repository GitHub:** `https://github.com/EsperCS/ZL26-TASK1.git`  
> **Thư mục Drive trung tâm:** `Google Drive > ZaloAI_2026_ASR` (Đường dẫn Colab: `/content/drive/MyDrive/ZaloAI_2026_ASR`)  
> **Mục tiêu tài liệu:** Cung cấp hướng dẫn chi tiết, chuẩn xác 100% cho các phiên làm việc tiếp theo của AI Assistant và kỹ sư, tránh lãng phí thời gian huấn luyện lại hoặc nhầm lẫn đường dẫn tài nguyên.

---

## 1. Bản Đồ Tài Nguyên Đã Huấn Luyện Trên Google Drive

Toàn bộ các checkpoints và kết quả nghiên cứu đã được người dùng lưu trữ an toàn tại thư mục **`ZaloAI_2026_ASR`** trên Google Drive (`/content/drive/MyDrive/ZaloAI_2026_ASR/`).

```
📁 Drive của tôi / ZaloAI_2026_ASR/
├── 📁 xlm-roberta-capu-zalo-ft/          <-- Checkpoint khôi phục dấu câu & viết hoa đã fine-tune
│   ├── 📁 vocabulary/                   <-- Từ điển labels & token của mô hình GecBERT
│   │   ├── labels.txt
│   │   └── d_tags.txt
│   ├── model.safetensors                <-- TRỌNG SỐ CHÍNH (~1.0 GB, định dạng safetensors)
│   ├── config.json                      <-- Cấu hình Seq2Labels model
│   ├── configuration_seq2labels.py      <-- File định nghĩa config tương thích Hugging Face
│   ├── gec_model.py                     <-- Interface GecBERTModel
│   ├── modeling_seq2labels.py           <-- Kiến trúc mô hình mạng
│   ├── tokenizer.json                   <-- Tokenizer XLM-RoBERTa
│   ├── tokenizer_config.json
│   ├── utils.py                         <-- Tiện ích tiền xử lý / hậu xử lý token
│   ├── verb-form-vocab.txt
│   └── vocabulary.py
│
└── 📁 logs/                             <-- Toàn bộ báo cáo khoa học & dữ liệu đối chứng
    ├── asr_final_leaderboard.md         <-- Bảng xếp hạng ASR chốt hạ (ZipFormer vô địch)
    ├── ablation_exp2_xlm_ft.md          <-- Báo cáo chi tiết Thí nghiệm 2 (Fine-tune Capu)
    ├── asr_500_error_analysis.md        <-- Phân tích lỗi trên 500 mẫu kiểm định cứng
    ├── asr_benchmark_500_summary.md     <-- Tổng hợp điểm số WER, Match Rate, RTF
    └── asr_comparison_diff.md           <-- File đối chứng câu dự đoán trước vs sau fine-tune
```

> [!IMPORTANT]
> **Đặc điểm nhận diện file trọng số Capu:**
> - Tên file trọng số trong thư mục `xlm-roberta-capu-zalo-ft` là **`model.safetensors`** (không phải `pytorch_model.bin`).
> - Bất kỳ script nào kiểm tra sự tồn tại của mô hình phải kiểm tra cả `model.safetensors` lẫn `pytorch_model.bin`.

---

## 2. Quy Trình Khởi Tạo Chuẩn Trên Google Colab Mới (1-Click Setup)

Mỗi khi mở một phiên Google Colab mới, AI Assistant hoặc người dùng chỉ cần chạy 2 cell code sau:

### Cell 1: Kết nối Drive, Clone Repo & Cài đặt môi trường
```python
# 1. Kết nối Google Drive
from google.colab import drive
import os
from pathlib import Path

if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

# 2. Clone hoặc Cập nhật Repository
REPO_DIR = Path("/content/zaloai-challenge-2026")
REPO_URL = "https://github.com/EsperCS/ZL26-TASK1.git"

if not REPO_DIR.exists():
    !git clone {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
!git pull origin main

# 3. Cài đặt các thư viện lõi phục vụ ASR & LLM
!pip install -q sherpa-onnx soundfile transformers evaluate jiwer accelerate
```

### Cell 2: Khôi phục Checkpoints & Tài nguyên từ Google Drive
```python
# Tự động nạp toàn bộ Capu Fine-Tuned và Hotwords từ Google Drive về máy Colab
!python scripts/backup_to_drive.py --action restore --drive-dir /content/drive/MyDrive/ZaloAI_2026_ASR
```
*Lệnh trên sẽ tự động sao chép `xlm-roberta-capu-zalo-ft` và các báo cáo vào thư mục `models/asr/` và `logs/` trong môi trường làm việc chỉ mất khoảng 5–10 giây.*

---

## 3. Tổng Kết Kiến Trúc ASR Đã Chốt (Phase 1)

AI trong các phiên sau **KHÔNG ĐƯỢC THAY ĐỔI** kiến trúc ASR cốt lõi vì đã được chứng minh qua thực nghiệm chặt chẽ:

| Tầng | Thành phần | Mô hình / Công nghệ | Vai trò kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Tầng 1** | **VAD & Chuẩn hóa** | Silero VAD (ONNX) | Cắt bỏ khoảng lặng, loại bỏ nhiễu, chuẩn hóa biên độ [-0.95, 0.95] |
| **Tầng 2** | **Acoustic Transducer** | `hynt/ZipFormer-150M` (Sherpa-ONNX) | Nhận dạng âm thanh tiếng Việt siêu tốc (RTF ~0.06x), chống nuốt âm |
| **Bổ trợ 2** | **Context-Biasing** | `models/asr/zalo_hotwords.txt` | 418 từ vựng đặc thù LifeMate (Zalo, Mini App, thiết bị, phòng ban, tiền tệ) |
| **Tầng 3** | **Khôi phục ngữ pháp** | `xlm-roberta-capu-zalo-ft` | Phục hồi viết hoa tên riêng, thực thể và dấu câu theo văn phong LifeMate |
| **Tầng 4** | **Số hóa & Đơn vị** | Vietnamese ITN (`VietnameseITN`) | Chuyển đổi số nói ("hai tiếng", "50 nghìn") thành số viết chuẩn cho ToolEnv |

### Kết quả benchmark chính thức trên 500 mẫu `val_500` (Không nhiễm dữ liệu):
- **WER:** **15.67%** (Thấp nhất toàn bộ các phương án thử nghiệm).
- **Entity Recall:** **84.30%** (Vượt trội hoàn toàn so với Whisper Small 78.4%).
- **Tốc độ xử lý:** **RTF 0.0612x** (~400ms trên đoạn audio 6–10 giây).
- **Tổng tham số ASR:** **~0.433B** (Chiếm chưa đầy 8% trần ngân sách cho phép 6.0B).

---

## 4. Công Cụ Đa Năng: Phiên Âm Bất Kỳ File Âm Thanh Nào

Khi người dùng cần nhận dạng âm thanh cho bất kỳ bài giảng, cuộc họp hay file ghi âm cá nhân nào (ví dụ: `mydrive/uit-cnxhkh/on_gk.m4a`):

```bash
python scripts/transcribe_any.py "mydrive/uit-cnxhkh/on_gk.m4a" --device cuda --chunk-duration 30.0
```
- **Hỗ trợ định dạng:** Tự động gọi `ffmpeg` giải mã `.m4a`, `.mp3`, `.aac`, `.flac`, `.wav` sang 16kHz mono.
- **Xử lý file dài:** Tự động chia khối 30 giây, in timestamps `[MM:SS]` thời gian thực.
- **Tự động xuất báo cáo:** Lưu song song cả file `.md` (định dạng đẹp) và `.txt` (văn bản thuần) ngay cạnh file audio gốc trên Google Drive.
- **Ưu tiên nạp model:** Tự động dò và nạp `ZaloAI_2026_ASR/xlm-roberta-capu-zalo-ft` chứa `model.safetensors`.

---

## 5. Luật Cấm Tuyệt Đối Cho AI Ở Phase 2 (LLM Agent & ToolEnv)

Khi tiếp tục thực hiện **Phase 2 (Xây dựng ReAct LLM Agent tích hợp ToolEnv)**, AI phải tuân thủ nghiêm ngặt các điều luật của ban tổ chức Zalo AI Challenge 2026:

1. **Tuyệt đối cấm đọc file `state.json`:** Mọi thông tin trạng thái thiết bị, danh bạ, thời gian phải được truy vấn qua `zaic_tools.ToolEnv`. Đọc trộm file từ ổ cứng sẽ bị loại trực tiếp.
2. **Khởi tạo `ToolEnv` duy nhất 1 lần:** CẤM khởi tạo lại `ToolEnv` nhiều hơn 1 lần cho mỗi `question_id`.
3. **Quy tắc Kiểm duyệt trước khi thực thi:** CẤM gọi trực tiếp `env.call(name, args)` khi chưa kiểm tra qua `env.validate(name, args)`. Nếu `env.validate()` báo lỗi (`ok == False`), Agent phải dùng phản hồi lỗi đó để tự sửa (Self-Correction Loop).
4. **Offline hoàn toàn:** Container chấm điểm chạy với cờ `--network none`. Mọi trọng số mô hình LLM và thư viện phải được chuẩn bị cục bộ, không gọi OpenAI API hay Hugging Face Hub trong lúc suy luận.
5. **Ngân sách tham số:** Tổng tham số ASR + LLM < 6 tỷ (6B). Do ASR đã dùng ~0.433B, LLM phải có tham số **< 5.0 tỷ (5B)**. Ứng viên tối ưu: `Qwen2.5-3B-Instruct` hoặc `Qwen2.5-1.5B-Instruct` / `Gemma-2-2B-it`.
6. **Độ dài Context:** Tối đa 16,384 tokens.
7. **Tính tái lập 100%:** Cố định toàn bộ random seeds (`seed = 42`) và luôn đặt `temperature = 0.0` (greedy decoding).
