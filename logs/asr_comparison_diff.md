# BÁO CÁO ĐỐI SOÁT PHIÊN ÂM ASR (TOP 30 MẪU CHÊNH LỆCH LỚN NHẤT)

> **Mục đích:** Đánh giá khả năng viết hoa tên riêng (ví dụ: 'Sếp Dũng', 'Đà Nẵng'), chuẩn hóa chữ số/giờ giấc và độ bền âm học giữa `PhoASR-small` và `PhoWhisper-large`.

| STT | Question ID | Chênh lệch Delta | WER PhoASR-small | WER PhoWhisper-large |
|---|---|---|---|---|
| 1 | `train_0002` | 0.09 | 0.43 | 0.35 |
| 2 | `train_0003` | 0.09 | 0.52 | 0.43 |
| 3 | `train_0001` | 0.05 | 0.47 | 0.42 |

---

## CHI TIẾT TỪNG MẪU

### Mẫu 1: `train_0002` (Delta WER: 0.09)
- **Chuẩn BTC (Gold):** `Nếu mai mưa thì hủy lịch đá bóng... À mà thôi, cứ để lịch đó, chỉ cần xem giúp mai có mưa không.`
- **PhoASR-whisper-small (0.24B):** `Nếu mai mưa thì lịch đã bóng à? Mà thôi cứ để lịch đó chị còn xem giúp mai có mưa không?` *(WER: 0.43)*
- **PhoWhisper-large (1.54B):** `nếu may mưa thì huỷ lịch đá bón à mà thôi cứ để lịch đó chỉ cần xem giúp may có mưa không.` *(WER: 0.35)*

### Mẫu 2: `train_0003` (Delta WER: 0.09)
- **Chuẩn BTC (Gold):** `Ơi, nạp giúp mình 50 nghìn điện thoại cho bác Ba với chuyển ví cho cô Mai số tiền điện tháng này nhé.`
- **PhoASR-whisper-small (0.24B):** `Ơi! Nằm cái mình năm mươi ngàn tiền điện thoại cho bác Ba với chuyển ví trong con Mai số tiền điện Khánh Lan nhé!` *(WER: 0.52)*
- **PhoWhisper-large (1.54B):** `này nằm trước mình năm mươi ngàn tiền điện thoại cho bác ba với chuyển ví cho cô mai số tiền điện hắn này nhé.` *(WER: 0.43)*

### Mẫu 3: `train_0001` (Delta WER: 0.05)
- **Chuẩn BTC (Gold):** `Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn Sếp Dũng giờ mới; không mưa thì thôi.`
- **PhoASR-whisper-small (0.24B):** `Nếu mai mưa thì giời chạy võ muộn. Một tiếng nhắn xích dũng. Giờ mới không mưa thì thôi.` *(WER: 0.47)*
- **PhoWhisper-large (1.54B):** `nếu may mưa thì rời chạy bộ muộn một tiếng nhắn sức dũng giờ mới không mưa thì thôi.` *(WER: 0.42)*

