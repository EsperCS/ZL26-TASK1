# BÁO CÁO ĐỐI SOÁT PHIÊN ÂM ASR (TOP 30 MẪU CHÊNH LỆCH LỚN NHẤT)

> **Mục đích:** Đánh giá khả năng viết hoa tên riêng (ví dụ: 'Sếp Dũng', 'Đà Nẵng'), chuẩn hóa chữ số/giờ giấc và độ bền âm học giữa `PhoASR-small` và `PhoWhisper-large`.

| STT | Question ID | Chênh lệch Delta | WER PhoASR-small | WER PhoWhisper-large |
|---|---|---|---|---|
| 1 | `train_0195` | 0.67 | 0.33 | 1.00 |
| 2 | `train_0092` | 0.44 | 0.72 | 0.28 |
| 3 | `train_0030` | 0.38 | 0.23 | 0.62 |
| 4 | `train_0103` | 0.36 | 0.79 | 0.43 |
| 5 | `train_0219` | 0.33 | 0.17 | 0.50 |
| 6 | `train_0048` | 0.33 | 0.33 | 0.67 |
| 7 | `train_0225` | 0.33 | 0.00 | 0.33 |
| 8 | `train_0188` | 0.30 | 0.50 | 0.20 |
| 9 | `train_0032` | 0.27 | 0.73 | 0.47 |
| 10 | `train_0012` | 0.25 | 0.30 | 0.55 |
| 11 | `train_0181` | 0.25 | 0.33 | 0.58 |
| 12 | `train_0063` | 0.25 | 0.62 | 0.38 |
| 13 | `train_0068` | 0.25 | 0.17 | 0.42 |
| 14 | `train_0089` | 0.25 | 0.25 | 0.50 |
| 15 | `train_0147` | 0.25 | 0.50 | 0.75 |
| 16 | `train_0045` | 0.25 | 0.10 | 0.35 |
| 17 | `train_0164` | 0.24 | 0.65 | 0.41 |
| 18 | `train_0014` | 0.24 | 0.53 | 0.29 |
| 19 | `train_0191` | 0.24 | 0.35 | 0.59 |
| 20 | `train_0134` | 0.23 | 0.13 | 0.37 |
| 21 | `train_0078` | 0.23 | 0.54 | 0.77 |
| 22 | `train_0035` | 0.22 | 0.78 | 0.56 |
| 23 | `train_0100` | 0.22 | 0.15 | 0.37 |
| 24 | `train_0192` | 0.22 | 0.37 | 0.59 |
| 25 | `train_0220` | 0.22 | 0.44 | 0.22 |
| 26 | `train_0002` | 0.22 | 0.13 | 0.35 |
| 27 | `train_0163` | 0.22 | 0.13 | 0.35 |
| 28 | `train_0144` | 0.22 | 0.30 | 0.52 |
| 29 | `train_0050` | 0.21 | 0.29 | 0.50 |
| 30 | `train_0098` | 0.21 | 0.54 | 0.32 |

---

## CHI TIẾT TỪNG MẪU

### Mẫu 1: `train_0195` (Delta WER: 0.67)
- **Chuẩn BTC (Gold):** `Zalo nhé em.`
- **PhoASR-whisper-small (0.24B):** `Zalo nhé em!` *(WER: 0.33)*
- **PhoWhisper-large (1.54B):** `dạt lộ nha em.` *(WER: 1.00)*

### Mẫu 2: `train_0092` (Delta WER: 0.44)
- **Chuẩn BTC (Gold):** `Đặt giúp em món bún bò giò ở quán bún bò Huế, mỗi người trong nhóm chat Gia đình một suất, không tính em nhé.`
- **PhoASR-whisper-small (0.24B):** `Các dịp y mong muốn vợ giàu ở Quế Bến Bờ Huế mỗi người trong giám sát gia đình một sức không thích yên nhất.` *(WER: 0.72)*
- **PhoWhisper-large (1.54B):** `đặt nếp em món bún bò giò ở quán bún bò huế mỗi người trong nhóm chắc gia đình một sức công tính em nhé.` *(WER: 0.28)*

### Mẫu 3: `train_0030` (Delta WER: 0.38)
- **Chuẩn BTC (Gold):** `Ơi, chuyển giúp tôi một triệu hai cho chủ nhà Phạm Hùng nhé.`
- **PhoASR-whisper-small (0.24B):** `Ấy rèn giúp cháu một triệu hai cho chủ nhà Phạm Hùng nhé.` *(WER: 0.23)*
- **PhoWhisper-large (1.54B):** `ơi chuyện giúp tấu một chuyện hay cho chủ nhà phan hùng nghe.` *(WER: 0.62)*

### Mẫu 4: `train_0103` (Delta WER: 0.36)
- **Chuẩn BTC (Gold):** `Đặt báo thức 6 rưỡi sáng mai, rồi gọi cho Minh giúp tôi nhé.`
- **PhoASR-whisper-small (0.24B):** `Đọc báo Đức sáu giờ là xong, ai lại gọi mình giúp tôi nhé!` *(WER: 0.79)*
- **PhoWhisper-large (1.54B):** `đặt báo trước sáu rưỡi sáng mai rồi gọi minh giúp tôi nhé.` *(WER: 0.43)*

### Mẫu 5: `train_0219` (Delta WER: 0.33)
- **Chuẩn BTC (Gold):** `Nhắc em đi họp với nha.`
- **PhoASR-whisper-small (0.24B):** `Nhắc em đi họp với nha` *(WER: 0.17)*
- **PhoWhisper-large (1.54B):** `nhắc em đi học vậy nha.` *(WER: 0.50)*

### Mẫu 6: `train_0048` (Delta WER: 0.33)
- **Chuẩn BTC (Gold):** `Ờ à, gọi cho Lan nhé.`
- **PhoASR-whisper-small (0.24B):** `À à, gọi cho Lan nhé!` *(WER: 0.33)*
- **PhoWhisper-large (1.54B):** `ờ à à gọi cho lan nhé.` *(WER: 0.67)*

### Mẫu 7: `train_0225` (Delta WER: 0.33)
- **Chuẩn BTC (Gold):** `Nhắn SMS cho Hương là chiều nay gặp ở quán cũ nha.`
- **PhoASR-whisper-small (0.24B):** `Nhắn SMS cho Hương là chiều nay gặp ở quán cũ nha.` *(WER: 0.00)*
- **PhoWhisper-large (1.54B):** `nhắn sms cho hương là chiều nay gặp ở quán cũ nhé.` *(WER: 0.33)*

### Mẫu 8: `train_0188` (Delta WER: 0.30)
- **Chuẩn BTC (Gold):** `45 phút nữa nhắc chị gọi lại cho khách nhé.`
- **PhoASR-whisper-small (0.24B):** `Bốn mươi lăm phút nữa nhé, chị còn lại cho khách nhé.` *(WER: 0.50)*
- **PhoWhisper-large (1.54B):** `bốn lăm phút nữa nhắc chị gọi lại cho khách nhé.` *(WER: 0.20)*

### Mẫu 9: `train_0032` (Delta WER: 0.27)
- **Chuẩn BTC (Gold):** `Chuyển dùm tui cho anh Trần Văn Tuấn số tiền truyền hình tháng này nha.`
- **PhoASR-whisper-small (0.24B):** `Chuyến giùm tôi chợ Anh Trần vang túng xuất hiện chừng hàng tháng này nha.` *(WER: 0.73)*
- **PhoWhisper-large (1.54B):** `chuyển giùm tôi chợ anh trần văn tuấn số tiền truyền hình tháng này nha.` *(WER: 0.47)*

### Mẫu 10: `train_0012` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Bạn ơi, chuyển cho chủ nhà Phạm Hùng hai triệu tám và nạp giúp sếp Dũng hai trăm nghìn nhé.`
- **PhoASR-whisper-small (0.24B):** `Bạn ơi, chị cũng cho rủ nhà Phạm Hùng hai triệu tém và nạp giúp sếp Dũng hai trăm nghề nhé!` *(WER: 0.30)*
- **PhoWhisper-large (1.54B):** `bạn ơ chuyển cho chủ nhà phạm hùng hay triệu tém về nạp giúp sếp dũng hay chăm nghề nhé.` *(WER: 0.55)*

### Mẫu 11: `train_0181` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Xoá giúp chị cái báo thức 7 giờ sáng mai thôi nhé.`
- **PhoASR-whisper-small (0.24B):** `Xõa giúp chị cái bào thuốc bảy giờ sáng mai thôi nhé.` *(WER: 0.33)*
- **PhoWhisper-large (1.54B):** `xoa chút chi tiền vào thức bẩy giờ sáng mai thôi nghe.` *(WER: 0.58)*

### Mẫu 12: `train_0063` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Nhắn tin cho Minh: nhớ mang tài liệu.`
- **PhoASR-whisper-small (0.24B):** `Nhắn tin ra mình, dớm ăn tài liệu` *(WER: 0.62)*
- **PhoWhisper-large (1.54B):** `nhắn tin cho minh giở mang tài liệu.` *(WER: 0.38)*

### Mẫu 13: `train_0068` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Đặt báo thức 6 giờ sáng mai với gọi cho Minh nhé.`
- **PhoASR-whisper-small (0.24B):** `Đặt báo thức sáu giờ sáng mai với gọi cho Minh nhé!` *(WER: 0.17)*
- **PhoWhisper-large (1.54B):** `đặt báo khích sáu giờ sáng nay với gọi cho manh nhé.` *(WER: 0.42)*

### Mẫu 14: `train_0089` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Gọi điện cho Hương dùm tôi cái nha.`
- **PhoASR-whisper-small (0.24B):** `Gọi điện cho Hương giùm tôi cái nha!` *(WER: 0.25)*
- **PhoWhisper-large (1.54B):** `gọi điện cho hương nhìn tôi cái nhé.` *(WER: 0.50)*

### Mẫu 15: `train_0147` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Dạ Zalo đi ạ.`
- **PhoASR-whisper-small (0.24B):** `Dạ zalo đi ạ` *(WER: 0.50)*
- **PhoWhisper-large (1.54B):** `dạ giao lô đi ạ.` *(WER: 0.75)*

### Mẫu 16: `train_0045` (Delta WER: 0.25)
- **Chuẩn BTC (Gold):** `Trợ lý ơi, chuyển giúp tôi cho anh Trần Văn Tuấn số tiền bằng hóa đơn nước tháng này nhé.`
- **PhoASR-whisper-small (0.24B):** `Trợ lý ơi, chuyển giúp tôi ra anh Trần Văn Tuấn số tiền bằng hóa đơn nước tháng này nhé!` *(WER: 0.10)*
- **PhoWhisper-large (1.54B):** `trợ lý ơi chuyển giúp tôi cho anh trần văn tuấn số tiền bằng hoạt động nước tháng này nhé.` *(WER: 0.35)*

### Mẫu 17: `train_0164` (Delta WER: 0.24)
- **Chuẩn BTC (Gold):** `Đặt báo thức 7 giờ sáng mai giúp anh, rồi chuyển ví cho Hương chút tiền nhé.`
- **PhoASR-whisper-small (0.24B):** `Tập nói của bảy giờ sáng mai giúp anh nói chuyện ví cho hương trực hiện nghĩa.` *(WER: 0.65)*
- **PhoWhisper-large (1.54B):** `tập vào khoảng bảy giờ sáng mai giúp anh rồi chuyển ví cho hương chục tiền nhé.` *(WER: 0.41)*

### Mẫu 18: `train_0014` (Delta WER: 0.24)
- **Chuẩn BTC (Gold):** `Tắt hết đèn tầng 2 dùm chị, phòng nào đang bật máy lạnh thì để nguyên nha.`
- **PhoASR-whisper-small (0.24B):** `Tết hết đèn, sành khai giùm chị phòng nào đang bận máy lạnh thì để luôn nha!` *(WER: 0.53)*
- **PhoWhisper-large (1.54B):** `tết hết đèn thằng khai giùm chị phòng nào đang bật máy lạnh thì để nguyên nha.` *(WER: 0.29)*

### Mẫu 19: `train_0191` (Delta WER: 0.24)
- **Chuẩn BTC (Gold):** `Đặt báo thức bảy giờ sáng mai cho anh, rồi chuyển ví cho Lan ít tiền nha.`
- **PhoASR-whisper-small (0.24B):** `Đặt báo xuất bảy giờ sáng mai ra ăn rồi chuyển ví cho Lan tiếp.` *(WER: 0.35)*
- **PhoWhisper-large (1.54B):** `đặt bớt giúp tẩy rửa sắc mai cho anh rồi chuyển ví cho làng thích tiền nhen.` *(WER: 0.59)*

### Mẫu 20: `train_0134` (Delta WER: 0.23)
- **Chuẩn BTC (Gold):** `Đặt bún bò giò ở quán bún bò Huế gần nhà giúp mình, nhóm chat Gia đình bao nhiêu người thì cứ gọi bấy nhiêu phần, không tính mình nha.`
- **PhoASR-whisper-small (0.24B):** `Đặt bún bò giò ở quán bún bò Huế gần nhà giúp mình, giám sát gia đình bao nhiêu người thì cứ gọi bấy nhiêu phần, không tính mình ra.` *(WER: 0.13)*
- **PhoWhisper-large (1.54B):** `đặt bún bò giò ở quán bún bò huế gần nhà giúp mình giám sát gia đình bao nhiêu người chỉ cứ gọi bậy như phần không tính mình ra.` *(WER: 0.37)*

### Mẫu 21: `train_0078` (Delta WER: 0.23)
- **Chuẩn BTC (Gold):** `Tắt hết đèn tầng hai giúp tôi, trừ phòng nào đang chạy quạt thì cứ để đèn đó, rồi bật điều hoà phòng ngủ 25 độ.`
- **PhoASR-whisper-small (0.24B):** `Tắt hết đèn phần hai giúp tôi từ phòng nào đến tiền ngoài thì cứ để đèn đó rồi bọc đìm vào phòng ngủ hai mươi lăm đường.` *(WER: 0.54)*
- **PhoWhisper-large (1.54B):** `tất hết điện thoại đã giúp tôi trừ phần nào điện ngoại thì cứ để điện đó rồi bật điện vào phòng ngủ hay đi làm việc điện là đúng.` *(WER: 0.77)*

### Mẫu 22: `train_0035` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Bạn ơi, tắt đèn phòng khách giúp tôi nhé.`
- **PhoASR-whisper-small (0.24B):** `Ba nữ tóc đen phong cách giúp tôi nhẹ` *(WER: 0.78)*
- **PhoWhisper-large (1.54B):** `bà nữ tắp đèn phòng khách giúp tối nhẹ.` *(WER: 0.56)*

### Mẫu 23: `train_0100` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Tiền mạng tháng này trả bằng ví điện tử giúp chị, ví thiếu thì lấy ngân hàng, xong nhắn SMS cho Hương báo chị đóng rồi nhé.`
- **PhoASR-whisper-small (0.24B):** `Tiền mạng tháng Ni trả bằng ví điện tử giúp chị, ví thiếu thì lấy ngân hàng xong nhắn MS cho Hương báo chị đóng rồi nghe.` *(WER: 0.15)*
- **PhoWhisper-large (1.54B):** `tiền mạng tháng ni làm ra bằng ví điện tử giúp chị ví thiếu thì lấy ngân hàng xong nhắn sms cho hương bảo chị đóng rồi nghe.` *(WER: 0.37)*

### Mẫu 24: `train_0192` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Bật đèn phòng khách lên cho tôi, mà chỉ 30 phần trăm thôi nhé, quạt thì thôi đừng bật, điều hoà phòng khách cứ để 26 độ.`
- **PhoASR-whisper-small (0.24B):** `Bật đèn phòng khách cho tôi mà chỉ 80 phần trăm thôi nhé. Quản thì thôi đùm bật điều phòng khách cứ để 26 độ đi.` *(WER: 0.37)*
- **PhoWhisper-large (1.54B):** `bật đèn phòng khách cho thuê mà chỉ năm mươi phần trăm thuê nhé quán thì thuê đường bật điều vào phòng khách cứ để hai cháu đỡ đi.` *(WER: 0.59)*

### Mẫu 25: `train_0220` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Trợ lý ơi, nhắc mình gọi cho mẹ nhé.`
- **PhoASR-whisper-small (0.24B):** `Trợ lý ơi nhắc mình hỏi cha mẹ nhé!` *(WER: 0.44)*
- **PhoWhisper-large (1.54B):** `trợ lý ơi nhắc mình gọi cho mẹ nhé.` *(WER: 0.22)*

### Mẫu 26: `train_0002` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Nếu mai mưa thì hủy lịch đá bóng... À mà thôi, cứ để lịch đó, chỉ cần xem giúp mai có mưa không.`
- **PhoASR-whisper-small (0.24B):** `Nếu mai mưa thì lịch đá bóng à mà thôi, cứ để lịch đó, chỉ cần xem giúp mai có mưa không.` *(WER: 0.13)*
- **PhoWhisper-large (1.54B):** `nếu may mưa thì huỷ lịch đá bón à mà thôi cứ để lịch đó chỉ cần xem giúp may có mưa không.` *(WER: 0.35)*

### Mẫu 27: `train_0163` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Chuyển cho Lê Minh đúng bằng tiền mạng tháng này nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm.`
- **PhoASR-whisper-small (0.24B):** `Chuyển cho Lê Minh đúng bằng chữ ngạn hát này nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm.` *(WER: 0.13)*
- **PhoWhisper-large (1.54B):** `chuyển cho lê minh đúng bằng tiền đánh á này nha nếu tài khoản an toàn không đủ thì lấy từ tiết kiệm.` *(WER: 0.35)*

### Mẫu 28: `train_0144` (Delta WER: 0.22)
- **Chuẩn BTC (Gold):** `Bật đèn phòng khách mà để 40 phần trăm thôi nha, quạt thì đừng bật, với mở điều hoà phòng khách 26 độ.`
- **PhoASR-whisper-small (0.24B):** `Bật đèn phòng khách mà để phần trăm thôi nha. Quẹt thì thằng Bật bé mở điều phòng khách 26 độ.` *(WER: 0.30)*
- **PhoWhisper-large (1.54B):** `bật đèn phòng khách mà để bốn mươi phần trăm thời gian quạt sẽ dừng bật và mở điều hòa phòng khách hai sáu độ.` *(WER: 0.52)*

### Mẫu 29: `train_0050` (Delta WER: 0.21)
- **Chuẩn BTC (Gold):** `Alô, cho em Linh hỏi giá vàng SJC hôm nay bao nhiêu dùm nha.`
- **PhoASR-whisper-small (0.24B):** `A lô cho em lên hỏi giá vàng SJC hôm nay bao nhiêu dùm nhau?` *(WER: 0.29)*
- **PhoWhisper-large (1.54B):** `a lô cho em linh hỏi giá vàng ép chi xin hôm nay bao nhiêu dùm nhau.` *(WER: 0.50)*

### Mẫu 30: `train_0098` (Delta WER: 0.21)
- **Chuẩn BTC (Gold):** `Trả tiền mạng bằng ví điện tử nhé, ví mà không đủ thì chuyển qua ngân hàng, rồi nhắn SMS cho sếp Dũng báo tôi đóng xong rồi.`
- **PhoASR-whisper-small (0.24B):** `Trả tiền mệnh bằng ví điện tử nhé. Ví rồi không đủ điều chỉnh qua ngân hàng. Rồi nhấn cái xe nhét cho xét nghiệm. Má tôi đóng xong rồi.` *(WER: 0.54)*
- **PhoWhisper-large (1.54B):** `trả tiền mệnh bằng ký điện tử thép ví mà không đủ để chuyển qua ngân hàng rồi nhắn thép cho sếp nhận váy tôi đóng xong rồi.` *(WER: 0.32)*

