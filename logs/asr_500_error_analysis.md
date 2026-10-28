# PHÂN TÍCH 50 MẪU SAI SỐ LỚN NHẤT / LỆCH NHAU NHẤT (ERROR ANALYSIS)
**Tổng số mẫu khảo sát:** 500 | **Số mẫu được phân tích chi tiết:** 50

Mỗi mẫu hiển thị 4 dòng định dạng chuẩn:
- **Gold**: Văn bản đối chiếu gốc
- **Raw Acoustic**: Đầu ra âm học chữ thường (Gipformer vs ZipFormer)
- **Gipformer Final**: Văn bản hoàn chỉnh sau Capu + ITN của Gipformer-68M
- **ZipFormer Final**: Văn bản hoàn chỉnh sau Capu + ITN của ZipFormer-150M

---

### Mẫu 1 [ID: train_0047] - Gip WER: 83.3% | Zip WER: 72.2%
- **Gold**: `Tắt đèn phòng ngủ đi ơi... ờ à mà thôi, không phải, tắt đèn bếp giúp chị nhé.`
- **Raw Acoustic**: Gip: `lên phương ngộ đi ơi trời ơi không xài đúng không tên bẹp chiên nha` | Zip: `đêm phương ngủ đi ơi thôi không xài quá tên bẹp chuyền ạ`
- **Gipformer Final**: `Lên Phương Ngộ đi ơi. Trời ơi. Không xài đúng không? Tên Bẹp Chiên nha.`
- **ZipFormer Final**: `Đêm Phương ngủ đi ơi, thôi không xài quá tên bẹp chuyền ạ.`

### Mẫu 2 [ID: train_0009] - Gip WER: 75.0% | Zip WER: 70.0%
- **Gold**: `Nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha.`
- **Raw Acoustic**: Gip: `đặt giùm em linh năm mươi ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha` | Zip: `đặt giùm em lên năm mươi ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha`
- **Gipformer Final**: `Đặt giùm em Linh năm mươi ngàn cho số 0 9 5 0 5 3 6 0 3 6 7 nha.`
- **ZipFormer Final**: `Đặt giùm em lên 50 ngàn cho số 0 9 5 0 5 3 6 0 3 6 7 nha.`

### Mẫu 3 [ID: train_0483] - Gip WER: 72.7% | Zip WER: 18.2%
- **Gold**: `Vậy cho nó dọn chỗ đó luôn bây giờ đi em.`
- **Raw Acoustic**: Gip: `vậy cháu lựa chọn chủ đổi ý muốn bây giờ đi` | Zip: `vậy cho nó dọn chỗ đói luôn bây giờ đi`
- **Gipformer Final**: `Vậy cháu lựa chọn chủ đổi ý muốn bây giờ đi.`
- **ZipFormer Final**: `Vậy cho nó dọn chỗ đói luôn bây giờ đi.`

### Mẫu 4 [ID: train_0202] - Gip WER: 68.4% | Zip WER: 52.6%
- **Gold**: `Tắt hết đèn tầng hai đi, phòng nào đang bật điều hoà thì cứ để đèn nguyên đó nha.`
- **Raw Acoustic**: Gip: `đến với nhà ta hai đi phần nào năm bật điều hòa thì cứ để đàng hoàng nha` | Zip: `tới nhà tầng hai đi phòng nào năm bạn đi hoa thì cứ để đèn với nó nha`
- **Gipformer Final**: `Đến với nhà ta 2D phần nào, 5Bật điều hòa thì cứ để đàng hoàng nha.`
- **ZipFormer Final**: `Tới nhà tầng 2 đi, phòng nào 5 bạn đi hoa thì cứ để đèn với nó nha.`

### Mẫu 5 [ID: train_0180] - Gip WER: 64.7% | Zip WER: 64.7%
- **Gold**: `Tắt đèn phòng làm việc đi, với nhắn SMS cho Lan là mai gửi mình file nhé.`
- **Raw Acoustic**: Gip: `chắc nam phòng làm việc đi với những em bé cho lăng và may hiểu ý mình phải nhé` | Zip: `tắt nằm phòng làm việc đi với những em bé cho làng và may hiểu im mày phải nhé`
- **Gipformer Final**: `Chắc Nam Phòng làm việc đi với những em bé cho Lăng và May hiểu ý mình phải nhé.`
- **ZipFormer Final**: `Tắt nằm phòng làm việc đi với những em bé cho Làng và May hiểu im mày phải nhé.`

### Mẫu 6 [ID: train_0490] - Gip WER: 57.1% | Zip WER: 42.9%
- **Gold**: `Mở Mưa trên phố nhỏ đi em.`
- **Raw Acoustic**: Gip: `mở mưa cho em phố nhào đi anh` | Zip: `mở mưa cho em phố nhào đi em`
- **Gipformer Final**: `Mở mưa cho em phố nhào đi anh.`
- **ZipFormer Final**: `Mở mưa cho em phố nhào đi em.`

### Mẫu 7 [ID: train_0199] - Gip WER: 56.0% | Zip WER: 52.0%
- **Gold**: `Đèn phòng khách bật dùm tôi 30 phần trăm là đủ, quạt khỏi bật hen, rồi mở điều hoà ngoài phòng khách 26 độ luôn.`
- **Raw Acoustic**: Gip: `đèn phòng khách mà chỉ thuê ba mươi phần trăm là được khoản khoi và then rồi mời điều hòa phòng khách hay sái đồ luôn` | Zip: `đèn phòng khách mà chỉ dùng thuê ba mươi phần trăm là đường quả khoai và then rồi mở điều hoà phòng khách hai sái đồ luôn`
- **Gipformer Final**: `Đèn phòng khách mà chỉ thuê 30 phần trăm là được khoản khoi và then rồi. Mời điều hòa phòng khách hay sái đồ luôn.`
- **ZipFormer Final**: `Đèn phòng khách mà chỉ dùng thuê 30 phần trăm là đường, quả khoai và then rồi mở điều hoà phòng khách hai sái đồ luôn.`

### Mẫu 8 [ID: train_0166] - Gip WER: 35.3% | Zip WER: 52.9%
- **Gold**: `Gọi cho Lan đi ạ, với lại đặt báo thức sáu giờ sáng mai cho em nữa.`
- **Raw Acoustic**: Gip: `gọi cho lan thì với quá thức sáu giờ sáng mai cho em nữa` | Zip: `cho nên quá thức sáu giờ sáng mai cho em nữa`
- **Gipformer Final**: `Gọi cho Lan thì với quá thức 6 giờ sáng mai cho em nữa.`
- **ZipFormer Final**: `Cho nên quá thức, 6 giờ sáng mai cho em nữa.`

### Mẫu 9 [ID: train_0232] - Gip WER: 52.6% | Zip WER: 31.6%
- **Gold**: `Nếu vàng SJC mua vào dưới 123 triệu thì nhắc tôi 9 giờ sáng mai đi mua vàng nhé.`
- **Raw Acoustic**: Gip: `nếu vàng mua vào dưới một trăm hai mươi ba triệu thì nếu tôi tính ra sáng nay đi mua vàng nhé` | Zip: `nếu vàng mua vào dưới một trăm hai ba triệu thì nhất tôi chín giờ sáng mai đi mua vàng nhé`
- **Gipformer Final**: `Nếu vàng mua vào dưới một trăm hai mươi ba triệu thì nếu tôi tính ra sáng nay đi mua vàng nhé.`
- **ZipFormer Final**: `Nếu vàng mua vào dưới một trăm hai ba triệu thì nhất tôi 9 giờ sáng mai đi mua vàng nhé.`

### Mẫu 10 [ID: train_0329] - Gip WER: 52.4% | Zip WER: 42.9%
- **Gold**: `Mở đèn phòng khách 50 phần trăm thôi nha, quạt đừng bật, còn điều hoà phòng khách thì để 26 độ.`
- **Raw Acoustic**: Gip: `mà ở đây một phòng khách năm mươi phần trăm thôi nha các bạn còn điều hòa phòng khách thì là hai sáu độ` | Zip: `mà đây là phòng khách năm mươi phần trăm thôi nha quạng động vật còn điều hoà phòng khách thì rẻ hay sáu độ`
- **Gipformer Final**: `Mà ở đây một phòng khách 50 phần trăm thôi nha các bạn, còn điều hòa phòng khách thì là hai sáu độ.`
- **ZipFormer Final**: `Mà đây là phòng khách, 50 phần trăm thôi nha quạng động vật. Còn điều hoà phòng khách thì rẻ hay 6 độ.`

### Mẫu 11 [ID: train_0147] - Gip WER: 50.0% | Zip WER: 0.0%
- **Gold**: `Dạ Zalo đi ạ.`
- **Raw Acoustic**: Gip: `dạ a lô đi ạ` | Zip: `dạ zalo đi ạ`
- **Gipformer Final**: `Dạ a lô đi ạ.`
- **ZipFormer Final**: `Dạ Zalo đi ạ`

### Mẫu 12 [ID: train_0078] - Gip WER: 50.0% | Zip WER: 50.0%
- **Gold**: `Tắt hết đèn tầng hai giúp tôi, trừ phòng nào đang chạy quạt thì cứ để đèn đó, rồi bật điều hoà phòng ngủ 25 độ.`
- **Raw Acoustic**: Gip: `tắt hết đường phần hai giúp con kỳ phòng nào đang chạy thì cứ để đi đó rồi bật điều hòa vào buổi hai mươi lăm` | Zip: `tắt hết đường tận nơi giúp tôi tìm phòng nào đang chạy ngoài thì cứ để đi vào đó rồi bỏ tiền vào phòng ngủ hai mươi năm đến`
- **Gipformer Final**: `Tắt hết đường phần 2 giúp con. Kỳ phòng nào đang chạy thì cứ để đi đó rồi bật điều hòa vào buổi hai mươi lăm.`
- **ZipFormer Final**: `Tắt hết đường tận nơi, giúp tôi tìm phòng nào đang chạy ngoài thì cứ để đi vào đó rồi bỏ tiền vào phòng ngủ. 20 năm đến.`

### Mẫu 13 [ID: train_0273] - Gip WER: 50.0% | Zip WER: 50.0%
- **Gold**: `Gọi cho Hương giúp tôi cái.`
- **Raw Acoustic**: Gip: `gọi hương hộ cái` | Zip: `gọi hương hộ cái`
- **Gipformer Final**: `Gọi Hương hộ cái.`
- **ZipFormer Final**: `Gọi Hương hộ cái.`

### Mẫu 14 [ID: train_0505] - Gip WER: 48.4% | Zip WER: 25.8%
- **Gold**: `Ờ năm mét vải thì được bao nhiêu xăng ti mét ấy nhỉ, đổi thử giúp em, rồi thêm vào danh sách mua một tấm vải dài đúng chừng đấy nhé.`
- **Raw Acoustic**: Gip: `năm mét vạ thì đúng bao nhiêu centimet đấy nhỉ giúp em rồi thêm vào danh sách mua một chấm bảy dài đúng rồi nhé` | Zip: `năm mét vải thì được bao nhiêu cm đấy nhỉ đưa thử giúp em rồi thêm vào danh sách mua một tấm vải dài đúng chừng ấy nhé`
- **Gipformer Final**: `5 mét vạ thì đúng bao nhiêu centimet đấy nhỉ? Giúp em rồi thêm vào danh sách mua 1 chấm 7 dài đúng rồi nhé.`
- **ZipFormer Final**: `5 mét vải thì được bao nhiêu cm đấy nhỉ? Đưa thử giúp em rồi thêm vào danh sách mua một tấm vải dài đúng chừng ấy nhé.`

### Mẫu 15 [ID: train_0192] - Gip WER: 48.1% | Zip WER: 44.4%
- **Gold**: `Bật đèn phòng khách lên cho tôi, mà chỉ 30 phần trăm thôi nhé, quạt thì thôi đừng bật, điều hoà phòng khách cứ để 26 độ.`
- **Raw Acoustic**: Gip: `bật đèn phòng khách cho thuê mà chỉ tăng mươi phần trăm thôi nhé quản trị thuê động vật điều hòa phòng khách cứ để hai sáu độ đi` | Zip: `bật đèn phòng khách cho tôi mà chỉ tăng nguyên phần trăm thôi nhé quản thịt nuôi động vật điều hoà phòng khách cứ để hai cháu đỡ đi`
- **Gipformer Final**: `Bật đèn phòng khách cho thuê mà chỉ tăng mươi phần trăm thôi nhé. Quản trị thuê động vật điều hòa phòng khách cứ để hai sáu độ đi.`
- **ZipFormer Final**: `Bật đèn phòng khách cho tôi mà chỉ tăng nguyên phần trăm thôi nhé. Quản thịt nuôi động vật, điều hoà phòng khách cứ để hai cháu đỡ đi.`

### Mẫu 16 [ID: train_0502] - Gip WER: 47.1% | Zip WER: 47.1%
- **Gold**: `Gọi giúp tôi tổng đài hỗ trợ, số một chín không không, một hai ba bốn ấy.`
- **Raw Acoustic**: Gip: `gọi giúp tôi tổng đài hỗ trợ số một chín không không một hai ba bốn ấy` | Zip: `gọi giúp tôi tổng đài hỗ trợ số một chín không không một hai ba bốn ấy`
- **Gipformer Final**: `Gọi giúp tôi tổng đài hỗ trợ số 1 9 0 0, 1 2 3 4 ấy.`
- **ZipFormer Final**: `Gọi giúp tôi tổng đài hỗ trợ số 1 9 0 0, 1 2 3 4 ấy.`

### Mẫu 17 [ID: train_0085] - Gip WER: 45.7% | Zip WER: 42.9%
- **Gold**: `Chiều nay em có hẹn đi chơi công viên, đặt xe bốn chỗ từ công ty đến đó, đón sớm bốn mươi lăm phút, nhắc em mang laptop trước lúc xe tới mười lăm phút.`
- **Raw Acoustic**: Gip: `chiều nay anh có hẹn đi chơi công kinh đặt xe hỗ trợ từ công ty đến đó đón sớm bốn mươi lăm phút nếu ngày tới mười lăm phút` | Zip: `cho nên em có hẹn đi chơi công ty đạp xe bổ trợ từ công ty đến đó đón sớm bốn mươi năm phút nhóm em lên laptop chọn xe tới mười lăm phút`
- **Gipformer Final**: `Chiều nay anh có hẹn đi chơi, Công Kinh đặt xe hỗ trợ từ công ty đến đó, đón sớm 45 phút, nếu ngày tới 15 phút.`
- **ZipFormer Final**: `Cho nên em có hẹn đi chơi công ty, đạp xe bổ trợ từ công ty đến đó đón sớm 45 phút, nhóm em lên laptop chọn xe tới 15 phút.`

### Mẫu 18 [ID: train_0402] - Gip WER: 36.4% | Zip WER: 45.5%
- **Gold**: `Mai mà mưa thì cho lịch đá bóng sáng mai lùi hai tiếng, rồi nhắn SMS cho Tuấn báo giờ mới nhé.`
- **Raw Acoustic**: Gip: `mai mà mưa thì tra địch bóng đá sáng mai lùi hai tiếng rồi nhấn sms ra tuấn báo giờ mới nha` | Zip: `mai nào mưa thì ra đợt bóng đá sáng mai lùi hai tiếng rồi nhấn sms ra trấn báo giờ mới nha`
- **Gipformer Final**: `Mai mà mưa thì tra địch bóng đá sáng mai lùi 2 tiếng rồi nhấn sms ra Tuấn báo giờ mới nha`
- **ZipFormer Final**: `Mai nào mưa thì ra đợt bóng đá sáng mai, lùi 2 tiếng rồi nhấn SMS ra trấn báo giờ mới nha.`

### Mẫu 19 [ID: train_0193] - Gip WER: 25.8% | Zip WER: 45.2%
- **Gold**: `Đặt ô tô 4 chỗ từ công ty đến chỗ họp team chiều nay, đón trước giờ hẹn 45 phút, với nhắc tôi mang laptop trước giờ xe đón 15 phút.`
- **Raw Acoustic**: Gip: `đặt ô tô bốn chỗ từ công ty đến chỗ hộp tìm chiều nay đóng trước hạn giờ hạn bốn lăm phút mình nhắc tôi mang laptop trước giờ sẽ đón mười lăm phút` | Zip: `đã có tôi bốn chỗ từ công ty đến chỗ hộp team chiều nay đóng trước hạn giờ hạn món năm phút mình nhắc tôi mang laptop trước giờ sẽ đón mười năm phút`
- **Gipformer Final**: `Đặt ô tô bốn chỗ từ công ty đến chỗ hộp tìm chiều nay đóng trước hạn giờ hạn 45 phút. Mình nhắc tôi mang laptop trước giờ sẽ đón 15 phút.`
- **ZipFormer Final**: `Đã có tôi bốn chỗ từ công ty đến chỗ hộp team chiều nay đóng trước hạn giờ, hạn món 5 phút. Mình nhắc tôi mang laptop trước giờ sẽ đón mười năm phút.`

### Mẫu 20 [ID: train_0482] - Gip WER: 45.0% | Zip WER: 30.0%
- **Gold**: `Ừ, em xem lại lịch hôm đó cho chắc rồi mở phòng họp online đúng giờ trên lịch luôn nhé.`
- **Raw Acoustic**: Gip: `ừ em xin ảnh lịch hôm đó cho chắc làm ở phòng học on lai đúng giờ cho mình luôn nhé` | Zip: `ừ em xin lại cái hôm đó cho chắc mở phòng học online đúng giờ sinh nhật luôn nhé`
- **Gipformer Final**: `Ừ em xin ảnh lịch hôm đó cho chắc làm ở phòng học on lai đúng giờ cho mình luôn nhé.`
- **ZipFormer Final**: `Ừ, em xin lại cái hôm đó cho chắc. Mở phòng học online đúng giờ sinh nhật luôn nhé.`

### Mẫu 21 [ID: train_0035] - Gip WER: 44.4% | Zip WER: 11.1%
- **Gold**: `Bạn ơi, tắt đèn phòng khách giúp tôi nhé.`
- **Raw Acoustic**: Gip: `bà nươi tắt đèn phong khách giúp tôi nhẹ` | Zip: `bạn ơi thắp đèn phòng khách giúp tôi nhé`
- **Gipformer Final**: `Bà Nươi tắt đèn, phong khách giúp tôi nhẹ.`
- **ZipFormer Final**: `Bạn ơi, thắp đèn phòng khách giúp tôi nhé.`

### Mẫu 22 [ID: train_0135] - Gip WER: 44.4% | Zip WER: 11.1%
- **Gold**: `Mai bạn đặt báo thức gọi mình dậy nhé.`
- **Raw Acoustic**: Gip: `mai văn đạt bảo thức gọi mình dây nhé` | Zip: `mai bạn đặt bảo thức gọi mình dậy nhé`
- **Gipformer Final**: `Mai Văn Đạt Bảo Thức gọi mình dây nhé`
- **ZipFormer Final**: `Mai bạn đặt bảo thức gọi mình dậy nhé.`

### Mẫu 23 [ID: train_0528] - Gip WER: 44.4% | Zip WER: 11.1%
- **Gold**: `Ừ, thế huỷ luôn cái gói đó cho tớ.`
- **Raw Acoustic**: Gip: `ừ tớ hủy đăng ký gói đó cho tớ` | Zip: `ừ thế huỷ đưa cái gói đó cho tớ`
- **Gipformer Final**: `Ừ, tớ hủy đăng ký gói đó cho tớ.`
- **ZipFormer Final**: `Ừ, thế Huỷ đưa cái gói đó cho tớ.`

### Mẫu 24 [ID: train_0280] - Gip WER: 33.3% | Zip WER: 44.4%
- **Gold**: `Tắt đèn phòng ngủ giúp tôi, với nhắn SMS cho Minh là tối nay ăn cơm nhà nhé.`
- **Raw Acoustic**: Gip: `tắt điện phòng ngủ giúp tôi với nhánh trong minh là tối nay ăn cơm nhà gì hết` | Zip: `đắt tiền phục vụ giúp tôi với nhánh săn đế cho minh là tối nay ăn cơm nhà dế`
- **Gipformer Final**: `Tắt điện phòng ngủ giúp tôi với, nhánh trong minh là tối nay ăn cơm nhà gì hết.`
- **ZipFormer Final**: `đắt tiền phục vụ giúp tôi với nhánh săn đế cho minh là tối nay ăn cơm nhà dế.`

### Mẫu 25 [ID: train_0472] - Gip WER: 36.7% | Zip WER: 43.3%
- **Gold**: `Lịch rà soát quý dời sang mười giờ sáng thứ ba, hôm hai chín ấy, giúp em nhé. Xong mở luôn một phòng họp online cùng tên, đúng giờ mới.`
- **Raw Acoustic**: Gip: `lịch rà soát quỹ trời sáng mười giờ sáng sẽ ba hôm hai chị mấy chúc em nhé xong rồi luôn một phòng họp online không tên giờ mới` | Zip: `lịch ra soát quỹ trời sáng mười giờ sáng thứ ba hôm hai chị mấy trước em nhé xong mời luôn một phòng họp online không tên huyện giờ mới`
- **Gipformer Final**: `Lịch rà soát quỹ trời sáng 10 giờ sáng sẽ ba hôm hai chị mấy chúc em nhé. Xong rồi luôn một phòng họp online không tên giờ mới.`
- **ZipFormer Final**: `Lịch ra soát quỹ trời sáng 10 giờ sáng thứ 3 hôm 2 chị mấy trước em nhé. Xong mời luôn một phòng họp online không tên huyện giờ mới.`

### Mẫu 26 [ID: train_0526] - Gip WER: 14.3% | Zip WER: 42.9%
- **Gold**: `Ừ, thế khoá cổng lại giúp anh.`
- **Raw Acoustic**: Gip: `ừ thế khóa cổng lại giúp anh` | Zip: `cái khoá cổng này giúp anh`
- **Gipformer Final**: `Ừ, thế khóa cổng lại giúp anh.`
- **ZipFormer Final**: `Cái khoá cổng này giúp anh`

### Mẫu 27 [ID: train_0229] - Gip WER: 42.9% | Zip WER: 17.9%
- **Gold**: `Chuyển cho Trần Văn Tuấn đúng bằng tiền mạng tháng này giúp tôi, trừ vào tài khoản thanh toán, nếu không đủ thì lấy từ tiết kiệm nhé.`
- **Raw Acoustic**: Gip: `chuyển cho trần văn tuấn đúng không tiến ạ hôm nay giúp tôi chưa tài khoản anh bán nhớ không đủ thì lấy từ tiếp điện nhé` | Zip: `chuyển cho trần văn tuấn đúng bằng tiền ạ em ơi giúp tôi trừ vào tài khoản anh bán nếu không đủ thì lấy từ tiết kiệm nhé`
- **Gipformer Final**: `Chuyển cho Trần Văn Tuấn đúng không Tiến ạ? Hôm nay giúp tôi chưa? Tài khoản anh bán nhớ không đủ thì lấy từ tiếp điện nhé.`
- **ZipFormer Final**: `Chuyển cho Trần Văn Tuấn đúng bằng tiền ạ. Em ơi, giúp tôi trừ vào tài khoản anh bán, nếu không đủ thì lấy từ tiết kiệm nhé.`

### Mẫu 28 [ID: train_0393] - Gip WER: 21.4% | Zip WER: 42.9%
- **Gold**: `Đặt báo thức 6 giờ sáng mai, rồi gọi cho Lan giúp mình nha.`
- **Raw Acoustic**: Gip: `nhà báo thức sáu giờ sáng mai rồi gọi ra giúp mình nha` | Zip: `bà thích sáu giờ sáng mai rồi gọi chala giúp mày nha`
- **Gipformer Final**: `Nhà báo thức 6 giờ sáng mai rồi, gọi ra giúp mình nha.`
- **ZipFormer Final**: `Bà Thích: 6 giờ sáng mai rồi, gọi Chala giúp mày nha.`

### Mẫu 29 [ID: train_0228] - Gip WER: 42.9% | Zip WER: 42.9%
- **Gold**: `Đặt báo thức sáu rưỡi sáng mai, rồi gọi cho Lan dùm tôi nha.`
- **Raw Acoustic**: Gip: `đặt vào thức sáu rưỡi sáng mai vừa gọi cho loanh thuê nha` | Zip: `đặt bảo thức sáu rưỡi sáng mai phải gọi cho lan dù thuê nhà`
- **Gipformer Final**: `Đặt vào thức 6 rưỡi sáng mai vừa gọi cho loanh thuê nha.`
- **ZipFormer Final**: `Đặt bảo thức 6 rưỡi sáng mai phải gọi cho Lan dù thuê nhà.`

### Mẫu 30 [ID: train_0108] - Gip WER: 30.8% | Zip WER: 42.3%
- **Gold**: `Tắt hết đèn tầng hai đi, trừ phòng nào đang bật quạt thì cứ để đèn đó, rồi bật điều hoà phòng ngủ 26 độ nha.`
- **Raw Acoustic**: Gip: `hết đèn cái tầng hai đi trừ phòng môn đang mệt quát thì cứ để đèn đó rồi bật điều hòa phòng ngủ hai mươi sáu độ nghe` | Zip: `hết đèn cái tầng hai đi trừ phong mô đang mệt quạt thì cứ để đèn đó rồi bật điều hoà phòng ngủ hai mươi sáu đô nghe`
- **Gipformer Final**: `Hết đèn cái tầng 2 đi, trừ phòng Môn đang mệt quát thì cứ để đèn đó rồi bật điều hòa phòng ngủ 26 độ nghe.`
- **ZipFormer Final**: `Hết đèn cái tầng 2 đi, trừ Phong Mô đang mệt quạt thì cứ để đèn đó rồi bật điều hoà phòng ngủ. Hai mươi sáu đô nghe.`

### Mẫu 31 [ID: train_0246] - Gip WER: 41.7% | Zip WER: 8.3%
- **Gold**: `Tắt hết đèn tầng 2, trừ phòng nào quạt đang chạy thì để nguyên đèn nhé, xong bật điều hoà phòng ngủ 26 độ.`
- **Raw Acoustic**: Gip: `thắt áp đèn thành hai trừ phần nào quậy đang chạy thì để nguyên đèn nhé bật điều hòa phần bố hai mươi sáu độ` | Zip: `tắt hết đèn tầng hai trừ phòng nào quạt đang chạy thì để nguyên đèn nhé xong bật điều hoà phòng ngủ hai sáu độ`
- **Gipformer Final**: `Thắt áp đèn thành hai, trừ phần nào quậy đang chạy thì để nguyên đèn nhé. Bật điều hòa phần bố 26 độ.`
- **ZipFormer Final**: `Tắt hết đèn tầng 2, trừ phòng nào quạt đang chạy thì để nguyên đèn nhé. Xong bật điều hoà phòng ngủ 2 6 độ.`

### Mẫu 32 [ID: train_0083] - Gip WER: 41.7% | Zip WER: 20.8%
- **Gold**: `Bật đèn phòng khách giúp mình, mà dịu thôi, tầm 30 phần trăm, quạt thì khỏi bật, điều hoà phòng khách để 26 độ.`
- **Raw Acoustic**: Gip: `bật đèn phong cách giúp mình mà dịu thôi tầm năm mươi phần trăm quẹt thì khỏi bật điều hòa phong cách để hai mươi sáu` | Zip: `bật đèn phòng khách giúp mình mà dịu thôi tầm ba mươi phần trăm quẹt thì khỏi bật điều hoà phòng cách giữa hai`
- **Gipformer Final**: `Bật đèn phong cách giúp mình mà dịu thôi, tầm 50 phần trăm quẹt thì khỏi bật điều hòa phong cách để hai mươi sáu.`
- **ZipFormer Final**: `Bật đèn phòng khách giúp mình mà dịu thôi, tầm 30 phần trăm quẹt thì khỏi bật điều hoà phòng cách giữa hai.`

### Mẫu 33 [ID: train_0068] - Gip WER: 41.7% | Zip WER: 33.3%
- **Gold**: `Đặt báo thức 6 giờ sáng mai với gọi cho Minh nhé.`
- **Raw Acoustic**: Gip: `bắt hết sáu giờ sáng nay với gọi cho mình nhé` | Zip: `bắt hết sáu giờ sáng nay với gọi cho minh nhé`
- **Gipformer Final**: `Bắt hết 6 giờ sáng nay với. Gọi cho mình nhé.`
- **ZipFormer Final**: `Bắt hết 6 giờ sáng nay với gọi cho minh nhé.`

### Mẫu 34 [ID: train_0285] - Gip WER: 41.2% | Zip WER: 35.3%
- **Gold**: `Tắt đèn bếp giúp tôi, rồi nhắn tin cho Lan là tối nay ăn cơm nhà nhé.`
- **Raw Acoustic**: Gip: `bác tìm góp giúp tôi rồi nhắn tin cho đây là tôi đang ăn cơm nhà đi` | Zip: `cắt tiền bếp giúp tôi rồi nhắn tin cho đây là tôi đang ăn cơm nhà dễ`
- **Gipformer Final**: `Bác tìm góp giúp tôi rồi nhắn tin cho đây là tôi đang ăn cơm nhà: D`
- **ZipFormer Final**: `Cắt tiền bếp giúp tôi rồi nhắn tin cho đây là tôi đang ăn cơm nhà. Dễ.`

### Mẫu 35 [ID: train_0037] - Gip WER: 20.0% | Zip WER: 40.0%
- **Gold**: `Nếu mai mưa thì huỷ giúp tôi lịch đi chơi công viên sáng mai nhé.`
- **Raw Acoustic**: Gip: `nếu mai mưa thì hãy giúp tôi được đi chơi công viên sáng mai` | Zip: `nhớ mai mơ thì là hãy giúp tôi được đi chơi công viên sáng mai ước`
- **Gipformer Final**: `Nếu mai mưa thì hãy giúp tôi được đi chơi công viên sáng mai.`
- **ZipFormer Final**: `Nhớ mai mơ thì là hãy giúp tôi được đi chơi công viên sáng mai ước.`

### Mẫu 36 [ID: train_0039] - Gip WER: 40.0% | Zip WER: 20.0%
- **Gold**: `Ờ, à, nhắn qua Zalo cho Minh là tối nay ăn cơm ở nhà nhé.`
- **Raw Acoustic**: Gip: `à nhắn qua gia lô cho mình là tối nay anh không ở nhà nhé` | Zip: `à nhắn qua zalo cho minh là tối nay anh không ở nhà nhé`
- **Gipformer Final**: `À, nhắn qua gia lô cho mình là tối nay anh không ở nhà nhé.`
- **ZipFormer Final**: `À, nhắn qua Zalo cho minh là tối nay anh không ở nhà nhé.`

### Mẫu 37 [ID: train_0494] - Gip WER: 20.0% | Zip WER: 40.0%
- **Gold**: `Ừ, ghi vào giúp anh.`
- **Raw Acoustic**: Gip: `ờ ghi vào giúp anh` | Zip: `ờ ghi nhớ giúp anh`
- **Gipformer Final**: `Ờ, ghi vào giúp anh.`
- **ZipFormer Final**: `Ờ, ghi nhớ giúp anh.`

### Mẫu 38 [ID: train_0411] - Gip WER: 28.6% | Zip WER: 40.0%
- **Gold**: `Vàng SJC bán ra hôm nay trên 122 triệu thì nhắc tôi chín giờ sáng mai bán vàng, chưa tới thì chuyển Trần Văn Tuấn hai triệu từ tài khoản thanh toán, góp mua vàng.`
- **Raw Acoustic**: Gip: `vàng sjc bán ra hôm nay trên một trăm hai mươi hai triệu thì nhắc tôi chín giờ sáng mai bán vàng chưa tới thì chiến trong vân tuấn hai triệu từ tài khoản thanh toán góp mua vàng` | Zip: `vàng sgc bán ra hôm nay trên một trăm hai mươi hai triệu thì nhắc tôi chiến lược sáng mai bán vàng chưa tới thì chuyến trang vẫn tốn hai triệu từ tài khoản thanh toán gấp mua vàng`
- **Gipformer Final**: `Vàng SJC bán ra hôm nay trên một trăm hai mươi hai triệu thì nhắc tôi 9 giờ sáng mai bán vàng chưa tới thì chiến trong Vân Tuấn 2 triệu từ tài khoản thanh toán góp mua vàng.`
- **ZipFormer Final**: `Vàng SGC bán ra hôm nay trên một trăm hai mươi hai triệu thì nhắc tôi chiến lược sáng mai bán vàng chưa tới thì chuyến trang vẫn tốn 2 triệu từ tài khoản thanh toán gấp mua vàng.`

### Mẫu 39 [ID: train_0352] - Gip WER: 30.0% | Zip WER: 40.0%
- **Gold**: `Đặt báo thức bảy giờ sáng mai hộ chị, à rồi chuyển ví điện tử cho Bố ít tiền nữa.`
- **Raw Acoustic**: Gip: `đẹp bố thích bảy giờ sáng ra hộ trị à rồi chuyển ví điện tử cho bố ít tiền nữa` | Zip: `đạt bố thích bảy giờ sáng ba hội trị rồi chuyển ví điện tử cho bố ít tiền nữa`
- **Gipformer Final**: `Đẹp bố thích. 7 giờ sáng ra hộ trị à? Rồi chuyển ví điện tử cho bố ít tiền nữa.`
- **ZipFormer Final**: `Đạt bố thích 7 giờ sáng ba hội trị rồi chuyển ví điện tử cho bố ít tiền nữa.`

### Mẫu 40 [ID: train_0222] - Gip WER: 40.0% | Zip WER: 31.4%
- **Gold**: `Mai mà mưa thì huỷ lịch họp team rồi nhắn SMS báo Bố giúp mình, còn không mưa thì đặt ô tô 4 chỗ từ nhà đến chỗ hẹn, đón trước giờ hẹn nửa tiếng.`
- **Raw Acoustic**: Gip: `hôm nay mà mưa thì hãng lời nhắn báo bố giúp mình còn không mưa thì đặt bốn ô tô bốn chỗ từ nhà đến chỗ hãng đóng trước kỳ hạn lúc nửa tiếng` | Zip: `may mà mưa thì hẹn gặp tìm rồi nhắn báo bố giúp mình còn không mưa thì đặt bốn ô tô bốn chỗ từ nhà đến chỗ hẹn đóng trước kỳ hạn lúc nửa tiếng`
- **Gipformer Final**: `Hôm nay mà mưa thì hãng lời nhắn báo bố giúp mình, còn không mưa thì đặt 4 ô tô 4 chỗ từ nhà đến chỗ hãng đóng trước kỳ hạn lúc nửa tiếng.`
- **ZipFormer Final**: `May mà mưa thì hẹn gặp, tìm rồi nhắn báo bố giúp mình, còn không mưa thì đặt 4 ô tô 4 chỗ từ nhà đến chỗ hẹn đóng trước kỳ hạn lúc nửa tiếng.`

### Mẫu 41 [ID: train_0012] - Gip WER: 40.0% | Zip WER: 40.0%
- **Gold**: `Bạn ơi, chuyển cho chủ nhà Phạm Hùng hai triệu tám và nạp giúp sếp Dũng hai trăm nghìn nhé.`
- **Raw Acoustic**: Gip: `à chuyển cho chủ nhà phạm hùng hai triệu tám nè nạp giúp sếp dũng hai trăm ngàn nhé` | Zip: `à chuyển cho chủ nhà phạm hùng hai triệu tém tên đẹp nạp giúp sếp dũng hai trăm nghìn nhé`
- **Gipformer Final**: `À chuyển cho chủ nhà Phạm Hùng 2 triệu 8 nè, nạp giúp sếp Dũng 200 ngàn nhé.`
- **ZipFormer Final**: `À chuyển cho chủ nhà Phạm Hùng 2 triệu tém tên đẹp nạp giúp sếp Dũng 200 nghìn nhé.`

### Mẫu 42 [ID: train_0272] - Gip WER: 39.5% | Zip WER: 39.5%
- **Gold**: `Giá vàng SJC bán ra bữa nay trên 124 triệu thì nhắc tôi đi bán vàng 9 giờ sáng mai, không thì chuyển 2 triệu cho Nguyễn Thị Lan từ tài khoản thanh toán, tiền góp mua vàng.`
- **Raw Acoustic**: Gip: `giá vàng scb ra bữa nay cho nên một trăm hai mươi bốn triệu thì nó ở điểm bán vàng chín giờ sáng mai không tự chuyển hai triệu cho nguyễn thị lan từ tài khoản thanh toán tiền họ mua về` | Zip: `giá vàng sgc biến ra bữa nay trả năm một trăm hai mươi bốn triệu thì nó ở địa bán vàng chín giờ sáng mai không tự chuyển hai triệu cho nguyễn thị lan từ tài khoản thanh toán trường hợp mua vàng`
- **Gipformer Final**: `Giá vàng SCB ra bữa nay cho nên một trăm hai mươi bốn triệu thì nó ở điểm bán vàng 9 giờ sáng mai không tự chuyển 2 triệu cho Nguyễn Thị Lan từ tài khoản thanh toán tiền họ mua về.`
- **ZipFormer Final**: `Giá vàng SGC biến ra bữa nay trả năm một trăm hai mươi bốn triệu thì nó ở địa bán vàng 9 giờ sáng mai không tự chuyển 2 triệu cho Nguyễn Thị Lan từ tài khoản thanh toán. Trường hợp mua vàng.`

### Mẫu 43 [ID: train_0298] - Gip WER: 39.1% | Zip WER: 39.1%
- **Gold**: `Đặt món bún bò giò ở quán bún bò Huế giùm tôi, nhóm Gia đình mỗi người một phần, không tính tôi nghen.`
- **Raw Acoustic**: Gip: `món bún bò dạ ngoài bướn bò ghế giùm tôi nhóm gia đình mỗi người một phần không chính thôi nha` | Zip: `món bún bò dạ quầy bún bò ghế dùm tôi nhóm gia đình mỗi người một phần ông chính tôi nha`
- **Gipformer Final**: `Món bún bò dạ ngoài bướn bò ghế giùm tôi nhóm gia đình mỗi người một phần không chính thôi nha.`
- **ZipFormer Final**: `Món bún bò dạ quầy bún bò ghế dùm tôi, nhóm gia đình mỗi người một phần, ông chính tôi nha.`

### Mẫu 44 [ID: train_0019] - Gip WER: 30.8% | Zip WER: 38.5%
- **Gold**: `Chị ơi, giúp tôi đặt phòng khách sạn ở Đà Lạt với nhé.`
- **Raw Acoustic**: Gip: `chị ơi chúng tôi cần đặt phòng khách sạn ở đà lạt` | Zip: `tôi cần đặt phòng khách sạn ở đà lạt với giá`
- **Gipformer Final**: `Chị ơi chúng tôi cần đặt phòng khách sạn ở đà lạt`
- **ZipFormer Final**: `Tôi cần đặt phòng khách sạn ở Đà Lạt với giá.`

### Mẫu 45 [ID: train_0030] - Gip WER: 38.5% | Zip WER: 30.8%
- **Gold**: `Ơi, chuyển giúp tôi một triệu hai cho chủ nhà Phạm Hùng nhé.`
- **Raw Acoustic**: Gip: `ơi chuyển giúp tao một triệu hai cho chú nhà phạm hùng nghe` | Zip: `ơi chuyển giúp cháu một triệu hai cho chủ nhà phạm hùng nghe`
- **Gipformer Final**: `ơi chuyển giúp tao 1 triệu 2 cho chú nhà phạm hùng nghe`
- **ZipFormer Final**: `ơi chuyển giúp cháu 1 triệu 2 cho chủ nhà phạm hùng nghe`

### Mẫu 46 [ID: train_0500] - Gip WER: 30.8% | Zip WER: 38.5%
- **Gold**: `Chuyển cho Hòa một trăm hai mươi nghìn qua ví giúp anh cái.`
- **Raw Acoustic**: Gip: `chuyển cho hòa một trăm hai mươi nghìn qua ví giúp anh cái` | Zip: `chuyển cho hoà một trăm hai mươi nghìn qua ví giúp anh cái`
- **Gipformer Final**: `Chuyển cho Hòa 120 nghìn qua ví giúp anh cái.`
- **ZipFormer Final**: `Chuyển cho Hoà 120 nghìn qua ví giúp anh cái.`

### Mẫu 47 [ID: train_0076] - Gip WER: 38.2% | Zip WER: 38.2%
- **Gold**: `Đặt xe bốn chỗ từ công ty đến chỗ hẹn cà phê với gia đình chiều nay, đón sớm bốn mươi lăm phút, nhắc tôi mang laptop trước giờ xe đón mười lăm phút.`
- **Raw Acoustic**: Gip: `đặt xe đúng chỗ từ công ty đến chỗ hàng chặt tre với gia đình chiều nay toán sớm bốn mươi lăm phút nhắc tôi may lắp ráp trước giờ xe đón mười lăm phút` | Zip: `đặt xe đúng chỗ từ công ty đến chỗ hàng chặt ke với gia đình hiện nay sáng sớm bốn mươi năm phút nhắc tôi may lác đác trước giờ xe đón mười năm phút`
- **Gipformer Final**: `Đặt xe đúng chỗ từ công ty đến chỗ hàng chặt tre với gia đình, chiều nay Toán sớm 45 phút, nhắc tôi may lắp ráp trước giờ xe đón 15 phút.`
- **ZipFormer Final**: `Đặt xe đúng chỗ từ công ty đến chỗ hàng chặt ke với gia đình hiện nay sáng sớm 45 phút, nhắc tôi may lác đác trước giờ xe đón mười năm phút.`

### Mẫu 48 [ID: train_0162] - Gip WER: 19.0% | Zip WER: 38.1%
- **Gold**: `Chiều mai 5 giờ nhắc anh đi đổ xăng nha, với nhắn SMS cho Hương là nhớ mang hộ cái sạc.`
- **Raw Acoustic**: Gip: `chiều mai năm giờ nhắc anh đi đổ xăng nghe với nhắn ems cho hương là nhớ mang hộ khách sạn` | Zip: `chiều mai năm giờ nhắc anh đi đổ xăng nghe với nhánh yết đêm ít cha hương này nhớ mang hồ cái sạc`
- **Gipformer Final**: `Chiều mai 5 giờ nhắc anh đi đổ xăng nghe với nhắn ems cho Hương là nhớ mang hộ khách sạn.`
- **ZipFormer Final**: `Chiều mai 5 giờ nhắc anh đi đổ xăng nghe với nhánh yết đêm ít cha hương này nhớ mang hồ cái sạc.`

### Mẫu 49 [ID: train_0474] - Gip WER: 37.8% | Zip WER: 32.4%
- **Gold**: `Chị Lan đổi số rồi, em sửa trong danh bạ thành không chín không một, hai ba bốn, chín chín chín sáu giúp anh, xong nhắn Zalo cho chị ấy là em lưu số mới rồi nhé.`
- **Raw Acoustic**: Gip: `chị lan đổi số rồi em sửa trong danh bạ thành không chín không một hai ba bốn chín chín sáu giúp anh em nhắn gia lô cho chị ấy là em lưu số mới rồi nhé` | Zip: `chị lan đổi số rồi em sửa trong danh bạ thành không chín không một hai ba bốn chín chín sáu giúp anh em nhắn zalo cho chị ấy là em lưu số mới rồi nhé`
- **Gipformer Final**: `Chị Lan đổi số rồi, em sửa trong danh bạ thành 0 9 0 1 2 3 4 9 9 6 giúp anh em nhắn gia lô cho chị ấy là em lưu số mới rồi nhé.`
- **ZipFormer Final**: `Chị Lan đổi số rồi, em sửa trong danh bạ thành 0 9 0 1 2 3 4 9 9 6 giúp anh. Em nhắn Zalo cho chị ấy là em lưu số mới rồi nhé.`

### Mẫu 50 [ID: train_0387] - Gip WER: 28.6% | Zip WER: 37.1%
- **Gold**: `Vàng SJC bán ra mà vượt 123 triệu thì mai chín giờ sáng nhắc tôi đi bán vàng, chưa vượt thì chuyển Lê Minh 2 triệu từ tài khoản thanh toán, tiền góp mua vàng.`
- **Raw Acoustic**: Gip: `vàng hic bán ra mà vượt một trăm hai mươi ba triệu thì mai chín giờ sáng nhốt tôi đi bán vạn chưa vượt chuyển lê minh hai triệu từ tài khoản thanh toán tiền góp mua vàng` | Zip: `vàng sgc bán ra mà vượt một trăm hai ba triệu thì mai chín giờ sáng nhất tôi đi bán vạn thì chuyển lên hai triệu triệu tài khoản thanh toán tiền góp mua vàng`
- **Gipformer Final**: `Vàng Hic bán ra mà vượt một trăm hai mươi ba triệu thì mai 9 giờ sáng nhốt tôi đi bán. Vạn chưa vượt chuyển Lê Minh 2 triệu từ tài khoản thanh toán tiền góp mua vàng.`
- **ZipFormer Final**: `Vàng SGC bán ra mà vượt một trăm hai ba triệu thì mai 9 giờ sáng nhất tôi đi bán vạn thì chuyển lên 2 triệu triệu tài khoản thanh toán tiền góp mua vàng.`
