# ZALO AI CHALLENGE 2026 - TỔNG HỢP TOÀN BỘ THÔNG SỐ KỸ THUẬT & QUY CHẾ

> **Nguồn trích xuất chính thức:** Cổng thông tin [Zalo AI Challenge 2026](https://challenge.zalo.ai/), hệ thống API cổng thi (`api-challenge.zalo.ai`), và Thể lệ chính thức (*Zalo AI Challenge 2026 Regulations* - Bản tiếng Việt & tiếng Anh).  
> **Thời gian giải đấu:** 26/10/2026 – 12/12/2026  
> **Tổng giá trị giải thưởng:** 12.000 USD (Tiền mặt + Quà tặng)

---

## 1. TÊN CHÍNH THỨC CỦA BÀI THI & MÔ TẢ BÀI TOÁN (PROBLEM STATEMENT)

Cuộc thi năm nay bao gồm 2 đề bài (Track/Task) độc lập xoay quanh chủ đề **Agentic AI**:

---

### Task 1: LifeMate — Agentic AI That Turns Your Voice Into Action

* **Slogan & Khái quát:** "Thế hệ thiết bị thông minh tiếp theo (loa thông minh, trợ lý trên xe hơi, tai nghe, smart TV...) sẽ có màn hình rất nhỏ hoặc không có màn hình, nhưng phải kết nối với hàng loạt ứng dụng và dịch vụ hàng ngày. Người dùng muốn nói bằng tiếng Việt tự nhiên (Bắc, Trung, Nam) để hoàn thành các tác vụ đa bước xuyên suốt nhiều ứng dụng: nhà thông minh, ngân hàng, gọi xe, giao đồ ăn, email, nhắc nhở... Agent được đánh giá dựa trên những gì nó LÀM ĐƯỢC (thay đổi trạng thái môi trường hệ thống), không phải những gì nó NÓI."
* **Mục tiêu bài toán (Problem Statement):**
  Xây dựng một hệ thống AI Agent có khả năng tiếp nhận yêu cầu bằng giọng nói tiếng Việt tự nhiên (kèm bối cảnh người dùng và các ứng dụng đã cài đặt trên thiết bị), lên kế hoạch các bước thực thi (planning), lựa chọn và gọi đúng các công cụ (tool calling) với tham số chính xác để đưa dữ liệu ứng dụng về đúng trạng thái người dùng mong muốn; hoặc chủ động hỏi lại thông tin còn thiếu (`sys.ask_missing_info`) nếu yêu cầu chưa đủ dữ kiện; hoặc không thực hiện hành động nào nếu yêu cầu nằm ngoài phạm vi các ứng dụng đã cài đặt.
* **Đầu vào (Input):**
  1. **Yêu cầu bằng giọng nói (Spoken request):** Tệp âm thanh `.wav` chứa yêu cầu mới nhất của người dùng bằng tiếng Việt (có thể pha trộn từ tiếng Anh thông dụng, giọng nói đa vùng miền Bắc - Trung - Nam, gồm cả giọng thật và giọng tổng hợp).
  2. **Bối cảnh người dùng (User context):** Thời gian hiện tại (`current_time`), múi giờ (`timezone`), tọa độ vị trí (`lat`, `lng`), thông tin hồ sơ (`user_profile`: địa chỉ nhà, công ty, thành phố...).
  3. **Ứng dụng khả dụng (Available apps & tools catalog):** Các ứng dụng được cài đặt trên thiết bị của người dùng, mỗi ứng dụng cung cấp một tập hợp các công cụ được mô tả bằng JSON Schema.
  4. **Lịch sử hội thoại (Conversation history):** Các lượt thoại trước đó giữa người dùng và trợ lý ảo (nếu có).
  5. **Môi trường mô phỏng (Simulated Tool Environment):** Chứa trạng thái dữ liệu các app của người dùng (thiết bị, tài khoản, danh bạ, lời nhắc...). Agent chỉ được đọc và ghi dữ liệu thông qua các lệnh gọi công cụ (tool calls).
* **Đầu ra (Output):**
  Agent xuất ra một danh sách có thứ tự các lệnh gọi công cụ (`tool_calls`) kèm tham số (`arguments`) để xử lý yêu cầu:
  * **Thực thi hành động:** Gọi tuần tự các tool tương ứng để hoàn thành tác vụ.
  * **Yêu cầu thông tin còn thiếu:** Gọi `sys.ask_missing_info` chỉ rõ `tool_name` dự định gọi và danh sách các tham số còn thiếu `params`. Lệnh này có thể diễn ra sau khi đã thực hiện một số tool đọc dữ liệu trước đó.
  * **Xử lý yêu cầu ngoài phạm vi (Out-of-scope):** Trả về mảng rỗng `[]` nếu không có ứng dụng nào cài đặt hỗ trợ yêu cầu.
  * **Mã băm trạng thái cuối (State Hash):** Sau khi thực thi chuỗi tool, agent lấy mã băm trạng thái môi trường từ `env.get_state_hash()` để nộp cùng danh sách tool calls.

---

### Task 2: DataMate — Talking To Your Data Through Agentic AI

* **Slogan & Khái quát:** "Hàng triệu chủ cửa hàng, giáo viên, phòng khám, văn phòng nhỏ tại Việt Nam quản lý công việc trên bảng tính Excel, CSV hoặc cơ sở dữ liệu SQLite đơn giản nhưng hầu như không ai biết viết câu truy vấn SQL. DataMate thách thức việc xây dựng một AI Agent có thể trả lời các câu hỏi thực tế bằng tiếng Việt trực tiếp trên dữ liệu bảng biểu hỗn tạp, nhiều sheet, tên sản phẩm không đồng nhất, từ ngữ địa phương và nghiệp vụ thực tế (doanh thu, giá vốn, công nợ, hạn sử dụng, tồn kho...). Agent phải tự tìm ra đáp án chính xác hoặc biết từ chối khi dữ liệu không đủ trả lời. Đúng có điểm, sai bị trừ điểm, không trả lời (abstain) thì không mất điểm."
* **Mục tiêu bài toán (Problem Statement):**
  Xây dựng một hệ thống AI Agent nhận vào câu hỏi bằng ngôn ngữ tự nhiên tiếng Việt và một gói dữ liệu của doanh nghiệp (bao gồm các file `.csv`, `.xlsx`, `.sqlite` và tài liệu mô tả `README.md`), cùng với ngày mốc (`as_of_date`), bộ luật nghiệp vụ chuẩn (`rulebook.md`) và lịch âm - dương (`calendar.csv`). Agent phân tích và trả về câu trả lời có định dạng chính xác theo yêu cầu hoặc chủ động từ chối trả lời (`abstain`) / xác định câu hỏi không thể trả lời từ dữ liệu (`unanswerable`).
* **Đầu vào (Input):**
  1. **Gói dữ liệu doanh nghiệp (Business Data Package):** Gồm một hoặc nhiều tệp `.csv`, `.xlsx`, `.sqlite` kèm theo tệp `README.md` mô tả bối cảnh kinh doanh, cấu trúc dữ liệu và bảng quy đổi đơn vị của doanh nghiệp.
  2. **Câu hỏi (Question):** Câu hỏi bằng tiếng Việt thông dụng (ví dụ: *"sp bán dc nhiều nhất tuần rồi là gì"*, *"Còn ai nợ tiền chưa trả?"*).
  3. **Ngày mốc (As-of date):** Ngày mốc (`YYYY-MM-DD`) để giải quyết các mốc thời gian tương đối ("hôm nay", "tuần rồi", "tháng này", "cùng kỳ năm ngoái").
  4. **Quy tắc nghiệp vụ (Rulebook) & Lịch âm dương (`calendar.csv`):** Tập quy tắc chuẩn hóa chung cho toàn bộ tập dữ liệu (cách tính doanh thu, đơn hủy, đơn trả, công nợ, giá vốn, tồn kho, phân loại ngành hàng, tra cứu ngày Tết/âm lịch...).
* **Đầu ra (Output):**
  Câu trả lời chuẩn hóa gồm kiểu dữ liệu (`type`) và giá trị (`answer`):
  * Thuộc một trong các kiểu giá trị: `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`.
  * Hoặc `{"type": "unanswerable", "reason": "..."}` khi dữ liệu không đủ cơ sở để trả lời.
  * Hoặc `{"type": "abstain"}` khi agent không chắc chắn và chủ động bỏ qua để tránh bị trừ điểm.

---

## 2. ĐỊNH DẠNG DỮ LIỆU ĐẦU VÀO & ĐẦU RA (DATA FORMAT)

### 2.1. Cấu trúc dữ liệu Task 1 (LifeMate)

#### Cấu trúc thư mục gói dữ liệu:
```text
train+public_test/
├── README.md                           # Tài liệu hướng dẫn chi tiết của Task 1
├── VERSION                             # Phiên bản gói dữ liệu, ngày build, số mẫu
├── train/
│   ├── train.jsonl                     # Mẫu huấn luyện có đầy đủ nhãn và audio_text
│   ├── train_tools_catalog.json        # JSON Schema toàn bộ công cụ của tập train
│   ├── train_environment/<id>/state.json  # Dữ liệu app của từng mẫu tại thời điểm người dùng ra lệnh
│   ├── audio/<id>/                     # <id>.wav (lượt cuối); <id>_u1.wav, <id>_u2.wav... (history)
│   └── zaic_tools/                     # Thư viện mã nguồn môi trường mô phỏng ToolEnv
├── public_test/
│   ├── public_test.jsonl               # Mẫu kiểm thử công khai (không có nhãn, không có audio_text)
│   ├── public_test_tools_catalog.json  # JSON Schema công cụ của public test
│   ├── public_test_environment/<id>/state.json # Dữ liệu khởi tạo môi trường public test
│   ├── audio/<id>/                     # File âm thanh tương ứng
│   ├── sample_submission.jsonl         # File nộp mẫu
│   └── zaic_tools/                     # Thư viện ToolEnv cho public test
└── private_test/                       # Phát hành sau khi đóng băng Docker
```

#### Cấu trúc file `train.jsonl`:
Mỗi dòng là một đối tượng JSON đại diện cho 1 mẫu:
```json
{
  "id": "train_0001",
  "audio": "audio/train_0001/train_0001.wav",
  "audio_text": "Nếu mai mưa thì dời chạy bộ muộn 1 tiếng, nhắn Sếp Dũng giờ mới; không mưa thì thôi.",
  "context": {
    "current_time": "2026-11-28T07:10:00+07:00",
    "timezone": "Asia/Ho_Chi_Minh",
    "location": {
      "lat": 16.0544,
      "lng": 108.2022
    },
    "user_profile": {
      "name": "Em Linh",
      "home_address": "88 Nguyễn Văn Linh, Hải Châu, Đà Nẵng",
      "work_address": "Công viên phần mềm, Hải Châu, Đà Nẵng",
      "city": "Đà Nẵng"
    },
    "installed_apps": {
      "weather": {"tools": ["weather.get_current", "weather.get_forecast"]},
      "calendar": {"tools": ["calendar.list_events", "calendar.create_event", "calendar.update_event", "calendar.cancel_event"]},
      "contacts": {"tools": ["contacts.search", "contacts.get", "contacts.update"]},
      "sms": {"tools": ["sms.send", "sms.read_recent"]},
      "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set", "smart_home.climate_set"]}
    }
  },
  "history": [
    {
      "role": "user",
      "audio": "audio/train_0001/train_0001_u1.wav",
      "audio_text": "..."
    },
    {
      "role": "assistant",
      "text": "...",
      "tool_calls": [...],
      "tool_results": [...]
    }
  ],
  "label": {
    "tool_calls": [
      {"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}},
      {"name": "calendar.list_events", "arguments": {}},
      {"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": "2026-11-29T09:00:00+07:00", "end": "2026-11-29T10:00:00+07:00"}},
      {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
      {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "mai chạy bộ dời sang 09:00 nhé"}}
    ],
    "tool_results": [
      {"ok": true, "forecast": [{"location": "Đà Nẵng", "date": "2026-11-29", "condition": "mưa rào", "rain": true}]},
      {"ok": true, "events": [{"id": "evt_01", "title": "chạy bộ", "start": "2026-11-29T08:00:00+07:00"}]},
      {"ok": true, "item": {"id": "evt_01", "title": "chạy bộ", "start": "2026-11-29T09:00:00+07:00"}},
      {"ok": true, "contacts": [{"id": "c_08", "name": "Sếp Dũng", "phone": "09925652941"}]},
      {"ok": true, "message": {"id": "msg_01", "contact_id": "c_08", "content": "mai chạy bộ dời sang 09:00 nhé"}}
    ],
    "final_state": {
      "calendar": {"events": [...]},
      "sms": {"messages": [...]}
    }
  }
}
```

#### Cấu trúc file `public_test.jsonl` / `private_test.jsonl`:
Tương tự `train.jsonl` nhưng **không có** trường `audio_text` và trường `label`. Agent bắt buộc phải dùng mô hình ASR để phiên âm audio trước khi đưa vào LLM Agent.

#### Cấu trúc & Cơ chế hoạt động của `ToolEnv` (`zaic_tools`):
1. **Khởi tạo:**
   ```python
   from zaic_tools import ToolEnv
   env = ToolEnv(split="public_test", question_id=item["id"], root=".")
   ```
   * Nạp `state.json` ban đầu của sample từ `<split>_environment/<id>/state.json`.
   * **Quy định nghiêm ngặt:** Mỗi mẫu (`id`) chỉ được khởi tạo duy nhất **1 instance** `ToolEnv`. Mọi cuộc gọi `env.call(...)` dù thành công hay lỗi schema đều được hệ thống ghi nhận lại vào lịch sử. Cấm tạo lại `ToolEnv` để reset trạng thái.
2. **Kiểm tra Schema (không đổi state):**
   ```python
   res = env.validate("smart_home.light_set", {"power": "off"})
   # Trả về: {"ok": False, "error": "thiếu tham số bắt buộc: device_id"}
   ```
3. **Thực thi gọi công cụ (thay đổi state trong RAM):**
   ```python
   res = env.call("smart_home.climate_set", {"device_id": "ac_living_room", "mode": "cool", "temperature": 26})
   # Trả về: {"ok": True, ...} hoặc {"ok": False, "error": "..."}
   ```
4. **Hỏi thông tin còn thiếu:**
   ```python
   env.call("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})
   ```
5. **Lấy kết quả nộp bài:**
   ```python
   tool_calls = env.get_tool_calls()   # Toàn bộ danh sách tool_calls đã thực hiện
   state_hash = env.get_state_hash()   # Mã băm SHA-256 trạng thái cuối của các app
   ```

#### Định dạng file nộp bài `submission.jsonl` (Task 1):
Mỗi dòng là 1 đối tượng JSON tương ứng với 1 mẫu trong file test:
```json
{"id": "public_0039", "tool_calls": [{"name": "alarm.list", "arguments": {}}, {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}], "state_hash": "461fcd40625c9968434b56ee113c0953f332f0c1189a3d385575e1dabec263f3"}
{"id": "public_0003", "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]}}], "state_hash": "23f1c7d2419741b689aaa6bf48a180d3d250730740bafdfe9e69a745fbb4bf7a"}
{"id": "public_0002", "tool_calls": [], "state_hash": "0c9a2ec0fff5f8137d1f5e256c0caabca26af73ff9f8619c1efbf7f321b07f98"}
```

---

### 2.2. Cấu trúc dữ liệu Task 2 (DataMate)

#### Cấu trúc thư mục gói dữ liệu:
```text
train/ (hoặc public_test/)
├── calendar.csv                        # Lịch âm - dương tra cứu ngày Tết, ngày lễ
├── rulebook.md                         # Quy tắc nghiệp vụ kinh doanh chuẩn
├── questions.jsonl                     # Danh sách câu hỏi
├── answers.jsonl                       # Đáp án chuẩn (chỉ có trong tập train)
└── data/                               # Dữ liệu của từng doanh nghiệp
    ├── biz_101/
    │   ├── README.md                   # Bối cảnh shop, quy ước cột, bảng quy đổi đơn vị
    │   ├── DonHang_ChiTiet.xlsx        # File bảng tính
    │   ├── BangGia.xlsx
    │   └── ... (.csv, .xlsx, .sqlite)
    ├── biz_102/
    └── ...
```

#### Cấu trúc file `questions.jsonl`:
```json
{"id": "q_001883280e83", "business_id": "biz_128", "question": "tháng này tổng giá trị các phiếu nhập hàng là bao nhiu", "as_of_date": "2026-09-07"}
{"id": "q_00195e1e4049", "business_id": "biz_124", "question": "Thuốc có hoạt chất Natri clorid tháng này bán được tổng cộng bao nhiêu tiền?", "as_of_date": "2026-05-25"}
{"id": "q_002d03ebbded", "business_id": "biz_120", "question": "Tổng doanh thu trong ngày hôm nay là bao nhiêu?", "as_of_date": "2025-04-30"}
```

#### Cấu trúc file `answers.jsonl` (tập train):
```json
{"id": "q_001883280e83", "type": "number", "answer": 89526400, "alias": []}
{"id": "q_00195e1e4049", "type": "number", "answer": 2954200, "alias": []}
{"id": "q_002d03ebbded", "type": "number", "answer": 10309400, "alias": []}
```

#### Định dạng file nộp bài `submission.jsonl` (Task 2):
```json
{"id": "q_001883280e83", "type": "number", "answer": 89526400}
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
{"id": "q_unans_example", "type": "unanswerable", "reason": "Dữ liệu không có thông tin chi phí vận chuyển"}
{"id": "q_abstain_exam", "type": "abstain"}
```

---

## 3. CÔNG THỨC CHẤM ĐIỂM (EVALUATION METRIC)

### 3.1. Đánh giá Task 1: LifeMate

Mỗi mẫu được chấm dựa trên 2 yếu tố:
1. **Khớp trạng thái (State Match):**
   * Giá trị `state_hash` nộp lên phải **bằng tuyệt đối** với `state_hash` của đáp án chuẩn (gold state hash).
   * Nếu `state_hash` khác nhau $\rightarrow$ Mẫu đó nhận **0 điểm** ngay lập tức (không tính điểm tool-call).
   * Các trường văn bản tự do (fuzzy text fields) không ảnh hưởng đến `state_hash`.
   * Ban tổ chức sẽ replay lại toàn bộ `tool_calls` trên môi trường chuẩn `zaic_tools` để tái tạo `state_hash`. Nếu có sự sai lệch giữa kết quả nộp và kết quả replay của BTC, kết quả của BTC là quyết định cuối cùng.
2. **Khớp chuỗi gọi công cụ (Tool-call Match):**
   * Một lệnh gọi nộp được tính là khớp với một lệnh gọi gold khi:
     - Tên tool (`name`) giống nhau hoàn toàn.
     - Mọi tham số không phải fuzzy (`non-fuzzy arguments`) giống nhau tuyệt đối.
     - Mọi tham số dạng text tự do (`fuzzy arguments`) đạt **token-F1 > 0.6** (sau khi chuyển chữ thường lowercase và chuẩn hóa Unicode NFC).
   * Mỗi lệnh gọi nộp chỉ được ghép với tối đa một lệnh gọi gold; thứ tự gọi tool không ảnh hưởng đến việc đối sánh khớp.
   * Lệnh `sys.ask_missing_info` được chấm như một tool thông thường.
3. **Mẫu ngoài phạm vi (Out-of-scope samples):**
   * Nếu danh sách gold là rỗng (`[]`), mẫu chỉ được **1 điểm** khi danh sách nộp cũng là rỗng (`[]`); nếu nộp bất kỳ tool call nào thì nhận **0 điểm**.

#### Công thức toán học tính điểm:
* Nếu danh sách gold rỗng ($|G| = 0$):
  $$\text{Score} = \begin{cases} 1, & \text{nếu danh sách nộp } \text{tool\_calls} = [] \\ 0, & \text{ngược lại} \end{cases}$$
* Nếu danh sách gold không rỗng ($|G| > 0$):
  $$P = \frac{M}{M + U}, \quad R = \frac{M}{|G|}$$
  $$\text{Score} = \begin{cases} 0, & \text{nếu } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ hoặc } M = 0 \\ \frac{2 \cdot P \cdot R}{P + R}, & \text{ngược lại} \end{cases}$$
* **Điểm tổng kết bài thi (Final Score):**
  $$\text{Final score} = \frac{100}{N} \sum_{i=1}^{N} \text{Score}_i$$
  *Trong đó:*
  * $N$: Tổng số mẫu trong tập kiểm thử.
  * $M$: Số lượng tool call nộp khớp với tool call gold.
  * $|G|$: Tổng số lượng tool call trong đáp án gold.
  * $U$: Số lượng tool call nộp không khớp (bao gồm cả các lệnh gọi bị từ chối do lỗi schema).
  * $P$: Precision (Độ chuẩn xác).
  * $R$: Recall (Độ bao phủ).
  * $\text{Score}_i$: Điểm của mẫu thứ $i$.

---

### 3.2. Đánh giá Task 2: DataMate

#### Bảng quy chuẩn kiểu dữ liệu và quy tắc so khớp (Matching Rules):

| Type | Nội dung đáp án | Ví dụ mẫu | Quy tắc so khớp (Matching Rule) |
|---|---|---|---|
| `number` | Giá trị số (tiền VND, số đếm, số lượng...) | `48250000`, `18.6` | Khớp tuyệt đối (`Exact match`). Giá trị chuẩn là số nguyên (VND, số lượng) hoặc số thực làm tròn 2 chữ số thập phân. Không kèm đơn vị, không dấu phân cách nghìn. |
| `percent` | Giá trị phần trăm | `-12.45` | Khớp tuyệt đối sau khi làm tròn 2 chữ số thập phân; dấu âm/dương phải chính xác. Không kèm ký hiệu `%`. |
| `string` | Tên gọi (sản phẩm, khách hàng, chi nhánh, thứ...) | `"Sting dâu 330ml"` | Chuẩn hóa cả câu trả lời và các dạng chấp nhận được của đáp án rồi so sánh bằng tuyệt đối: (1) Unicode NFC; (2) Chuyển chữ thường; (3) Thay thế mọi ký tự không phải chữ cái, chữ số hoặc khoảng trắng thành 1 dấu cách; (4) Xóa khoảng trắng đầu/cuối và thu gọn chuỗi khoảng trắng liên tiếp về 1 dấu cách. Giữ nguyên dấu tiếng Việt. Đáp án chấp nhận mọi cách viết xuất hiện trong dữ liệu hoặc mã tương ứng (ví dụ: `sting dâu 330ml`, `sp0042`). |
| `date` | Ngày theo chuẩn ISO | `"2026-08-15"`, `"2025-09"` | Khớp tuyệt đối theo định dạng `yyyy-mm-dd`; câu hỏi theo tháng: `yyyy-mm`. |
| `boolean` | Giá trị logic đúng/sai | `true` hoặc `false` | Khớp tuyệt đối (`true` / `false` dạng JSON boolean, không dùng chuỗi `"true"` hay số `1`). |
| `list` | Danh sách tên hoặc số có thứ tự | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | Luôn có thứ tự, so khớp từng vị trí (`position by position`). Thứ tự theo tiêu chí câu hỏi yêu cầu, hoặc theo quy định mặc định của Rulebook. Các phần tử khớp theo luật của `string` hoặc `number`. |
| `table` | Bảng nhỏ gồm tên cột và các dòng | `{"columns": ["khach_hang", "no"], "rows": [["Chị Lan", 3200000]]}` | Cột được so khớp theo tên cột đã chuẩn hóa; nếu tên không khớp sẽ so theo vị trí. Các dòng được so khớp dưới dạng một tập hợp (set). Ô chuỗi khớp theo luật `string`, ô số khớp theo luật `number`. |
| `unanswerable` | Dữ liệu không đủ thông tin để trả lời | `{"type": "unanswerable", "reason": "..."}` | Tính là ĐÚNG nếu và chỉ nếu câu hỏi đó thực sự không thể trả lời từ dữ liệu. |
| `abstain` | Agent chủ động từ chối trả lời | `{"type": "abstain"}` | Được chấm là `abstain` bất kể đáp án chuẩn là gì. Bỏ trống dòng cũng coi là `abstain`. |

#### Quy tắc câu hỏi xếp hạng (Ranking Questions):
* Xếp hạng theo tiêu chí câu hỏi nêu, hoặc quy định mặc định trong Rulebook, theo thứ tự **giảm dần** của tiêu chí đó và chỉ tính trên các đơn hợp lệ.
* Các phần tử có giá trị bằng nhau được sắp xếp theo **mã tăng dần** (nếu không có mã thì theo tên).
* Các câu hỏi mà đáp án tham chiếu có sự hòa điểm ở ranh giới top-$k$ (hoặc ở vị trí số 1 đối với câu hỏi "...nhất") đều đã được loại bỏ khỏi bộ test.
* Khi có ít hơn $k$ ứng viên thỏa mãn, đáp án là toàn bộ các ứng viên đó.

#### Công thức tính điểm Penalized Accuracy (PA):
Điểm của cả bài thi được tính bằng độ chính xác có phạt (Penalized Accuracy) trên $N$ câu hỏi của bộ test:
$$\text{PA} = \frac{1}{N} \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} +1, & \text{nếu câu trả lời } i \text{ đúng (correct)} \\ 0, & \text{nếu câu trả lời } i \text{ là abstain (hoặc bỏ trống)} \\ -1, & \text{nếu câu trả lời } i \text{ sai (wrong, hoặc trả lời vào câu unanswerable)} \end{cases}$$
* Không tính điểm thành phần (No partial credit).
* Tất cả câu hỏi có trọng số bằng nhau.
* Bất kỳ câu trả lời nào đưa ra cho một câu hỏi `unanswerable` (thay vì trả về unanswerable hoặc abstain) đều bị tính là **sai (-1 điểm)**.

---

## 4. QUY ĐỊNH & GIỚI HẠN KỸ THUẬT (RULES & CONSTRAINTS)

### 4.1. Bảng so sánh giới hạn kỹ thuật giữa 2 Task

| Tiêu chí | Task 1: LifeMate | Task 2: DataMate |
|---|---|---|
| **Giới hạn số lượng tham số mô hình** | **Toàn bộ hệ thống < 6 tỷ tham số (6B)** (tính tổng cả ASR + LLM + các thành phần khác). Riêng LLM **< 5B tham số**. | **Tổng số tham số $\le$ 15 tỷ (15B)** tính gộp trên tất cả các mô hình được dùng ở inference (LLM, Embedding, Reranker, Classifier, tính toàn bộ tham số của MoE). |
| **Giới hạn độ dài ngữ cảnh (Context length)** | LLM có context length tối đa **16K (16.384 tokens)**. | Không giới hạn cứng context length, nhưng phải chạy trong giới hạn RAM/VRAM và thời gian. |
| **Quy định mô hình huấn luyện sẵn (Pre-trained)** | Được phép dùng mô hình nguồn mở phát hành trước ngày **26/10/2026** với giấy phép hợp lệ. | Được phép dùng mô hình công khai trọng số trước ngày **01/10/2026** với giấy phép hợp lệ. Cho phép fine-tune tự do. Cho phép mọi phương pháp lượng tử hóa (Quantization). |
| **Quy định về External API & Network** | **CẤM HOÀN TOÀN** truy cập Internet hoặc gọi external API (OpenAI, Anthropic, Gemini, Zalo API thương mại...) ở bước inference. Container chạy offline. | **CẤM HOÀN TOÀN** gọi API bên ngoài, cấm tải dữ liệu, cấm truy cập mạng. Docker image chạy với cờ `--network none`. |
| **Dữ liệu huấn luyện & Sinh dữ liệu** | Được sinh dữ liệu tổng hợp bằng LLM/TTS trong giai đoạn phát triển, nhưng không được dùng dịch vụ ngoài khi inference. Cấm gán nhãn public/private test. | Được dùng dữ liệu BTC, dữ liệu tự tổng hợp, dữ liệu công khai hợp lệ. Cấm dùng test set để train hoặc gán nhãn thủ công. |
| **Thực thi mã lệnh (Code Execution)** | Không hỗ trợ chạy code tùy ý ngoài `ToolEnv`. Cấm đọc trực tiếp `state.json`. | **Được phép tạo và thực thi code** (Python/SQLite) bên trong Docker container để tính toán trên dữ liệu. |
| **Cấu hình phần cứng chấm thi (BTC)** | **1 × NVIDIA A30 GPU**, CPU Intel Xeon Gold 6442Y (16 cores), **64 GB RAM**. | **2 × NVIDIA RTX 5090 (32 GB VRAM mỗi card, tổng 64 GB VRAM)**, **512 GB RAM**. Yêu cầu CUDA $\ge$ 12.8 cho kiến trúc Blackwell. |
| **Giới hạn thời gian chạy (Runtime limit)** | **Tối đa 30 giây / mẫu**. Mẫu quá thời gian hoặc lỗi crash bị **0 điểm**. | **Tối đa 6 giờ (wall-clock time)** cho toàn bộ tập Private Test (bao gồm cả thời gian nạp mô hình). |
| **Môi trường nộp bài Docker (Vòng Private)** | Mount thư mục test tại `/data`, script inference chạy và ghi kết quả ra `/result/submission.jsonl`. | Mount thư mục test (chế độ read-only) tại `/data`, script inference ghi kết quả ra `/result/submission.jsonl`. |
| **Yêu cầu tính tái lập (Reproducibility)** | Cố định random seeds (PyTorch, NumPy, Python...). Giải mã tất định (`temperature = 0` hoặc greedy decoding). Nếu kết quả lệch, BTC chạy 5 lần lấy điểm trung bình. | Giải pháp phải có tính tất định: Độ lệch điểm $|\text{PA}_{\text{uploaded}} - \text{PA}_{\text{re-run}}| \le 0.03$. Lệch quá 0.03 mà không giải trình được sẽ bị hủy tư cách thi. |
| **Tiêu chí phụ phá vỡ thế hòa (Tie-breaking)** | 1. Tính F1 của `tool_calls` bỏ qua kiểm tra `state_hash`.<br>2. Đội có thời gian xử lý trung bình mỗi sample thấp hơn sẽ xếp trên. | 1. Tổng thời gian inference trong lần re-run của BTC thấp hơn.<br>2. Thời điểm nộp bài sớm hơn. |

---

### 4.2. Giới hạn số lượt nộp bài (Submission Quota)

Theo cấu hình hệ thống từ API BTC:
* **Public Test:** Tối đa **5 lượt nộp / ngày** cho mỗi đội (quota tự động reset vào 00:00 GMT+7 mỗi ngày). File nộp định dạng `.jsonl`.
* **Final Solution Submission (Docker image):** Tối đa **2 lần nộp** qua form nộp giải pháp (chỉ đánh giá bản nộp cuối cùng). Do file Docker rất lớn, các đội cung cấp mã băm checksum (MD5/SHA256) kèm link Google Drive công khai.
* **Private Test:** Tối đa **2 lần nộp** file kết quả `.jsonl` trong khoảng thời gian mở private test.

---

## 5. CÂU HỎI THƯỜNG GẶP (FAQ) & LƯU Ý ĐẶC BIỆT TỪ BAN TỔ CHỨC

### 5.1. Câu hỏi thường gặp chính thức (Official FAQ)

* **Q: Có điều kiện gì để tham gia cuộc thi không?**  
  *A:* Zalo AI Challenge mở cho tất cả những người đam mê AI mang quốc tịch Việt Nam, từ 18 tuổi trở lên (hoặc có sự đồng ý hợp pháp của phụ huynh/người giám hộ). Khuyến khích thí sinh có kiến thức lập trình cơ bản. Nhân viên, cộng tác viên đang làm việc cho VNG và các công ty thuộc tập đoàn không được phép tham gia.
* **Q: Đã từng tham gia hoặc đạt giải ở các mùa Zalo AI Challenge trước thì năm nay có được thi không?**  
  *A:* Hoàn toàn ĐƯỢC! BTC chào đón tất cả các tài năng AI quay trở lại thử sức.
* **Q: Có thể thay đổi thành viên trong đội sau khi đăng ký không?**  
  *A:* Đội trưởng có thể vào mục "Quản lý đội thi" để thêm hoặc xóa thành viên (tối đa 5 thành viên/đội, mỗi cá nhân chỉ được tham gia duy nhất 1 đội).
* **Q: Thời hạn chót để thay đổi thành viên trong đội là khi nào?**  
  *A:* Khuyến khích chốt danh sách trước ít nhất 1 tuần trước hạn nộp giải pháp. Từ ngày **17/11/2026**, tính năng thêm/xóa thành viên trên hệ thống sẽ bị khóa hoàn toàn.
* **Q: Khi nào các đội đạt giải nhận được tiền thưởng?**  
  *A:* Đội thắng giải phải nộp đầy đủ hồ sơ xác nhận nhận thưởng trong vòng 2 tuần kể từ ngày thông báo (nếu không sẽ bị hủy giải). Tiền thưởng sẽ được chi trả trong vòng 30 ngày sau khi BTC nhận đủ hồ sơ hợp lệ. Giải thưởng không được chuyển nhượng cho người khác.
* **Q: Kênh hỗ trợ kỹ thuật của BTC là gì?**  
  *A:* Gửi email trực tiếp đến địa chỉ: `aichallenge@zalo.me`.

---

### 5.2. Các lưu ý đặc biệt & Hành vi nghiêm cấm (Critical Rules)

1. **Tuyệt đối không truy cập trực tiếp `state.json` (Task 1):**  
   Mọi dữ liệu ứng dụng phải được đọc và thay đổi thông qua instance `ToolEnv` do thư viện `zaic_tools` cung cấp. Mọi hành vi đọc trực tiếp file `state.json` từ đĩa hoặc can thiệp mã nguồn `zaic_tools` đều bị coi là gian lận và loại khỏi cuộc thi.
2. **Quy tắc gộp đội (Team Merge):**  
   Được phép gộp đội nếu thỏa mãn 2 điều kiện: (1) Tổng số thành viên sau khi gộp không quá 5 người; (2) Tổng số lượt nộp bài tích lũy của đội sau khi gộp không vượt quá tổng hạn ngạch tối đa tính đến thời điểm gộp:
   $$\text{Max Submissions} = (\text{Số lượt cho phép/ngày}) \times (\text{Số ngày diễn ra cuộc thi})$$
3. **Cấm chia sẻ mã nguồn riêng tư:**  
   Trong suốt thời gian thi đấu, cấm chia sẻ riêng mã nguồn, mô hình hoặc dữ liệu thi đấu ra ngoài đội thi. Mọi chia sẻ công khai chỉ được chấp nhận nếu đăng tải công khai cho toàn bộ thí sinh trên diễn đàn chính thức của cuộc thi theo giấy phép mã nguồn mở OSI.
4. **Không gắn nhãn dữ liệu Test (No Human-in-the-loop):**  
   Nghiêm cấm mọi hành vi gắn nhãn thủ công hoặc sử dụng mô hình/dịch vụ bên ngoài để giải các câu hỏi trên tập Public Test và Private Test.
5. **Cơ cấu giải thưởng từng Track:**
   * **Giải Nhất (First Prize):** 3.500 USD tiền mặt (~ 90.412.000 VND).
   * **Giải Nhì (Second Prize):** 2.500 USD tiền mặt (~ 64.580.000 VND).
   * **Top 5 mỗi đề bài:** Nhận gói quà tặng vinh danh từ Zalo AI Challenge (ZAIC gift package).

---

## 6. MỐC THỜI GIAN QUAN TRỌNG (TIMELINE)

* **26/10/2026:** Chính thức mở cổng thi đấu. Công bố bộ phát triển (Dev kit), dữ liệu huấn luyện (Train), dữ liệu kiểm thử công khai (Public Test) và mã nguồn đánh giá.
* **16/11/2026 (23:59 GMT+7):** Hạn chót nộp toàn bộ giải pháp đóng gói Docker (Training code, Testing/Inference code, Models weights, Checksum & Link Drive).
* **17/11/2026:** Khóa tính năng thay đổi thành viên đội thi.
* **20/11/2026 (09:00 GMT+7):** Công bố tập dữ liệu kiểm thử riêng (Private Test).
* **23/11/2026 (23:59 GMT+7):** Hạn chót nộp file kết quả Private Test (`submission.jsonl`).
* **02/12/2026:** Công bố Top 2 đội xuất sắc nhất của mỗi đề bài vào vòng chung kết.
* **12/12/2026:** Lễ trao giải và công bố thứ hạng chung cuộc chính thức tại Zalo AI Summit 2026.
