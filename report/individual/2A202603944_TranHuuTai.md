# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603944**
- **Họ và tên: Trần Hữu Tài**
- **SEED** (in ở ô đầu notebook): https://colab.research.google.com/drive/192QRJMZOMPvTu_xqRfK4gVwV5lmF78jB?usp=sharing

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

|Phương	 pháp|	F1 test |	F1 SUB 40|	p95 (ms	/tin)|	USD/1k tin|
| --- | --- | --- | --- | --- |
|BoW + Logistic Regr	ession|		0.878|		0.889|		0.055|	0|
|BoW + MLP|	0.948|	0.94|	7|	0.470|	0|
|MiniLM + kNN|	0.907|	1.000|	52.425|	0|
|MiniLM + Logistic Regression|	0.955|	1.000|	104.471|	0|
|Mi	nLM + MLP head	|0.969  |1.000|	35.213|	0|
|LLM nhỏ — zero-shot|	—	|1.000|	1351.324|	0.011|
|LLM lớn — zero-shot|	—	|1.000|	1507.913|	0.422|
|LLM nhỏ — few-shot|	—	|1.000|	1251.832|	0.025|

Nhận xét: MiniLM + MLP head có F1 test cao nhất trong nhóm ML/DL/Embedding. Tuy nhiên, BoW + MLP nhanh hơn nhiều, có p95 chỉ 0.470 ms. LLM đạt F1_SUB = 1.000 nhưng có độ trễ và chi phí cao.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

##### Bỏ ReLU: Mình dự đoán F1 sẽ gần Logistic Regression hoặc thấp hơn vì hai lớp Linear liên tiếp không học được quan hệ phi tuyến. Thực tế F1 vẫn đạt 0.948, bằng MLP có ReLU. Điều này cho thấy ReLU không tạo khác biệt đáng kể trên dữ liệu BoW hiện tại.

##### LLM lớn và nhỏ: Mình dự đoán model lớn có F1 cao hơn. Thực tế cả hai đều đạt F1_SUB = 1.000. Trong khi đó, chi phí model lớn là 0.422 USD/1k tin, cao hơn nhiều so với 0.011 USD/1k tin của model nhỏ.

##### Few-shot: Mình dự đoán thêm 4 ví dụ sẽ cải thiện F1. Thực tế F1_SUB vẫn là 1.000, không tăng so với zero-shot. Chi phí tăng từ 0.011 lên 0.025 USD/1k tin, khoảng 2.27 lần.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

##### BoW + Logistic Regression tụt F1 nhiều nhất, từ 0.889 trên SUB xuống 0.333 trên 50 case khó. BoW + MLP cũng giảm mạnh xuống 0.400. Ngược lại, LLM nhỏ zero-shot đạt 0.982. Hai ví dụ: Spam giọng tự nhiên: “Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed...” Tin nhắn được viết như lời chia sẻ cá nhân nhưng thực chất là quảng cáo việc làm đáng ngờ. BoW dễ bỏ sót vì thiếu từ khóa spam quen thuộc. Spam né từ khóa: Các cách viết như C0ngr4tulat10ns, F R E E, c.a.s.h khiến từ khóa không còn khớp với từ điển của BoW.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Với 10 triệu tin nhắn/ngày:
| Model|	Chi phí/tháng|
| --- | --- |
|LLM Small Zero-shot	|$3,300|
|LLM Large Zero-shot	|$126,600|
|LLM Small Few-shot	|$7,500|

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Chọn BoW + MLP làm mô hình chính vì đáp ứng yêu cầu tốc độ và có chi phí API bằng 0.
- Số liệu: p95 = 0.47 ms/tin, throughput = 43.591 tin/giây, F1 = 0.948.
- Tải dự kiến: 10 triệu SMS/ngày tương đương trung bình khoảng 116 tin/giây.
- Đánh đổi: F1 thấp hơn MiniLM + MLP (0.969), nhưng tốc độ cao hơn và dễ triển khai hơn.
- Hướng cải thiện: Có thể dùng MiniLM hoặc LLM để kiểm tra các tin nhắn có độ tin cậy thấp.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

|Phần |	Bạn hỏi AI gì (prompt)|	AI trả lời/viết gì|	Bạn đã sửa / kiểm chứng gì|
| --- | --- | --- | --- |
|Phân loại SMS|	Cách đánh giá mô hình spam|	Giải thích Accuracy, Precision, Recall, F1	|Đối chiếu kết quả notebook|
|MLP|	Cách cải thiện mô hình|	Giải thích ReLU và Overfitting|	Thử nghiệm trực tiếp trên notebook|
|LLM	|Cách tạo prompt JSON|	Đề xuất prompt và parse kết quả|	Kiểm tra JSON output|