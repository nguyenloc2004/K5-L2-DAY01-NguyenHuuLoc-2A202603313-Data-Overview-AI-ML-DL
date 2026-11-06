# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU (`Device: cuda`)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả ba `PASS` checksum)

**Sample:** `traffic` (COCO 210273), `kitchen` (COCO 397133), `dining` (COCO 166918) (cả ba `PASS` checksum)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1. Chép record hạng 1:**

`class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

**2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao?**

Record nói về **cả bức ảnh**. Nó không có `bbox_xyxy` hay `polygon_xy` nên không chỉ vào vị trí nào. Mỗi ảnh cũng chỉ có một bộ 5 hạng. Vì vậy `cab` không có nghĩa là model đã tìm thấy một chiếc taxi cụ thể trong ảnh.

**3. Danh sách tên lớp do đâu mà có?**

Danh sách lớp đi kèm taxonomy ImageNet-1K của checkpoint `yolo11n-cls.pt`, không phải do model tự nghĩ ra. Model chỉ xếp hạng các lớp có sẵn, nên kết quả luôn nằm trong danh sách đó, kể cả khi không lớp nào mô tả đúng ảnh.

**4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`?**

Vì tên lớp và mã lớp chỉ có nghĩa trong một taxonomy. Ví dụ trong output của bài lab, bàn ăn là `dining_table` với `class_id`=532 ở ImageNet-1K, nhưng là `dining table` với `class_id`=60 ở COCO-80. Nếu chỉ ghi tên hoặc chỉ ghi mã thì khi gộp dữ liệu từ hai taxonomy có thể bị nhầm lớp.

**5. Nếu ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**

Guideline cần nói rõ nhãn mô tả **cái gì**: chủ thể chính hay toàn cảnh. Nếu theo chủ thể chính, cần có tiêu chí chọn, ví dụ vật chiếm diện tích lớn nhất hoặc nằm ở trung tâm ảnh. Guideline cũng cần nói phải làm gì khi nhiều vật ngang nhau: khi đó chuyển escalation, không chọn đại. Ảnh `traffic` có nhiều xe buýt, ô tô và người đi bộ, nên nếu không có quy tắc này thì hai annotator có thể chọn hai lớp khác nhau.

**6. Vì sao không được lấy `score` của model làm ground truth?**

`score` chỉ cho biết model xếp hạng lớp đó cao đến đâu, chưa có người nào đối chiếu với guideline. Score cao cũng không chứng minh prediction đúng. Trong ảnh `kitchen`, cùng một cái bát có box gần trùng nhau nhưng nhận `score` 0.701282 ở detection và 0.372406 ở segmentation (`kitchen-010`). Vật không đổi mà score đổi theo model, nên score không thể là nhãn đúng hay thước đo chất lượng nhãn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

**1. Chép một record:**

`class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

**2. Giải thích box bằng lời thường:**

Gốc `(0, 0)` nằm ở góc trên trái của ảnh 640×427 pixel. Góc trên-trái của box cách mép trái khoảng 385 pixel và cách mép trên khoảng 69 pixel. Góc dưới-phải cách mép trái khoảng 499 pixel và cách mép trên khoảng 349 pixel. Box rộng khoảng 114 pixel, cao khoảng 280 pixel và bao người đứng quay lưng từ đầu đến chân.

**3. Chọn hai ngưỡng. Mỗi ngưỡng có bao nhiêu prediction?**

| Ngưỡng | Số prediction | Các lớp |
| --- | ---: | --- |
| 0.20 | 17 | 2 person, 5 bowl, 2 oven, 2 cup, 3 spoon, 1 potted plant, 1 dining table, 1 bottle |
| 0.60 | 6 | 2 person, 2 bowl, 2 oven |

Từ 0.60 xuống 0.20 có thêm 11 prediction. Ở ngưỡng 0.35 có 11 prediction.

**4. Khi hạ ngưỡng, reviewer phải làm thêm việc gì?**

Reviewer phải xem thêm nhiều box hơn, ở đây là 11 box khi hạ từ 0.60 xuống 0.20. Với mỗi box cần kiểm tra vật có thật không, lớp có đúng không, box có sát không và có bị trùng không. Hạ ngưỡng có thể giữ lại vật đúng nhưng cũng có thể thêm box sai, nên reviewer không được dựa vào score để bỏ qua bước đối chiếu. Ngưỡng chỉ lọc prediction: vật nào không có box vẫn phải được gán nhãn nếu guideline yêu cầu.

**5. Viết một quy tắc box:**

Với mỗi vật thuộc class list, vẽ đúng một box sát phần nhìn thấy được: bốn cạnh chạm điểm ngoài cùng của vật, không chừa thêm nền và không ước đoán phần bị che hoặc nằm ngoài ảnh.

**6. Chỉ ra một vật bị che khuất hoặc cắt mép. Viết thành quy tắc hay escalation? Vì sao?**

Vật được chọn là cánh tay và bàn tay ở mép trái ảnh: `person`, `score`=0.610931, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78]. `x_min` gần bằng 0 cho thấy vật bị cắt bởi mép ảnh, phần còn lại của người nằm ngoài khung hình.

Mình viết ca này thành **quy tắc trong guideline**, vì vật bị cắt mép xuất hiện ở nhiều ảnh và cách xử lý áp dụng lặp lại được: vẫn gán `person` nếu nhận ra được là người, và box chỉ bao phần nhìn thấy. Chỉ escalation khi phần còn lại quá ít để nhận ra lớp, hoặc khi dự án chưa quy định kích thước tối thiểu cần gán nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

**1. Chép một record:**

`instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

**2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**

Box chỉ cho biết vị trí và kích thước bao ngoài, còn polygon đi theo đường viền của đầu, tay, thân và chân. Với `kitchen-001`, box của record có diện tích khoảng 31.760 px², còn polygon chỉ khoảng 18.170 px², tức khoảng 57% box. Phần còn lại của box là nền. Vật càng dài và mảnh thì khác biệt càng lớn: polygon của `kitchen-008` (spoon) chỉ bằng khoảng 24% box.

**3. `instance_id` dùng để làm gì? Hiểu thế nào về việc nó không phải class ID và không phải tracking ID?**

`instance_id` dùng để phân biệt từng đối tượng trong output của bài lab, theo dạng `<sample_id>-NNN`.

- **Không phải class ID:** `kitchen-002`, `kitchen-003`, `kitchen-006` và `kitchen-010` có bốn `instance_id` khác nhau nhưng cùng `class_id`=45 (bowl). Class ID cho biết vật là lớp gì, còn `instance_id` cho biết đó là vật nào trong các vật cùng lớp.
- **Không phải tracking ID:** tracking ID theo một vật qua nhiều frame của video. Bài lab chỉ dùng ảnh tĩnh riêng lẻ, nên `instance_id` không nói gì về việc hai vật ở hai ảnh có phải là một hay không.

**4. Viết một quy tắc vẽ biên mask:**

Mỗi instance có một mask riêng, vẽ sát mép ngoài phần nhìn thấy được và chỉ gồm pixel của chính vật đó. Khi vật khác che phía trước, biên dừng ở mép vật che, không vẽ đè lên vật che và không đoán phần bị khuất.

**5. Chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Dùng guideline hay escalation? Vì sao?**

Chỗ được chọn là hai vật chạm nhau ở góc trái bàn: `kitchen-002` (bowl, `score`=0.735744, box [31.44, 343.77, 100.89, 385.55]) là đĩa trắng viền gợn sóng, nằm phía trước vật tròn lớn `kitchen-006` (bowl, `score`=0.528201, box [0.11, 302.85, 85.7, 370.15]). Hai box giao nhau ở khoảng x 31–86, y 344–370.

- **Phần biên dùng guideline:** vật phía trước giữ toàn bộ pixel của mình, vật phía sau chỉ giữ phần còn thấy. Cách này lặp lại được ở nhiều ảnh.
- **Phần lớp dùng escalation:** `kitchen-006` trông giống một chiếc bánh trong khuôn hơn là bát rỗng. Cần hỏi người có thẩm quyền xem vật này tính là `bowl`, tính là đồ ăn hay bỏ qua, vì đây là quyết định về taxonomy, annotator không nên tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho cả ảnh, không có tọa độ | `traffic` có top-1 `cab` (0.510915), nhưng detection trên cùng ảnh ghi car=14, bus=8, person=4. Top-5 đều là loại xe riêng lẻ, không có lớp nào mô tả cảnh đường phố | Xem cả ảnh, chọn lớp theo guideline; nếu nhiều chủ đề ngang nhau thì escalation | Nhãn có đúng quy tắc chọn lớp không; các annotator có chọn nhất quán trên ảnh tương tự không |
| Phát hiện vật thể | Mỗi vật một lớp và một box pixel | `oven` 0.6343 tại [489.09, 200.73, 617.81, 344.88]: trên có chậu kim loại giống bồn rửa, dưới là tủ đen, chưa chắc là lò; segmentation không có instance ở đây. Nồi chảo treo tường không có box vì COCO-80 không có lớp nồi/chảo | Vẽ box sát từng vật trong class list; ghi lại vật khó xác định lớp; không chép prediction thành nhãn | Lớp đúng chưa, box có sát không, có trùng hoặc sót vật không; xác nhận cách xử lý vật ngoài taxonomy |
| Instance segmentation | Mỗi instance một lớp và một polygon/mask riêng | `kitchen-004` `potted plant` (0.632235) thực ra là bó lá treo, không thấy chậu. Mask `kitchen-005` `dining table` tràn sang cái xô dưới gầm bàn, và bàn này giống bàn bếp hơn bàn ăn | Vẽ mask theo từng instance, tách vật chạm nhau, loại vật không thuộc đối tượng | Biên có sát không, mask có tràn sang vật khác không, mỗi vật có đúng một instance không; trả rework nếu sai |

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?**

Chỉ dùng ba ảnh COCO công khai đã được notebook kiểm tra checksum, không đưa ảnh cá nhân, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab/GitHub, và không ghi thông tin cá nhân vào báo cáo hoặc output.

**2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**

Mình sẽ dừng ngay, không xử lý hay chia sẻ tiếp, và báo cho Lab Coach hoặc GV.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
