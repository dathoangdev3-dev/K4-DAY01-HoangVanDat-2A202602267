# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

---

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

**Record hạng 1:**

| Trường | Giá trị |
|---|---|
| `class_id` | `468` |
| `class_name` | `cab` |
| `rank` | `1` |
| `score` | `0.510915` |
| `taxonomy_name` | `ImageNet-1K` |

**Record này mô tả toàn ảnh như thế nào?**
Model phân tích toàn bộ ảnh như một đơn vị duy nhất và gán nhãn `cab` (taxi) với xác suất 51.1%. Đây là nhãn cấp ảnh — không định vị vật thể trong ảnh mà chỉ nói "toàn bộ khung hình này trông giống một bức ảnh có taxi". Các hạng tiếp theo (`minibus` 16.4%, `police_van` 8.6%, `recreational_vehicle` 5.4%, `streetcar` 4.8%) cho thấy model đang phân phối xác suất sang nhiều lớp phương tiện khác — hợp lý vì đây là cảnh giao thông đông đúc.

**Ai định nghĩa class list mà checkpoint có thể dự đoán?**
Nhóm nghiên cứu ImageNet/Stanford tổng hợp và công bố danh sách 1.000 lớp của **ImageNet-1K**, sau đó Ultralytics sử dụng danh sách này khi huấn luyện `yolo11n-cls.pt`. Đội annotation không tự chọn nhãn — họ bị ràng buộc bởi taxonomy đã được định nghĩa sẵn từ trước.

**Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**
- `class_id` (số nguyên) là khoá tra cứu ổn định, không bị thay đổi khi đổi ngôn ngữ hay chuẩn hoá tên.
- `class_name` là chuỗi con người đọc được, giúp reviewer kiểm tra nhanh không cần mapping.
- `taxonomy_name` (`ImageNet-1K`) xác định *phiên bản từ điển nhãn* được dùng. Khi đổi sang taxonomy khác (ví dụ COCO-80 hay OpenImages), cùng một `class_id` có thể trỏ sang class hoàn toàn khác — mất context sẽ gây lỗi không thể trace.

**Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**
Guideline phải xác định rõ **chủ thể chính** là gì (vật thể chiếm diện tích lớn nhất, nằm trung tâm, hay được đề cập trong caption nguồn). Nếu không, hai annotator có thể gán nhãn khác nhau cho cùng một ảnh có cả người lẫn xe. Nên thêm trường `label_basis` (ví dụ: `dominant_object`, `scene_type`) để trace quyết định.

**Vì sao model score không phải ground truth?**
Score là xác suất do model tự ước lượng dựa trên pattern đã thấy trong tập huấn luyện — nó phản ánh **mức độ tự tin của model**, không phải sự thật khách quan của ảnh. Model có thể tự tin sai (false positive cao score) hoặc do sai phân phối dữ liệu. Ground truth chỉ được xác lập khi con người (annotator + reviewer) xác nhận nhãn sau khi nhìn vào nội dung thực của ảnh.

---

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

**Một record điển hình (hạng 1 – kitchen):**

| Trường | Giá trị |
|---|---|
| `class_name` | `person` |
| `score` | `0.912625` |
| `bbox_xyxy` | `[385.33, 69.24, 498.92, 348.92]` |
| `bbox_width` | `113.58 px` |
| `bbox_height` | `279.68 px` |

**Diễn giải vị trí box bằng lời:**
Box bắt đầu tại điểm `(385, 69)` — tức nằm ở khoảng **2/3 bề ngang**, gần **sát mép trên** của ảnh 640×427 px. Box kéo rộng ~114 px sang phải và ~280 px xuống dưới, bao phủ một người đứng chiếm phần lớn nửa phải khung hình, từ đầu đến khoảng ngang hông.

**So sánh số prediction ở hai threshold:**

| Threshold | Số prediction (kitchen) | Ghi chú |
|---|---|---|
| `score ≥ 0.35` (hiện tại) | **11 records** | person ×2, bowl ×4, oven ×2, cup ×2, (thêm 1 bowl thấp) |
| `score ≥ 0.50` | **7 records** | loại bỏ các bowl/cup ≤ 0.46 |

**Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer?**
Hạ threshold → nhiều vật thể nhỏ/mờ được bắt hơn (tăng recall), nhưng reviewer phải xem xét và xác nhận thêm nhiều prediction có xác suất thấp, dẫn đến **workload tăng** và nguy cơ reviewer bỏ sót lỗi khi mệt mỏi. Nâng threshold → ít false positive, reviewer làm việc nhanh hơn, nhưng có thể bỏ lọt vật thể thật (giảm recall).

**Đề xuất một quy tắc box chặt:**
> Box phải bao sát vật thể theo nguyên tắc **tight-fit**: khoảng trống từ cạnh box đến biên vật thể không vượt quá 5% chiều dài cạnh tương ứng. Không bao bóng, bề mặt phản chiếu, hay vùng nền lân cận. Áp dụng cho tất cả object có `score ≥ 0.35`.

**Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**
Guideline cần quy định tỉ lệ tối thiểu vật thể nhìn thấy để vẫn annotate (ví dụ: ≥ 30% diện tích visible). Trường hợp vật thể cắt mép ảnh — box được phép chạm/vượt biên ảnh hay bị clamp về 0/max? Các trường hợp mơ hồ (bị che >70%, chồng lấn 2 vật thể, loại chưa có trong taxonomy) cần **escalation** lên lead annotator hoặc ontology owner để ra quyết định nhất quán, tránh mỗi annotator tự xử lý một kiểu.

---

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `traffic`.

**Một record điển hình (instance `traffic-001`):**

| Trường | Giá trị |
|---|---|
| `instance_id` | `traffic-001` |
| `class_name` | `bus` |
| `score` | `0.925745` |
| Số điểm polygon | `120 điểm` |
| Một phần `polygon_xy` (5 điểm đầu) | `[148,189], [147,190], [145,190], [143,192], [142,192]` |

**Polygon bổ sung chi tiết gì so với box?**
Box chỉ cho biết *vùng chữ nhật bao quanh* vật thể, bao gồm cả nền xung quanh. Polygon vạch ra **biên thực tế của vật thể** theo từng pixel — giúp tách instance khỏi nền, xác định chính xác diện tích bề mặt vật thể, và hỗ trợ các tác vụ cần mask chính xác như inpainting, background removal, hay đo đạc kích thước thực. Với instance `traffic-001`, 120 điểm polygon bao sát thân xe buýt, phân biệt rõ phần xe với đường và các xe khác mà box không làm được.

**`instance_id` dùng để làm gì và không phải loại ID nào?**
`instance_id` (`traffic-001`, `traffic-002`, …) là **khoá định danh duy nhất cho từng đối tượng riêng lẻ trong một ảnh**, dùng để:
- Liên kết segmentation mask với metadata (class, score, bbox) trong pipeline.
- Theo dõi sự thay đổi annotation qua các vòng review (rework).
- Join dữ liệu khi xuất sang định dạng COCO hay Cityscapes.

Nó **không phải** `class_id` (không định danh lớp vật thể), không phải `coco_image_id` (không định danh ảnh), và không phải ID người dùng hay ID cá nhân — không mang thông tin nhạy cảm.

**Đề xuất một quy tắc biên mask:**
> Biên polygon phải bám theo contour thực của vật thể với độ chính xác ±2 px ở vùng cạnh sắc nét và ±5 px ở vùng mờ/nhòe. Không được để polygon cắt vào phần thân vật thể hoặc kéo ra ngoài biên quá 3 px. Số điểm tối thiểu là 20 cho vật thể nhỏ (<100×100 px) và tối đa 200 điểm cho vật thể lớn, trừ khi cần thiết để nắm biên cong.

**Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**
Guideline phải nêu rõ: Khi hai vật thể tiếp xúc/chồng lấn, đường biên polygon đi theo *vật thể phía trước* hay *chia đôi vùng giao*? Vùng bị che khuất hoàn toàn có được annotate polygon không? Nếu biên vật thể bị mờ do motion blur hoặc khuất sau kính, annotator áp dụng quy tắc nào để ước lượng? Những trường hợp không có đáp án rõ trong guideline cần **escalation** để lead hoặc ontology owner ra quyết định và cập nhật guideline cho nhất quán.

---

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn lớp duy nhất per ảnh (class_id + class_name từ ImageNet-1K taxonomy) | Ảnh đa chủ thể gây mơ hồ (traffic có cả taxi lẫn bus, minibus – model phân tán xác suất); `cab` 51% nhưng xung quanh đầy phương tiện khác | Gán nhãn theo chủ thể chiếm ưu thế hoặc theo quy tắc guideline (scene-type vs dominant-object); ghi chú nếu ảnh mơ hồ | So score với ngưỡng chất lượng; kiểm tra nhãn có hợp lý với nội dung ảnh; flag ảnh đa chủ thể cho lead xét lại |
| Phát hiện vật thể | Bounding box xyxy (pixel) + class_id + class_name theo COCO-80 taxonomy, mỗi instance một record | Object bị cắt mép ảnh (person ở `x=0.08` trong traffic); box không tight (oven bbox rộng gần toàn khung ngang); score thấp gần threshold (bowl 0.38) có thể false positive | Vẽ tight-fit bounding box bao sát vật thể, không bao nền; ghi rõ nếu object bị cắt mép hay che khuất >50% | Kiểm tra box không quá lỏng/quá chặt; xác nhận class đúng; review tỉ lệ precision/recall so baseline; flag duplicate hay missed object |
| Instance segmentation | Polygon pixel (danh sách tọa độ xy) + instance_id + class_id + bbox theo COCO-80 taxonomy | Polygon 120 điểm cho bus có thể over-segmented ở vùng biên thẳng; traffic-001 và traffic-002 (bus và car liền kề) tiềm ẩn nguy cơ biên chồng lấn; vùng che khuất không có trong ground truth | Vạch polygon bám sát contour thực; xử lý vùng chồng lấn theo quy tắc guideline (vật thể phía trước ưu tiên); tạo instance_id duy nhất cho mỗi đối tượng | So polygon với biên thực trong ảnh; kiểm tra instance_id không trùng; xác nhận class đúng; đo IoU giữa annotation và model prediction nếu có; flag vùng mờ/che khuất phức tạp |

---

## 5. An toàn dữ liệu

**Một quy tắc bảo vệ dữ liệu:**
> Không sao chép, lưu trữ, hay chia sẻ ảnh gốc hoặc dữ liệu annotation ra ngoài môi trường làm việc được phê duyệt. Mọi output (JSON, PNG, report) chỉ được commit lên repository dự án trong tổ chức; không upload lên dịch vụ bên thứ ba hay lưu trên máy cá nhân khi chưa được cho phép.

**Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**
Lead annotator / project manager của batch và bộ phận data governance / compliance của tổ chức. Không tiếp tục xử lý dữ liệu đó cho đến khi có hướng dẫn chính thức.

---

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
