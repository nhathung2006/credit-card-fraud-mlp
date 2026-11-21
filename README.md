# Predicting Credit Card Fraud với PyTorch

Project huấn luyện MLP + Embedding để ước tính xác suất gian lận cho từng giao dịch trong cuộc thi [Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11).

## Cấu trúc project

```text
credit-card-fraud-mlp/
├── fraud_mlp/
│   ├── configuration.py  # Đọc TOML và áp dụng tham số dòng lệnh
│   ├── data.py        # 24 đặc trưng, chuẩn hóa và mã hóa
│   ├── model.py       # Kiến trúc MLP và bảng embedding
│   ├── train.py       # Huấn luyện và chọn checkpoint bằng ROC-AUC
│   ├── evaluate.py    # Đánh giá ROC-AUC và Recall trên validation
│   ├── predict.py     # Dự đoán và ghi CSV nộp Kaggle
│   └── runtime.py     # CPU/GPU và lưu tệp
├── tests/             # Kiểm tra tiền xử lý và luồng train → predict
├── config.toml        # Đường dẫn, thiết bị, huấn luyện và kiến trúc
├── requirements.txt
└── artifacts/         # Tự tạo khi huấn luyện hoặc dự đoán
```

## Chạy trên Windows

Mở PowerShell với Python 3.11 trở lên và chuyển vào thư mục project. Nếu môi trường Python chưa có các thư viện cần thiết, chạy lệnh cài đặt bên dưới. Với bản PyTorch hỗ trợ CUDA, project tự chọn GPU khi có thể.

```powershell
Set-Location -LiteralPath 'G:\UET_project\Ki_1_Nam_3\Machine_learning\credit-card-fraud-mlp'
python -m pip install -r requirements.txt
python -m fraud_mlp.train
python -m fraud_mlp.evaluate
python -m fraud_mlp.predict
```

Project tìm ba tệp **train.csv**, **test.csv** và **sample_submission.csv** trong thư mục **../dataset/playground-series-s6e11**. Nếu bạn đặt chúng ở nơi khác, truyền **--data-dir 'đường_dẫn_thư_mục'** cho cả hai lệnh.

- Lệnh train lưu **artifacts/best-80-20.pt** gồm trọng số, thông tin tiền xử lý và cách chia dữ liệu của mô hình tốt nhất theo ROC-AUC validation.
- Lệnh evaluate tự chọn tập đánh giá theo checkpoint. Với cấu hình 80/20, lệnh in Score (ROC-AUC) và Recall trên validation. Recall dùng ngưỡng mặc định 0,5.
- Lệnh predict đọc checkpoint đó và ghi **artifacts/submission.csv** với hai cột **id,is_fraud**, theo thứ tự ID của tệp mẫu.
- Có thể truyền **--device cpu** để chạy CPU. Dùng **--help** để xem các tùy chọn, như **--epochs**, **--batch-size** và **--checkpoint**.

## Thay đổi cấu hình

Hai lệnh tự đọc `config.toml` ở thư mục gốc. Bạn có thể chỉnh trực tiếp các nhóm sau.

| Nhóm | Thiết lập |
|---|---|
| `[paths]` | Thư mục dữ liệu, checkpoint và tệp CSV đầu ra |
| `[runtime]` | Thiết bị và batch size |
| `[training]` | Số epoch, learning rate, seed, tỷ lệ validation/test nội bộ và số epoch chờ dừng sớm |
| `[model]` | Số chiều embedding, kích thước các lớp ẩn và dropout |

Đường dẫn tương đối trong TOML tính từ thư mục chứa file TOML. Bạn có thể dùng `--config` để chọn một file cho thí nghiệm riêng. Các tham số dòng lệnh như `--device` và `--epochs` ghi đè giá trị tương ứng trong file cho lần chạy đó.

```powershell
python -m fraud_mlp.train --config experiment.toml --epochs 1
python -m fraud_mlp.predict --config experiment.toml
```

Tạo `experiment.toml` bằng cách sao chép `config.toml` rồi sửa giá trị cần thử. Khi dự đoán, kiến trúc và thông tin tiền xử lý được đọc từ checkpoint đã huấn luyện.

## Kiến trúc và huấn luyện mặc định

```text
13 cột số đã chuẩn hóa + 6 cờ nhị phân + 5 embedding × 4 = 39 chiều
39 → Linear(256) → Linear(128) → Linear(64) → Linear(1)
```

Các lớp ẩn dùng ReLU và Dropout(0.2). Lớp cuối tạo một logit. Lệnh predict dùng sigmoid để đổi logit thành xác suất. ID chỉ phục vụ việc ghép dự đoán theo mẫu nộp.

Mặc định, `train.csv` có nhãn được chia 80% train và 20% validation, phân tầng theo **is_fraud** và seed 42. Với 699.960 dòng, hai tập có lần lượt 559.968 và 139.992 dòng, với các ID tách biệt. Cấu hình dùng `validation_fraction = 0.2` và `test_fraction = 0.0`.

Chỉ phần train được dùng để cập nhật trọng số, tính thống kê chuẩn hóa và xây từ điển phân loại. Validation được dùng để chọn checkpoint, dừng sớm và tính các chỉ số trong lệnh evaluate. Giá trị số thiếu được thay bằng trung bình train. Các giá trị phân loại chưa gặp được mã hóa bằng chỉ số 0.

Tỷ lệ `validation_fraction` và `test_fraction` tính trên toàn bộ `train.csv`. Tệp `test.csv` của Kaggle là đầu vào cho lệnh predict để tạo bài nộp. Khi thay đổi tỷ lệ hoặc seed, huấn luyện lại để tạo checkpoint phù hợp.

Cấu hình mặc định gồm batch size 1024, AdamW với learning rate 0,001, BCEWithLogitsLoss, tối đa 30 epoch và dừng sớm sau 5 epoch liên tiếp khi ROC-AUC validation không tăng.

## Đánh giá Score (ROC-AUC) và Recall

```powershell
python -m fraud_mlp.evaluate
python -m fraud_mlp.evaluate --split validation
python -m fraud_mlp.evaluate --threshold 0.3
python -m fraud_mlp.predict --metrics
```

`evaluate` mặc định in Score (ROC-AUC) và Recall trên validation khi checkpoint dùng cách chia 80/20. `predict --metrics` in cùng kết quả rồi xuất dự đoán Kaggle. Score được tính từ các điểm dự đoán trước khi áp dụng ngưỡng. Recall dùng ngưỡng mặc định 0,5. Một giao dịch được xếp vào lớp gian lận nếu xác suất lớn hơn hoặc bằng ngưỡng.

Checkpoint mới lưu cách chia và mã SHA-256 của tệp huấn luyện để kiểm tra dữ liệu khi đánh giá. Tỷ lệ được lấy từ checkpoint. Để đánh giá theo tỷ lệ 80/20 mới, chạy train để tạo **best-80-20.pt**.

## Kiểm tra

```powershell
python -m unittest discover -s tests -v
```
