# Predicting Credit Card Fraud với PyTorch

Project huấn luyện MLP + Embedding để ước tính xác suất gian lận cho từng giao dịch trong cuộc thi [Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11).

## Cấu trúc project

```text
credit-card-fraud-mlp/
├── dataset/playground-series-s6e11/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
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

Mở PowerShell với Git và Python 3.11 trở lên. Repo là private, nên tài khoản GitHub dùng để clone cần có quyền truy cập. Với bản PyTorch hỗ trợ CUDA, project tự chọn GPU khi có thể.

```powershell
git clone https://github.com/nhathung2006/credit-card-fraud-mlp.git
Set-Location -LiteralPath 'credit-card-fraud-mlp'
python -m pip install -r requirements.txt
python -m fraud_mlp.train
python -m fraud_mlp.evaluate
python -m fraud_mlp.predict
```

Lệnh `git clone` tải kèm **train.csv**, **test.csv** và **sample_submission.csv** trong `dataset/playground-series-s6e11/`. Ba lệnh trên tự đọc bộ dữ liệu này theo cấu hình mặc định.

Project chia 699.960 dòng có nhãn trong `train.csv` thành **80% train (559.968 dòng)** và **20% validation (139.992 dòng)**, phân tầng theo `is_fraud` với `seed = 42`. Tệp `test.csv` của Kaggle dùng để tạo dự đoán cho bài nộp.

- Lệnh train lưu **artifacts/best-80-20.pt** gồm trọng số, thông tin tiền xử lý và cách chia dữ liệu của mô hình tốt nhất theo ROC-AUC validation.
- Lệnh evaluate tự chọn tập đánh giá theo checkpoint. Với cấu hình 80/20, lệnh in Score (ROC-AUC) và Recall trên validation. Recall dùng ngưỡng mặc định 0,5.
- Lệnh predict đọc checkpoint đó và ghi **artifacts/submission.csv** với hai cột **id,is_fraud**, theo thứ tự ID của tệp mẫu.
- Có thể truyền **--device cpu** để chạy CPU. Dùng **--help** để xem các tùy chọn, như **--epochs**, **--batch-size** và **--checkpoint**.

## Thay đổi cấu hình

Ba lệnh tự đọc `config.toml` ở thư mục gốc. Bạn có thể chỉnh trực tiếp các nhóm sau.

| Nhóm | Thiết lập |
|---|---|
| `[paths]` | Thư mục dữ liệu, checkpoint và tệp CSV đầu ra |
| `[runtime]` | Thiết bị và batch size |
| `[training]` | Số epoch, learning rate, seed, tỷ lệ validation/test nội bộ và số epoch chờ dừng sớm |
| `[model]` | Số chiều embedding, kích thước các lớp ẩn và dropout |

Đường dẫn tương đối trong TOML tính từ thư mục chứa file TOML. Bạn có thể dùng `--config` để chọn một file cho thí nghiệm riêng. Các tham số dòng lệnh như `--device` và `--epochs` ghi đè giá trị tương ứng trong file cho lần chạy đó.

Nếu bạn dùng dữ liệu ở thư mục khác, truyền `--data-dir 'đường_dẫn_thư_mục'` cho các lệnh train, evaluate và predict.

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

Chỉ phần train được dùng để cập nhật trọng số, tính thống kê chuẩn hóa và xây từ điển phân loại. Validation được dùng để chọn checkpoint, dừng sớm và tính các chỉ số trong lệnh evaluate. Giá trị số thiếu được thay bằng trung bình train. Các giá trị phân loại chưa gặp được mã hóa bằng chỉ số 0.

Tỷ lệ `validation_fraction` và `test_fraction` tính trên toàn bộ `train.csv`. Cấu hình mặc định dùng `validation_fraction = 0.2` và `test_fraction = 0.0`. Khi thay đổi tỷ lệ hoặc seed, huấn luyện lại để tạo checkpoint phù hợp.

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

## Phần mở rộng thực nghiệm (Minh Nghĩa)

- `BAO_CAO_MINH_NGHIA.docx`: báo cáo phương pháp, bảng kết quả đối chứng 80/20 và 5-Fold.
- `experiments/exp01..09_*.toml` và `experiments/run_experiments.py`: các cấu hình cải tiến MLP và Residual MLP trên tập chia khóa 80/20 (giữ nguyên `config.toml` gốc).
- `experiments/feature_pipeline.py`, `run_benchmark_80_20.py`, `run_champion_80_20.py`: đối chứng các nhánh đặc trưng và mô hình cây/mạng trên 80/20.
- `experiments/run_final_5fold.py`, `run_v4_glm_trees_5fold.py`, `run_full_refit_trees.py`, `build_final_champion.py`: quy trình kiểm chứng chéo 5-Fold, huấn luyện lại trên 100% tập train và tối ưu trọng số tổ hợp.
- `experiments/sanity_check_leak.py`, `sanity_check_diverse.py`: kiểm tra rò rỉ nhãn (Label-Shuffle).
- `kaggle/*.ipynb`: 3 notebook huấn luyện 5-Fold trên GPU Kaggle (`TabM + DAE`, `ModernNCA + FT-Transformer`, `RealMLP`).

Lệnh chạy các cấu hình MLP cải tiến và kiểm tra rò rỉ:

```powershell
python experiments/run_experiments.py experiments/exp07_resmlp_ple_tierA.toml
python experiments/sanity_check_leak.py
```
