#  DỰ ĐOÁN DOANH THU CỦA CHUỖI CỬA HÀNG BÁN LẺ 
Mã học phần: `CSE703020-2-3-25` | Nhóm: `9`

> Dự báo doanh số bán hàng cho hàng nghìn chuỗi thời gian (mỗi chuỗi là một cặp **cửa hàng × danh mục sản phẩm**) bằng cách kết hợp và so sánh 4 phương pháp: **TCN (Temporal Convolutional Network)**, **LSTM**, **ARIMA**, và **XGBoost**.

-----

## I. Thành viên nhóm

| Họ tên | MSSV |
| :--- | :--- |
| Phạm Thị Lương | `23017258` |
| Phạm Trọng Hoàng | `22010221` |
| Tô Thị Thùy Dương | `23010876` |

-----

## II. Mô tả bài toán

Dữ liệu gốc ở dạng bảng dài: mỗi dòng ứng với một ngày, một cửa hàng, một danh mục sản phẩm cụ thể. Với **54 cửa hàng** và **33 danh mục sản phẩm**, ta có tổng cộng **1.782 chuỗi thời gian** cần dự báo đồng thời.

Bài toán được đặt ra như sau:

> Với **120 ngày** dữ liệu doanh số đầu vào, dự báo **16 ngày** tiếp theo (forecast horizon) cho tất cả các chuỗi thời gian cùng lúc.

-----

## III. Dữ liệu

- Nguồn dữ liệu: [https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data]()
- File sử dụng: `train.csv`, `test.csv`

**Các trường dữ liệu chính:**

| Cột | Ý nghĩa |
|---|---|
| `date` | Ngày ghi nhận doanh số |
| `store_nbr` | Mã số cửa hàng |
| `family` | Danh mục/nhóm sản phẩm |
| `sales` | Doanh số bán ra (biến mục tiêu) |

-----

## IV. Quy trình xử lý dữ liệu

1. **Ghép khóa chuỗi thời gian**: Tạo cột `store_family` bằng cách nối `store_nbr` và `family` (ví dụ: `1_automotive`) để định danh duy nhất cho từng chuỗi.
2. **Pivot dữ liệu**: Chuyển từ dạng bảng dài sang bảng rộng — `date` làm index, mỗi `store_family` là một cột, giá trị là `sales`. Kết quả: ma trận có **1.782 cột** (kênh thời gian song song).
3. **Chuẩn hóa**: Dùng `StandardScaler` để chuẩn hóa dữ liệu trước khi đưa vào mạng nơ-ron.
4. **Tạo cửa sổ trượt**: Với mỗi vị trí trong chuỗi, lấy 120 ngày làm input (`X`) và 16 ngày tiếp theo làm target (`y`).
5. **Chia tập train/test**: 80% train — 20% test theo thời gian (không xáo trộn để tránh rò rỉ dữ liệu tương lai).

-----

## V. Các mô hình

| Mô hình | Cách tiếp cận | Ghi chú |
|---|---|---|
| **TCN** | Dự báo đồng thời toàn bộ 1.782 chuỗi | Conv1D nhiều lớp với dilation tăng dần (1, 2, 4, ...) để mở rộng receptive field theo thời gian mà không cần tăng kernel size |
| **LSTM** | Dự báo đồng thời toàn bộ 1.782 chuỗi | Cùng dữ liệu train/test với TCN để so sánh công bằng |
| **ARIMA** | Huấn luyện trên chuỗi **tổng doanh số toàn hệ thống** | Order `(5, 1, 0)`, do huấn luyện riêng cho 1.782 chuỗi là không khả thi về thời gian |
| **XGBoost** | Huấn luyện trên chuỗi **tổng doanh số toàn hệ thống** | Dự báo đa bước bằng lag features (cùng logic tạo `X, y` với TCN/LSTM) |

> ARIMA và XGBoost được đánh giá trên **tổng doanh số** (tổng tất cả `store_family` theo ngày) để đảm bảo phép so sánh công bằng, dễ diễn giải với TCN/LSTM — vốn dự báo trực tiếp ở mức chi tiết từng chuỗi rồi được cộng dồn lại.

-----

## VI. Đánh giá mô hình

Các mô hình được so sánh trên cùng một cửa sổ dự báo (120 ngày cuối tập train → 16 ngày đầu tập test) bằng 3 chỉ số:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

Kết quả cuối cùng được trực quan hóa qua:
- Biểu đồ cột so sánh RMSE & MAE giữa 4 mô hình
- Biểu đồ đường so sánh giá trị dự báo 16 ngày tới với giá trị thực tế

-----

## VII. Cấu trúc dự án

```
Group9_MachineLearning_Project/
├── .git/
├── README.md
├── notebooks/
│   └── salesPrediction.ipynb
├── data/                      # chỉ cần khi chạy local (train.csv, test.csv)
└── requirements/
    └── requirements.txt
```

-----

## VIII. Yêu cầu môi trường

```bash
pip install pandas numpy matplotlib scikit-learn torch xgboost statsmodels
```

**Bắt buộc chạy trên môi trường có GPU (CUDA)** — notebook gọi trực tiếp `.cuda()` ở nhiều bước (chuyển tensor, khởi tạo mô hình TCN/LSTM), không có cơ chế tự chuyển sang CPU. Khuyến nghị dùng **Google Colab** với GPU T4 (miễn phí). Nếu chạy local, máy cần có GPU NVIDIA hỗ trợ CUDA và đã cài PyTorch bản GPU tương ứng; nếu không, cần tự sửa các dòng `.cuda()` trong code sang `.to(device)` với `device = "cuda" if torch.cuda.is_available() else "cpu"`.

-----

## IX. Cách chạy

1. Clone repo và cài đặt các thư viện cần thiết (mục VIII).
2. **Chuẩn bị dữ liệu:**
   - Notebook được thiết kế để chạy trên **Google Colab**. Tải bộ dữ liệu theo link ở mục [III. Dữ liệu](#iii-dữ-liệu), nén thành `data.zip`, tải lên Google Drive cá nhân, sau đó chỉnh lại đường dẫn trong cell đầu notebook cho khớp với thư mục Drive của bạn:
     ```python
     drive.mount('/content/drive')
     !unzip "/content/drive/MyDrive/<đường-dẫn-của-bạn>/data.zip" -d "/content/data"
     ```
   - Nếu chạy local: tải và giải nén `train.csv`, `test.csv` vào thư mục `data/` ở gốc project, rồi bỏ 2 dòng `drive.mount(...)` và `!unzip ...` trong notebook — `pd.read_csv('data/train.csv')` sẽ tự đọc đúng.
3. Mở `salesPrediction.ipynb` và chạy tuần tự các cell theo thứ tự:
   - Khám phá dữ liệu
   - Tiền xử lý
   - Xây dựng & huấn luyện mô hình
   - Đánh giá mô hình
   - Huấn luyện trên toàn bộ dữ liệu & dự báo
   - So sánh với LSTM, ARIMA, XGBoost
     
-----
## X. Kết quả

Kết quả đánh giá các mô hình trên tập kiểm thử được trình bày trong bảng dưới đây.

| Mô hình | RMSE | MAE | MAPE (%) |
|---|---:|---:|---:|
| **XGBoost** | 65,175.404 | 56,742.212 | 8.733 |
| **LSTM** | 80,943.953 | 58,984.417 | 9.006 |
| **TCN** | 98,391.050 | 82,559.532 | 11.939 |
| **ARIMA** | 147,174.483 | 131,791.763 | 20.291 |

**Nhận xét:**

- **XGBoost** đạt kết quả tốt nhất với **RMSE = 65,175.404**, **MAE = 56,742.212** và **MAPE = 8.733%**, cho thấy khả năng dự báo tổng doanh số chính xác nhất trong thực nghiệm.
- **LSTM** là mô hình học sâu có hiệu năng tốt nhất, với sai số thấp hơn **TCN** và cao hơn **XGBoost**.
- **TCN** vẫn dự báo được xu hướng doanh số nhưng sai số lớn hơn LSTM trên bộ dữ liệu này.
- **ARIMA** có sai số cao nhất, phản ánh hạn chế của mô hình thống kê truyền thống khi áp dụng cho bài toán dự báo doanh số bán lẻ có tính biến động cao.

> **Lưu ý:** Trong nghiên cứu này, **TCN** và **LSTM** được huấn luyện trên dữ liệu đa chuỗi gồm **1.782 chuỗi thời gian** (54 cửa hàng × 33 nhóm sản phẩm), trong khi **ARIMA** và **XGBoost** được huấn luyện trên chuỗi **tổng doanh số toàn hệ thống (total_sales)**. Do đó, kết quả so sánh chủ yếu nhằm đánh giá hiệu quả của các hướng tiếp cận khác nhau đối với bài toán dự báo chuỗi thời gian, thay vì so sánh trực tiếp trên cùng một cấu trúc dữ liệu.

