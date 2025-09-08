<div align="center">

# Dự đoán tỷ lệ bỏ học của sinh viên dựa vào phân tích dữ liệu học tập
</div>

<div align="center">
<p align="center">
  <img width="1616" height="558" alt="image" src="https://github.com/user-attachments/assets/465191b9-6ed4-4fcb-b904-64c5fb4f095b" />
</p>
</div>
# Hệ thống Dự đoán Nguy cơ Bỏ học - Phiên bản Nâng cao

## Tính năng mới về biểu đồ và trích xuất dữ liệu

### 1. Tính năng biểu đồ nâng cao

#### 1.1 Biểu đồ tỷ lệ bỏ học theo lớp
- **Loại biểu đồ**: Cột (Bar chart)
- **Dữ liệu**: Tỷ lệ sinh viên bỏ học theo từng lớp
- **Tính năng**: Màu sắc động, tooltip chi tiết

#### 1.2 Biểu đồ tổng quan
- **Loại biểu đồ**: Tròn (Pie/Doughnut chart)
- **Dữ liệu**: Tỷ lệ tổng quan giữa sinh viên có nguy cơ và không có nguy cơ bỏ học

#### 1.3 Biểu đồ phân bố điểm
- **Loại biểu đồ**: Cột (Bar chart)
- **Dữ liệu**: Phân bố điểm trung bình theo các khoảng (0-2, 2-4, 4-6, 6-8, 8-10)

#### 1.4 Biểu đồ phân bố tín chỉ rớt
- **Loại biểu đồ**: Đường (Line chart)
- **Dữ liệu**: Phân bố số tín chỉ rớt của sinh viên

### 2. API endpoints mới

#### 2.1 `/chart_data` (POST)
Trả về dữ liệu cấu trúc cho các biểu đồ:
```json
{
  "class_data": [
    {
      "class": "Lớp A",
      "total_students": 50,
      "dropout_count": 5,
      "dropout_rate": 10.0
    }
  ],
  "overall_stats": {
    "total_students": 200,
    "dropout_count": 30,
    "dropout_rate": 15.0
  },
  "score_distribution": {
    "0-2": 5,
    "2-4": 10,
    "4-6": 50,
    "6-8": 100,
    "8-10": 35
  },
  "failed_credits_distribution": {
    "0": 150,
    "1": 25,
    "2": 15,
    "3": 10
  }
}
```

#### 2.2 `/export_chart_data` (POST)
Xuất toàn bộ dữ liệu và thống kê dưới dạng JSON:
```json
{
  "metadata": {
    "export_date": "2024-01-15 10:30:00",
    "total_students": 200
  },
  "students": [...],
  "summary": {
    "total_students": 200,
    "dropout_count": 30,
    "retention_count": 170
  }
}
```

### 3. Cách sử dụng

#### 3.1 Upload file Excel
1. Chuẩn bị file Excel với các cột:
   - Mã sinh viên (MaSV/MaSv/MSSV/StudentID)
   - Họ tên (HoTen/Name/FullName)
   - Lớp (Lop/Class/ClassName)
   - Điểm trung bình (DiemTB/diem_tb/AverageScore)
   - Số tín chỉ rớt (TinChiRot/tin_chi_rot/FailedCredits)
   - Số môn học lại (SoMonHocLai/so_mon_hoc_lai/FailedSubjects)

2. Upload file và hệ thống sẽ:
   - Dự đoán nguy cơ bỏ học cho từng sinh viên
   - Hiển thị bảng kết quả chi tiết
   - Tạo các biểu đồ thống kê
   - Cho phép xuất dữ liệu

#### 3.2 Xem biểu đồ
- Các biểu đồ tự động được tạo sau khi upload file
- Di chuột lên biểu đồ để xem chi tiết
- Các biểu đồ được cập nhật theo thời gian thực

#### 3.3 Xuất dữ liệu
- Nhấn nút "Xuất dữ liệu biểu đồ" để tải file JSON
- File chứa toàn bộ dữ liệu sinh viên và thống kê

### 4. Cấu trúc file mới

```
d:/ModelApp/
├── app.py (đã cập nhật với API mới)
├── templates/
│   ├── index.html (gốc)
│   ├── index_enhanced.html (phiên bản nâng cao)
│   └── student_detail.html
├── static/
│   ├── css/
│   │   ├── style.css (gốc)
│   │   └── enhanced-style.css (phiên bản nâng cao)
│   └── js/
│       ├── app.js (gốc)
│       └── enhanced-charts.js (chức năng biểu đồ mới)
└── README_ENHANCED.md (tài liệu này)
```

### 5. Cài đặt và chạy

1. Cài đặt dependencies:
```bash
pip install flask pandas numpy scikit-learn openpyxl
```

2. Chạy ứng dụng:
```bash
python app.py
```

3. Truy cập:
- Phiên bản gốc: http://localhost:5000
- Phiên bản nâng cao: http://localhost:5000 (sử dụng index_enhanced.html)

### 6. Lưu ý quan trọng

- Hệ thống tự động xử lý các tên cột khác nhau trong file Excel
- Nếu thiếu cột "Lop", hệ thống sẽ gán giá trị mặc định "Unknown Class"
- Các biểu đồ được tối ưu cho cả desktop và mobile
- Dữ liệu được xuất dưới dạng JSON chuẩn để dễ dàng phân tích thêm

### 7. Hỗ trợ kỹ thuật

Nếu gặp lỗi:
1. Kiểm tra format file Excel
2. Đảm bảo các cột dữ liệu có tên đúng chuẩn
3. Kiểm tra console log để debug
4. Liên hệ hỗ trợ nếu cần thiết
