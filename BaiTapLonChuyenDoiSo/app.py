from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Global variables
model = None
scaler = None

def load_and_train_model(filename):
    """Load data and train model with simplified columns"""
    global model, scaler
    
    try:
        # Read Excel file
        df = pd.read_excel(filename)
        
        # Handle different column names for required columns only
        column_mapping = {
            'DiemTB': ['DiemTB', 'diem_tb', 'AverageScore', 'Score', 'Điểm TB', 'Điểm trung bình'],
            'TinChiRot': ['TinChiRot', 'tin_chi_rot', 'FailedCredits', 'Tín chỉ rớt', 'SoTinChiRot', 'Số tín chỉ rớt'],
            'SoMonHocLai': ['SoMonHocLai', 'so_mon_hoc_lai', 'FailedSubjects', 'Số môn học lại', 'MonHocLai', 'Số môn rớt'],
            'BoHoc': ['BoHoc', 'bo_hoc', 'Dropout', 'Bỏ học'],
            'MaSV': ['MaSv', 'MaSV', 'ma_sv', 'MSSV', 'StudentID', 'Mã sinh viên'], # Đã ưu tiên 'MaSv'
            'HoTen': ['HoTen', 'ho_ten', 'Name', 'Họ tên', 'FullName', 'Họ và tên']
        }
        
        # Find actual column names
        actual_columns = {}
        available_columns = list(df.columns)
        
        for key, possible_names in column_mapping.items():
            for name in possible_names:
                if name in available_columns:
                    actual_columns[key] = name
                    break
        
        # Handle missing columns with default values
        if 'SoMonHocLai' not in actual_columns:
            print("SoMonHocLai column not found, using default values...")
            df['SoMonHocLai'] = 0
            actual_columns['SoMonHocLai'] = 'SoMonHocLai'
        
        # Đảm bảo cột MaSV được xử lý đúng cách, ưu tiên 'MaSv'
        if 'MaSV' not in actual_columns: # Nếu không tìm thấy bất kỳ tên nào trong mapping
            print("MaSV/MaSv column not found, using default values...")
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV' # Gán tên cột mặc định nếu không tìm thấy
        
        if 'HoTen' not in actual_columns:
            print("HoTen column not found, using default values...")
            df['HoTen'] = [f'Sinh viên {i+1}' for i in range(len(df))]
            actual_columns['HoTen'] = 'HoTen'
        
        # Prepare data using only required columns
        X = df[[actual_columns['DiemTB'], actual_columns['TinChiRot'], actual_columns['SoMonHocLai']]].values
        y = df[actual_columns['BoHoc']].values
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return True
        
    except Exception as e:
        print(f"Error during model loading/training: {e}")
        # Create sample data with required columns if dulieu1.xlsx fails
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame({
            'DiemTB': np.random.normal(7, 1.5, n_samples),
            'TinChiRot': np.random.randint(0, 10, n_samples),
            'SoMonHocLai': np.random.randint(0, 5, n_samples),
            'BoHoc': np.random.randint(0, 2, n_samples),
            'MaSV': [f'SV{i+1:03d}' for i in range(n_samples)],
            'HoTen': [f'Sinh viên {i+1}' for i in range(n_samples)]
        })
        
        X = df[['DiemTB', 'TinChiRot', 'SoMonHocLai']].values
        y = df['BoHoc'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction with simplified columns"""
    try:
        data = request.json
        
        diem_tb = float(data.get('diem_tb', 0))
        tin_chi_rot = int(data.get('tin_chi_rot', 0))
        so_mon_hoc_lai = int(data.get('so_mon_hoc_lai', 0))
        
        features = np.array([[diem_tb, tin_chi_rot, so_mon_hoc_lai]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        dropout_prob = probability[1] * 100
        
        return jsonify({
            'prediction': int(prediction),
            'dropout_probability': round(dropout_prob, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """Batch prediction from uploaded file with simplified columns"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    try:
        df = pd.read_excel(file)
        print("Columns in uploaded file:", df.columns.tolist())
        
        # Define column mappings for required columns
        column_mappings = {
            'DiemTB': ['DiemTB', 'diem_tb', 'AverageScore', 'Score', 'Điểm TB', 'Điểm trung bình'],
            'TinChiRot': ['TinChiRot', 'tin_chi_rot', 'FailedCredits', 'Tín chỉ rớt', 'SoTinChiRot', 'Số tín chỉ rớt'],
            'SoMonHocLai': ['SoMonHocLai', 'so_mon_hoc_lai', 'FailedSubjects', 'Số môn học lại', 'MonHocLai'],
            'MaSV': ['MaSv', 'MaSV', 'ma_sv', 'MSSV', 'StudentID', 'Mã sinh viên'], # Đã ưu tiên 'MaSv'
            'HoTen': ['HoTen', 'ho_ten', 'Name', 'Họ tên', 'FullName', 'Họ và tên'],
            'Lop': ['Lop', 'lop', 'Class', 'Lớp', 'ClassName']
        }
        
        # Find actual columns
        actual_columns = {}
        available_columns = list(df.columns)
        
        for key, possible_names in column_mappings.items():
            for name in possible_names:
                if name in available_columns:
                    actual_columns[key] = name
                    break
        
        # Handle missing columns with default values
        if 'SoMonHocLai' not in actual_columns:
            print("SoMonHocLai column not found, using default values...")
            df['SoMonHocLai'] = 0
            actual_columns['SoMonHocLai'] = 'SoMonHocLai'
        
        # Đảm bảo cột MaSV được xử lý đúng cách, ưu tiên 'MaSv'
        if 'MaSV' not in actual_columns: # Nếu không tìm thấy bất kỳ tên nào trong mapping
            print("MaSV/MaSv column not found, using default values...")
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV' # Gán tên cột mặc định nếu không tìm thấy
            
        if 'HoTen' not in actual_columns:
            print("HoTen column not found, using default values...")
            df['HoTen'] = [f'Sinh viên {i+1}' for i in range(len(df))]
            actual_columns['HoTen'] = 'HoTen'

        # Thêm xử lý cho cột 'Lop' nếu nó không có trong file, để tránh lỗi khi truy cập
        if 'Lop' not in actual_columns:
            print("Lop column not found in uploaded file, using default value 'Unknown Class'.")
            df['Lop'] = 'Unknown Class'
            actual_columns['Lop'] = 'Lop'
        
        # Đảm bảo đọc đúng mã sinh viên và loại bỏ khoảng trắng
        # Sử dụng actual_columns['MaSV'] để truy cập tên cột đã được xác định
        df[actual_columns['MaSV']] = df[actual_columns['MaSV']].astype(str).str.strip()
        
        # Ensure all values are numeric for prediction features
        df[actual_columns['DiemTB']] = pd.to_numeric(df[actual_columns['DiemTB']], errors='coerce').fillna(0)
        df[actual_columns['TinChiRot']] = pd.to_numeric(df[actual_columns['TinChiRot']], errors='coerce').fillna(0)
        df[actual_columns['SoMonHocLai']] = pd.to_numeric(df[actual_columns['SoMonHocLai']], errors='coerce').fillna(0)
        
        # Prepare features using only required columns
        features = df[[actual_columns['DiemTB'], actual_columns['TinChiRot'], actual_columns['SoMonHocLai']]].values
        features_scaled = scaler.transform(features)
        
        # Predict
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Prepare results
        results = []
        for index, row in df.iterrows():
            result = {
                'stt': int(index + 1),
                'masv': str(row[actual_columns['MaSV']]).strip(),
                'hoten': str(row[actual_columns['HoTen']]),
                'lop': str(row[actual_columns['Lop']]),
                'DiemTB': float(row[actual_columns['DiemTB']]),
                'tin_chi_rot': int(row[actual_columns['TinChiRot']]),
                'so_mon_hoc_lai': int(row[actual_columns['SoMonHocLai']]),
                'prediction': int(predictions[index]),
                'dropout_probability': float(probabilities[index][1] * 100)
            }
            results.append(result)

        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error details:", str(e))
        return jsonify({'error': f"Lỗi xử lý file: {str(e)}"})

# Helper function để chuyển đổi NumPy types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.route('/chart_data', methods=['POST'])
def chart_data():
    """Provide structured data for various charts"""
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'Không có dữ liệu'}), 400
        
        # Chart 1: Dropout rate by class
        class_data = {}
        for student in results:
            lop = student.get('lop', 'Unknown')
            if lop not in class_data:
                class_data[lop] = {'total': 0, 'dropout': 0}
            class_data[lop]['total'] += 1
            if student.get('prediction', 0) == 1:
                class_data[lop]['dropout'] += 1
        
        class_chart_data = []
        for lop, data in class_data.items():
            dropout_rate = (data['dropout'] / data['total'] * 100) if data['total'] > 0 else 0
            class_chart_data.append({
                'class': lop,
                'total_students': data['total'],
                'dropout_count': data['dropout'],
                'dropout_rate': round(dropout_rate, 2)
            })
        
        # Chart 2: Overall statistics
        total_students = len(results)
        dropout_count = sum(1 for s in results if s.get('prediction', 0) == 1)
        dropout_rate = (dropout_count / total_students * 100) if total_students > 0 else 0
        
        # Chart 3: Score distribution
        score_ranges = {
            '0-2': 0, '2-4': 0, '4-6': 0, '6-8': 0, '8-10': 0
        }
        for student in results:
            score = student.get('DiemTB', 0)
            if score <= 2:
                score_ranges['0-2'] += 1
            elif score <= 4:
                score_ranges['2-4'] += 1
            elif score <= 6:
                score_ranges['4-6'] += 1
            elif score <= 8:
                score_ranges['6-8'] += 1
            else:
                score_ranges['8-10'] += 1
        
        # Chart 4: Failed credits distribution
        failed_credits = {}
        for student in results:
            credits = student.get('tin_chi_rot', 0)
            if credits not in failed_credits:
                failed_credits[credits] = 0
            failed_credits[credits] += 1
        
        return jsonify({
            'class_data': class_chart_data,
            'overall_stats': {
                'total_students': total_students,
                'dropout_count': dropout_count,
                'dropout_rate': round(dropout_rate, 2)
            },
            'score_distribution': score_ranges,
            'failed_credits_distribution': failed_credits
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export_chart_data', methods=['POST'])
def export_chart_data():
    """Export chart data as JSON for download"""
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'Không có dữ liệu'}), 400
        
        # Prepare export data
        export_data = {
            'metadata': {
                'export_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_students': len(results)
            },
            'students': results,
            'summary': {
                'total_students': len(results),
                'dropout_count': sum(1 for s in results if s.get('prediction', 0) == 1),
                'retention_count': sum(1 for s in results if s.get('prediction', 0) == 0)
            }
        }
        
        return jsonify(export_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_excel', methods=['POST'])
def download_excel():
    """Download prediction results as Excel file"""
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'Không có dữ liệu để xuất'}), 400
        
        # Create DataFrame with required columns
        df_export = pd.DataFrame()
        
        # Map the data to required columns
        export_data = []
        for idx, student in enumerate(results, 1):
            export_data.append({
                'STT': idx,
                'Họ tên': student.get('hoten', ''),
                'Mã SV': student.get('masv', ''),
                'Lớp': student.get('lop', ''),
                'Khoa': student.get('khoa', 'Không xác định'),
                'Nguy cơ': 'Có nguy cơ' if student.get('prediction', 0) == 1 else 'Không có nguy cơ',
                'Tỷ lệ bỏ học': f"{student.get('dropout_probability', 0):.2f}%"
            })
        
        df_export = pd.DataFrame(export_data)
        
        # Create Excel file in memory
        from io import BytesIO
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Kết quả dự đoán', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Kết quả dự đoán']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        
        # Return file as attachment
        from flask import send_file
        
        return send_file(
            BytesIO(output.getvalue()),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'ket_qua_du_doan_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/student_detail')
def student_detail():
    """Renders the student detail page."""
    return render_template('student_detail.html')
@app.route('/student_detail/<masv>')
def student_detail_with_id(masv):
    """Renders the student detail page with specific student data."""
    return render_template('student_detail.html', masv=masv)
@app.route('/api/student/<masv>')
def get_student_detail(masv):
    """API endpoint to get detailed student information."""
    try:
        # Read the Excel file
        df = pd.read_excel('dulieu1.xlsx')
        
        # Define column mappings
        column_mappings = {
            'MaSV': ['MaSv', 'MaSV', 'ma_sv', 'MSSV', 'StudentID', 'Mã sinh viên'],
            'HoTen': ['HoTen', 'ho_ten', 'Name', 'Họ tên', 'FullName', 'Họ và tên'],
            'Lop': ['Lop', 'lop', 'Class', 'Lớp', 'ClassName'],
            'Khoa': ['Khoa', 'khoa', 'Faculty', 'Khoa', 'Department'],
            'DiemTB': ['DiemTB', 'diem_tb', 'AverageScore', 'Score', 'Điểm TB', 'Điểm trung bình'],
            'TinChiRot': ['TinChiRot', 'tin_chi_rot', 'FailedCredits', 'Tín chỉ rớt', 'SoTinChiRot', 'Số tín chỉ rớt'],
            'SoMonHocLai': ['SoMonHocLai', 'so_mon_hoc_lai', 'FailedSubjects', 'Số môn học lại', 'MonHocLai'],
            'BoHoc': ['BoHoc', 'bo_hoc', 'Dropout', 'Bỏ học']
        }
        
        # Find actual columns
        actual_columns = {}
        available_columns = list(df.columns)
        
        for key, possible_names in column_mappings.items():
            for name in possible_names:
                if name in available_columns:
                    actual_columns[key] = name
                    break
        
        # Handle missing columns
        if 'MaSV' not in actual_columns:
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV'
        
        # Find student by MaSV
        student_row = df[df[actual_columns['MaSV']].astype(str).str.strip() == masv.strip()]
        
        if student_row.empty:
            return jsonify({'error': 'Không tìm thấy sinh viên'}), 404
        
        # Get student data
        student = student_row.iloc[0]
        
        # Prepare features for prediction
        features = np.array([[
            float(student.get(actual_columns['DiemTB'], 0)),
            int(student.get(actual_columns['TinChiRot'], 0)),
            int(student.get(actual_columns['SoMonHocLai'], 0))
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        dropout_prob = probability[1] * 100
        
        # Prepare student detail
        student_detail = {
            'masv': str(student.get(actual_columns['MaSV'], '')).strip(),
            'hoten': str(student.get(actual_columns['HoTen'], '')),
            'lop': str(student.get(actual_columns['Lop'], 'Không xác định')),
            'khoa': str(student.get(actual_columns['Khoa'], 'Không xác định')),
            'diem_tb': float(student.get(actual_columns['DiemTB'], 0)),
            'tin_chi_rot': int(student.get(actual_columns['TinChiRot'], 0)),
            'so_mon_hoc_lai': int(student.get(actual_columns['SoMonHocLai'], 0)),
            'prediction': int(prediction),
            'dropout_probability': float(dropout_prob),
            'ty_le_bo_hoc_so': f"{dropout_prob:.2f}%",
            'ty_le_bo_hoc_chu': "Có nguy cơ bỏ học" if prediction == 1 else "Không có nguy cơ bỏ học"
        }
        
        # Add all other columns from Excel
        for col in df.columns:
            if col not in actual_columns.values():
                value = student.get(col)
                if pd.notna(value):
                    student_detail[col] = str(value)
        
        return jsonify(student_detail)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_and_train_model('dulieu1.xlsx')  # Specify the filename here
    app.run(debug=True)
    
