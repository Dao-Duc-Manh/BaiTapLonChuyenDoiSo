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

def load_and_train_model():
    """Load data and train model with simplified columns"""
    global model, scaler
    
    try:
        # Read Excel file
        df = pd.read_excel('dulieu1.xlsx')
        
        # Handle different column names for required columns only
        column_mapping = {
            'DiemTB': ['DiemTB', 'diem_tb', 'AverageScore', 'Score', 'Điểm TB', 'Điểm trung bình'],
            'TinChiRot': ['TinChiRot', 'tin_chi_rot', 'FailedCredits', 'Tín chỉ rớt', 'SoTinChiRot', 'Số tín chỉ rớt'],
            'SoMonHocLai': ['SoMonHocLai', 'so_mon_hoc_lai', 'FailedSubjects', 'Số môn học lại', 'MonHocLai', 'Số môn rớt'],
            'BoHoc': ['BoHoc', 'bo_hoc', 'Dropout', 'Bỏ học'],
            'MaSV': ['MaSV', 'ma_sv', 'MSSV', 'StudentID', 'Mã sinh viên'],
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
        
        if 'MaSV' not in actual_columns:
            print("MaSV column not found, using default values...")
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV'
            
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
        print(f"Error: {e}")
        # Create sample data with required columns
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
        
        # Define column mappings for required columns
        column_mappings = {
            'DiemTB': ['DiemTB', 'diem_tb', 'AverageScore', 'Score', 'Điểm TB', 'Điểm trung bình'],
            'TinChiRot': ['TinChiRot', 'tin_chi_rot', 'FailedCredits', 'Tín chỉ rớt', 'SoTinChiRot', 'Số tín chỉ rớt'],
            'SoMonHocLai': ['SoMonHocLai', 'so_mon_hoc_lai', 'FailedSubjects', 'Số môn học lại', 'MonHocLai'],
            'MaSV': ['MaSv', 'MaSV', 'ma_sv', 'MSSV', 'StudentID', 'Mã sinh viên'],  # Thêm 'MaSv' vào đầu danh sách
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
        
        if 'MaSV' not in actual_columns:
            print("MaSV column not found, using default values...")
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV'
            
        if 'HoTen' not in actual_columns:
            print("HoTen column not found, using default values...")
            df['HoTen'] = [f'Sinh viên {i+1}' for i in range(len(df))]
            actual_columns['HoTen'] = 'HoTen'
        
        # Đảm bảo đọc đúng mã sinh viên
        if 'MaSV' in actual_columns:
            # Chuyển đổi cột MaSV thành kiểu string và loại bỏ khoảng trắng
            df[actual_columns['MaSV']] = df[actual_columns['MaSV']].astype(str).str.strip()
        else:
            print("MaSV column not found, using default values...")
            df['MaSV'] = [f'SV{i+1:03d}' for i in range(len(df))]
            actual_columns['MaSV'] = 'MaSV'
        
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
        
        # Prepare results với mã sinh viên được xử lý đúng
        results = []
        for index, row in df.iterrows():
            result = {
                'stt': int(index + 1),
                'masv': str(row[actual_columns['MaSV']]).strip(),  # Đảm bảo loại bỏ khoảng trắng
                'hoten': str(row[actual_columns['HoTen']]),
                'DiemTB': float(row[actual_columns['DiemTB']]),
                'prediction': int(predictions[index]),
                'dropout_probability': float(probabilities[index][1] * 100)
            }
            results.append(result)

        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error details:", str(e))  # Thêm log chi tiết
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

if __name__ == '__main__':
    load_and_train_model()
    app.run(debug=True)
