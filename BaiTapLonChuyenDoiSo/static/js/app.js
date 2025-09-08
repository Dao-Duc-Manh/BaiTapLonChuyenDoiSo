document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const uploadResultDiv = document.getElementById('uploadResult');
    const statisticsElement = document.getElementById('statistics');
    const resultsTableDiv = document.getElementById('resultsTable');
    const chartSection = document.getElementById('chartSection');
    let allResultsData = [];
    let currentDisplayLimit = 10;
    const increment = 5;
    let dropoutChart = null;

    // Single prediction
    document.getElementById('singlePredictForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = {
            diem_tb: parseFloat(document.getElementById('diem_tb').value),
            tin_chi_rot: parseInt(document.getElementById('tin_chi_rot').value),
            so_mon_hoc_lai: parseInt(document.getElementById('so_mon_hoc_lai').value)
        };
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.error) {
                document.getElementById('singleResult').innerHTML = `<p class="error">Lỗi: ${result.error}</p>`;
            } else {
                document.getElementById('singleResult').innerHTML = `
                    <div class="alert alert-info">
                        <h6>Kết quả dự đoán</h6>
                        <p><strong>Dự đoán:</strong> ${result.prediction === 1 ? 'Có nguy cơ bỏ học' : 'Không có nguy cơ bỏ học'}</p>
                        <p><strong>Xác suất bỏ học:</strong> ${result.dropout_probability}%</p>
                    </div>
                `;
            }
        } catch (error) {
            document.getElementById('singleResult').innerHTML = `<p class="error">Lỗi kết nối: ${error.message}</p>`;
        }
    });

    // Batch prediction
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            const response = await fetch('/upload_predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                uploadResultDiv.innerHTML = `<p class="error">Lỗi: ${result.error}</p>`;
            } else {
                displayResults(result.results);
            }
        } catch (error) {
            uploadResultDiv.innerHTML = `<p class="error">Lỗi kết nối: ${error.message}</p>`;
        }
    });

    // Helper function to display results
    function displayResults(results) {
        allResultsData = results;
        currentDisplayLimit = 10;
        
        // Calculate and update statistics
        const totalStudents = results.length;
        const averageScore = results.reduce((sum, student) => sum + parseFloat(student.DiemTB || 0), 0) / totalStudents;
        const dropoutCount = results.filter(student => student.prediction === 1).length;
        const dropoutRate = ((totalStudents > 0) ? (dropoutCount / totalStudents) * 100 : 0).toFixed(2);

        // Show and update statistics section
        statisticsElement.style.display = 'block';
        document.getElementById('totalStudents').textContent = totalStudents;
        document.getElementById('averageScore').textContent = averageScore.toFixed(2);
        document.getElementById('dropoutRate').textContent = `${dropoutRate}%`;

        // Display results table and update the chart
        renderResults(results);
        chartSection.style.display = 'block';
        updateDropoutChart(results);
    }

    // Helper function to display results
    function renderResults(results) {
        let html = '<table class="table table-striped table-hover"><thead><tr><th>STT</th><th>Mã SV</th><th>Họ Tên</th><th>Lớp</th><th>Điểm TB</th><th>Dự đoán</th><th>Xác suất</th><th>Chi tiết</th></tr></thead><tbody>';

        const displayResults = results.slice(0, currentDisplayLimit);

        displayResults.forEach(result => {
            html += `
                <tr>
                    <td>${result.stt}</td>
                    <td>${result.masv}</td>
                    <td>${result.hoten}</td>
                    <td>${result.lop || 'N/A'}</td>
                    <td>${result.DiemTB.toFixed(2)}</td>
                    <td>
                        <span class="badge ${result.prediction === 1 ? 'bg-danger' : 'bg-success'}">
                            ${result.prediction === 1 ? 'Có nguy cơ' : 'Không có nguy cơ'}
                        </span>
                    </td>
                    <td>${result.dropout_probability.toFixed(2)}%</td>
                    <td>
                        <button class="btn btn-sm btn-primary" onclick="viewStudentDetail('${result.masv}')">
                            <i class="bi bi-eye"></i> Xem
                        </button>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table>';

        if (currentDisplayLimit > 10) {
            html += `<button onclick="showLessResults()" class="btn btn-secondary btn-sm me-2">Ẩn bớt</button>`;
        }
        if (results.length > currentDisplayLimit) {
            html += `<button onclick="showMoreResults()" class="btn btn-primary btn-sm">Xem thêm sinh viên</button>`;
        }

        resultsTableDiv.innerHTML = html;
    }

    // Helper function to update the chart
    function updateDropoutChart(results) {
        // Step 1: Group and count students by class
        const uniqueClasses = [...new Set(results.map(student => student.lop))].sort();
        
        // Step 2: Calculate dropout rate for each class
        const chartData = uniqueClasses.map(className => {
            const studentsInClass = results.filter(student => student.lop === className);
            const dropoutCount = studentsInClass.filter(student => student.prediction === 1).length;
            const dropoutRate = (studentsInClass.length > 0) ? (dropoutCount / studentsInClass.length * 100).toFixed(2) : 0;
            
            return {
                className,
                dropoutRate: parseFloat(dropoutRate)
            };
        });

        // Step 3: Destroy old chart if it exists
        if (dropoutChart) {
            dropoutChart.destroy();
        }

        // Step 4: Create a new chart
        const ctx = document.getElementById('dropoutChart').getContext('2d');
        dropoutChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.map(item => item.className),
                datasets: [{
                    label: 'Tỷ lệ sinh viên bỏ học (%)',
                    data: chartData.map(item => item.dropoutRate),
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Biểu Đồ Tỷ Lệ Sinh Viên Bỏ Học Theo Lớp',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        padding: 20
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Tỷ lệ bỏ học: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });

        console.log('Class-wise dropout statistics:', chartData);
    }

    // View student detail in modal
    window.viewStudentDetail = function(masv) {
        loadStudentDetail(masv);
    };

    // Load student detail content via AJAX
    async function loadStudentDetail(masv) {
        try {
            const response = await fetch(`/api/student/${encodeURIComponent(masv)}`);
            const student = await response.json();
            
            if (student.error) {
                throw new Error(student.error);
            }
            
            // Create student detail HTML
            const studentDetailHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Thông Tin Cơ Bản</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Mã SV:</strong></td><td>${student.masv || 'Không có'}</td></tr>
                            <tr><td><strong>Họ tên:</strong></td><td>${student.hoten || 'Không có'}</td></tr>
                            <tr><td><strong>Lớp:</strong></td><td>${student.lop || 'Không có'}</td></tr>
                            <tr><td><strong>Khoa:</strong></td><td>${student.khoa || 'Không có'}</td></tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h6>Kết Quả Dự Đoán</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Điểm TB:</strong></td><td>${student.diem_tb || 0}</td></tr>
                            <tr><td><strong>Tín chỉ rớt:</strong></td><td>${student.tin_chi_rot || 0}</td></tr>
                            <tr><td><strong>Môn học lại:</strong></td><td>${student.so_mon_hoc_lai || 0}</td></tr>
                            <tr><td><strong>Tỷ lệ bỏ học:</strong></td><td>${student.ty_le_bo_hoc_so || '0%'}</td></tr>
                            <tr><td><strong>Đánh giá:</strong></td><td class="${student.prediction === 1 ? 'text-danger fw-bold' : 'text-success fw-bold'}">${student.ty_le_bo_hoc_chu || 'Không xác định'}</td></tr>
                        </table>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Thông Tin Đầy Đủ</h6>
                        <div class="table-responsive">
                            <table class="table table-striped table-sm">
                                <thead>
                                    <tr><th>Trường dữ liệu</th><th>Giá trị</th></tr>
                                </thead>
                                <tbody>
                                    ${Object.keys(student)
                                        .filter(key => key !== 'prediction' && key !== 'dropout_probability')
                                        .sort()
                                        .map(key => {
                                            const value = student[key];
                                            if (value !== null && value !== undefined && value !== '') {
                                                return `<tr>
                                                    <td><strong>${key.replace(/_/g, ' ').toUpperCase()}</strong></td>
                                                    <td>${value}</td>
                                                </tr>`;
                                            }
                                            return '';
                                        })
                                        .join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('studentDetailContent').innerHTML = studentDetailHTML;
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('studentDetailModal'));
            modal.show();
            
        } catch (error) {
            console.error('Error loading student detail:', error);
            alert('Không thể tải thông tin sinh viên: ' + error.message);
        }
    }

    window.showMoreResults = () => {
        currentDisplayLimit += increment;
        renderResults(allResultsData);
    };

    window.showLessResults = () => {
        currentDisplayLimit = 10;
        renderResults(allResultsData);
    };

    // Download Excel functionality
    window.downloadExcel = async function() {
        if (!allResultsData || allResultsData.length === 0) {
            alert('Không có dữ liệu để xuất Excel!');
            return;
        }

        try {
            const response = await fetch('/download_excel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ results: allResultsData })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Lỗi khi tạo file Excel');
            }

            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ket_qua_du_doan_${new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_')}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (error) {
            console.error('Error downloading Excel:', error);
            alert('Lỗi khi tải xuống file Excel: ' + error.message);
        }
    };
});
