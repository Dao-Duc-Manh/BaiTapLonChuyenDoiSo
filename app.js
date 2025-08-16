document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadLoadingSpinner = document.getElementById('loadingSpinner');
    const resultsTableBody = document.getElementById('resultsTableBody');
    const totalStudents = document.getElementById('totalStudents');
    const dropoutRate = document.getElementById('dropoutRate');

    // Fetch dataset info on page load
    fetch('/data_info')
        .then(response => response.json())
        .then(data => {
            totalStudents.textContent = data.total_students;
            dropoutRate.textContent = (data.dropout_rate * 100).toFixed(2) + '%';
        })
        .catch(() => {
            totalStudents.textContent = '-';
            dropoutRate.textContent = '-';
        });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files.length) {
            alert('Vui lòng chọn file Excel để tải lên.');
            return;
        }

        uploadLoadingSpinner.classList.remove('d-none');
        resultsTableBody.innerHTML = '';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/upload_predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                if (result.length === 0) {
                    alert('File không có dữ liệu hợp lệ để dự đoán.');
                    return;
                }

                result.forEach((item, index) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${index + 1}</td>
                        <td>${item.hoten}</td>
                        <td>${item.lop}</td>
                        <td>${item.khoa}</td>
                        <td>${item.ty_le_bo_hoc_so}</td>
                    `;
                    resultsTableBody.appendChild(tr);
                });
            } else {
                alert('Lỗi dự đoán: ' + (result.error || 'Không xác định'));
            }
        } catch (error) {
            alert('Lỗi kết nối đến máy chủ.');
        } finally {
            uploadLoadingSpinner.classList.add('d-none');
        }
    });
});
