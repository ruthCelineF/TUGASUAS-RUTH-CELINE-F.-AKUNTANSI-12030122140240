# Analisis Data Penjualan

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan analisis data penjualan menggunakan Python. Data yang digunakan terdiri dari beberapa file CSV, termasuk data produksi, penjualan, persediaan, pemasaran, dan data pelanggan. Dalam proyek ini, kita akan melakukan berbagai tahapan analisis data mulai dari pengumpulan data, pembersihan data, exploratory data analysis (EDA), modelling data menggunakan regresi linear, hingga evaluasi model dan penyajian hasil.

## File Data
- `production_data.csv`: Data produksi yang mencakup tanggal produksi dan jumlah unit yang diproduksi.
- `sales_data.csv`: Data penjualan yang mencakup tanggal penjualan, produk, jumlah penjualan, dan harga.
- `inventory_data.csv`: Data persediaan yang mencakup tanggal, produk, dan jumlah unit yang tersedia.
- `marketing_data.csv`: Data pemasaran yang mencakup tanggal, produk, dan biaya pemasaran.
- `customer_data.csv`: Data pelanggan yang mencakup tanggal, produk, dan tingkat kepuasan pelanggan.

## Pertanyaan Analisis
1. Bagaimana tren total pendapatan dari waktu ke waktu?
2. Apakah ada korelasi antara variabel-variabel seperti harga, jumlah produksi, dan biaya pemasaran dengan penjualan?
3. Bagaimana distribusi jumlah penjualan terhadap jumlah produksi?
4. Bagaimana rata-rata kepuasan pelanggan untuk setiap produk?
5. Bagaimana kinerja model regresi linear dalam memprediksi penjualan?

## Jawaban dan Analisis
1. **Tren Total Pendapatan**: Tren total pendapatan dapat dilihat dari visualisasi "Total Revenue Over Time" yang menunjukkan perubahan pendapatan dari waktu ke waktu.
2. **Korelasi**: Matriks korelasi menunjukkan korelasi antara variabel-variabel seperti harga, jumlah produksi, dan biaya pemasaran dengan penjualan. Nilai korelasi yang lebih tinggi menunjukkan hubungan yang lebih kuat antara variabel.
3. **Distribusi Jumlah Penjualan vs Jumlah Produksi**: Scatter plot "Quantity vs Sales" menunjukkan distribusi jumlah penjualan terhadap jumlah produksi. Semakin tinggi jumlah produksi, semakin tinggi juga jumlah penjualan secara umum.
4. **Rata-Rata Kepuasan Pelanggan**: Bar plot "Average Customer Satisfaction by Product" menunjukkan rata-rata kepuasan pelanggan untuk setiap produk. Produk dengan rata-rata kepuasan yang lebih tinggi cenderung memiliki kinerja yang lebih baik dalam memuaskan pelanggan.
5. **Kinerja Model Regresi Linear**: Evaluasi model regresi linear menggunakan metrik seperti Mean Squared Error (MSE) dan R-squared Score. Semakin rendah MSE dan semakin tinggi R-squared Score, semakin baik kinerja model dalam memprediksi penjualan.

## Langkah-Langkah Analisis Data
1. **Pengumpulan Data**: Membaca data dari file CSV dan menggabungkannya menjadi satu DataFrame.
2. **Data Cleaning**: Memeriksa dan mengatasi nilai yang hilang, serta memastikan tipe data yang benar.
3. **Data Transformation**: Menambahkan kolom baru jika diperlukan, misalnya total pendapatan.
4. **Exploratory Data Analysis (EDA)**: Mengeksplorasi data menggunakan visualisasi seperti line plot, heatmap, scatter plot, dan bar plot.
5. **Modelling Data**: Memisahkan data menjadi fitur dan target, membagi data menjadi set pelatihan dan pengujian, dan melatih model regresi linear.
6. **Validasi dan Tuning Model**: Mengevaluasi kinerja model menggunakan metrik evaluasi dan melakukan validasi silang untuk memastikan keandalan model.
7. **Interpretasi dan Penyajian Hasil**: Menyajikan hasil analisis dalam bentuk tabel atau grafik, serta memberikan interpretasi dari hasil tersebut.
8. **Deploy dan Monitoring**: Menyimpan model yang telah dilatih dan memantau kinerja model di lingkungan produksi.
9. **Maintenance dan Iterasi**: Melakukan pemeliharaan dan retraining model dengan data terbaru jika diperlukan.

## Kontributor
- Nama: RUTH-CELINE-F
- NIM: 12030122140240
