import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Langkah 1: Pengumpulan Data
# Membaca data dari file CSV
production_data = pd.read_csv('production_data.csv')
sales_data = pd.read_csv('sales_data.csv')
inventory_data = pd.read_csv('inventory_data.csv')
marketing_data = pd.read_csv('marketing_data.csv')
customer_data = pd.read_csv('customer_data.csv')

# Menggabungkan data berdasarkan tanggal dan produk
data = pd.merge(production_data, sales_data, on=['Date', 'Product'])
data = pd.merge(data, inventory_data, on=['Date', 'Product'])
data = pd.merge(data, marketing_data, on=['Date', 'Product'])
data = pd.merge(data, customer_data, on=['Date', 'Product'])

# Menampilkan beberapa baris pertama data
print("Data gabungan:")
print(data.head())

# Langkah 2: Data Cleaning
# Memeriksa nilai yang hilang
print("\nNilai yang hilang pada setiap kolom:")
print(data.isnull().sum())

# Mengisi atau menghapus data yang hilang jika ada
data = data.dropna()

# Memastikan tipe data benar
data['Date'] = pd.to_datetime(data['Date'])
data['Sales'] = data['Sales'].astype(float)
data['Quantity'] = data['Quantity'].astype(int)
data['Price'] = data['Price'].astype(float)
data['Units_Produced'] = data['Units_Produced'].astype(int)
data['Units_In_Stock'] = data['Units_In_Stock'].astype(int)
data['Marketing_Spend'] = data['Marketing_Spend'].astype(float)
data['Customer_Satisfaction'] = data['Customer_Satisfaction'].astype(float)

# Langkah 3: Data Transformation
# Menambah kolom baru jika diperlukan, misalnya Total Revenue
data['Total_Revenue'] = data['Quantity'] * data['Price']

# Langkah 4: Exploratory Data Analysis (EDA)
# Statistik deskriptif
print("\nStatistik deskriptif:")
print(data.describe())

# Membuat layout untuk beberapa plot dalam satu frame
plt.figure(figsize=(15, 12))

# Visualisasi Total Revenue Over Time
plt.subplot(2, 2, 1)
sns.lineplot(x='Date', y='Total_Revenue', data=data)
plt.title('Total Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)

# Analisis korelasi
plt.subplot(2, 2, 2)
# Menggunakan hanya kolom numerik untuk korelasi
numerical_data = data.select_dtypes(include=[np.number])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Scatter plot Quantity vs Sales
plt.subplot(2, 2, 3)
sns.scatterplot(x='Quantity', y='Sales', data=data)
plt.title('Quantity vs Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales')

# Bar plot untuk rata-rata kepuasan pelanggan berdasarkan produk
plt.subplot(2, 2, 4)
sns.barplot(x='Product', y='Customer_Satisfaction', data=data, estimator=np.mean)
plt.title('Average Customer Satisfaction by Product')
plt.xlabel('Product')
plt.ylabel('Customer Satisfaction')

# Menampilkan semua plot dalam satu frame
plt.tight_layout()
plt.show()

# Langkah 5: Modelling Data
# Memisahkan data menjadi fitur dan target
X = data[['Quantity', 'Price', 'Units_Produced', 'Units_In_Stock', 'Marketing_Spend', 'Customer_Satisfaction']]
y = data['Sales']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi
y_pred = model.predict(X_test)

# Menilai model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Langkah 6: Validasi dan Tuning Model
# Validasi silang
scores = cross_val_score(model, X, y, cv=5)
print(f'\nCross-Validation Scores: {scores}')
print(f'Average CV Score: {scores.mean()}')

# Langkah 7: Interpretasi dan Penyajian Hasil
# Menyajikan hasil dalam bentuk tabel atau grafik
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nHasil prediksi vs actual:")
print(results.head())

# Membuat layout untuk plot hasil prediksi vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


