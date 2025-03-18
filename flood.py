# Import library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans

# Langkah 1: Tentukan path dataset
dataset_path = "dataset/images"  # Ganti dengan path dataset Anda
output_folder = "hasil_segmentasi"  # Folder untuk menyimpan hasil segmentasi

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Langkah 2: List semua file gambar dalam dataset (semua ekstensi)
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in valid_extensions]

# Langkah 3: Buat list untuk menyimpan hasil analisis
all_results = []

# Langkah 4: Loop melalui setiap gambar dalam dataset
for image_file in image_files:
    # Load gambar
    image = cv2.imread(image_file)

    # Pengecekan: Apakah gambar berhasil dibaca?
    if image is None:
        print(f"Gagal membaca gambar: {image_file}")
        continue  # Lanjut ke gambar berikutnya

    # Convert ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Langkah 5: Analisis warna (histogram dan intensitas)
    mean_intensity = np.mean(image, axis=(0, 1))  # Rata-rata intensitas warna (R, G, B)
    std_intensity = np.std(image, axis=(0, 1))    # Standar deviasi intensitas warna (R, G, B)

    # Langkah 6: Reshape gambar untuk K-Means Clustering
    pixel_values = image.reshape((-1, 3))  # Ubah gambar menjadi array 2D (setiap baris adalah pixel RGB)

    # Langkah 7: Terapkan K-Means Clustering
    kmeans = KMeans(n_clusters=2)  # Gunakan 2 cluster (banjir dan non-banjir)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_

    # Langkah 8: Bentuk kembali hasil clustering ke bentuk gambar
    segmented_image = labels.reshape(image.shape[0], image.shape[1])

    # Langkah 9: Hitung luas area banjir
    flood_area = np.sum(segmented_image == 1)  # Asumsikan cluster 1 adalah area banjir
    total_pixels = segmented_image.shape[0] * segmented_image.shape[1]  # Total pixel dalam gambar
    flood_percentage = (flood_area / total_pixels) * 100  # Persentase area banjir

    # Langkah 10: Simpan hasil analisis ke dalam dictionary
    results = {
        'image_file': image_file,
        'mean_intensity_r': mean_intensity[0],
        'mean_intensity_g': mean_intensity[1],
        'mean_intensity_b': mean_intensity[2],
        'std_intensity_r': std_intensity[0],
        'std_intensity_g': std_intensity[1],
        'std_intensity_b': std_intensity[2],
        'flood_area_pixels': flood_area,
        'flood_percentage': flood_percentage
    }

    # Tambahkan hasil ke list
    all_results.append(results)

    # Langkah 11: Visualisasi gambar dan hasil segmentasi
    plt.figure(figsize=(10, 5))

    # Tampilkan gambar asli
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Gambar Asli")
    plt.axis('off')

    # Tampilkan hasil segmentasi
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title("Hasil Segmentasi (K-Means)")
    plt.axis('off')

    # Simpan hasil segmentasi ke folder output
    output_file = os.path.join(output_folder, os.path.basename(image_file))
    plt.savefig(output_file)
    plt.close()

# Langkah 12: Simpan semua hasil analisis ke dalam file CSV
df_all_results = pd.DataFrame(all_results)
df_all_results.to_csv('all_flood_analysis_results.csv', index=False)

print("Analisis selesai. Hasil disimpan di 'all_flood_analysis_results.csv' dan gambar segmentasi disimpan di folder 'hasil_segmentasi'.")