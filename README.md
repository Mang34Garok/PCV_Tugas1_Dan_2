# PCV_Tugas1_Dan_2
Tugas Pengolahan Citra Video 
Tugas Pencitraan Video Dan Tracking kamera Karakter


1. HSV.py (Analisis Channel HSV)
Skrip ini bertujuan untuk membedah ruang warna HSV (Hue, Saturation, Value).

Fungsi: Mengambil feed webcam, mengonversinya ke HSV, lalu memecahnya menjadi 3 channel terpisah.


Output: Menampilkan 4 tampilan dalam satu layar (Grid 2x2): Frame Asli, Hue (Warna), Saturation (Kepekatan), dan Value (Kecerahan).


Kegunaan: Memahami bagaimana komputer melihat warna dibandingkan dengan mata manusia.

2. Konversi_HSV.py (Konversi Dasar)
Skrip pengantar sederhana untuk konversi ruang warna.

Fungsi: Mengubah format warna standar BGR (Blue-Green-Red) dari webcam menjadi HSV.


Output: Menampilkan dua jendela terpisah: Frame Asli dan Frame hasil konversi HSV berdampingan.

3. Thresholding.py (Alat Kalibrasi Warna)
Tool interaktif untuk mencari nilai HSV yang tepat untuk mendeteksi warna tertentu.


Fungsi: Menyediakan 6 Trackbar (Slider) untuk mengatur nilai Minimum dan Maksimum dari Hue, Saturation, dan Value secara real-time.

Fitur: Menghasilkan mask (topeng) biner hitam-putih. Bagian putih adalah warna yang masuk dalam rentang slider.

Kegunaan: Sangat penting untuk tahap "Kalibrasi" sebelum melakukan deteksi objek berwarna.

4. Pembersihan_mask.py (Reduksi Noise)
Mendemonstrasikan teknik Operasi Morfologi untuk membersihkan hasil deteksi yang kotor (banyak bintik noise).


Fungsi: Mengambil rentang warna (hardcoded untuk warna Biru) dan menerapkan filter Opening (menghapus bintik putih kecil) dan Closing (menutup lubang hitam pada objek).


Output: Membandingkan "Mask Awal" yang kotor dengan "Mask Bersih" hasil morfologi.

5. Tugas_DeteksiWarna.py (Multi-Color Tracking)
Implementasi lengkap pendeteksian objek berdasarkan warna.

Fungsi: Mendeteksi warna Merah, Hijau, dan Biru sekaligus secara real-time.


Logika Unik: Menangani kasus khusus warna Merah yang memiliki rentang Hue terpisah (di awal 0-10 dan di akhir 170-179) dengan menggabungkan dua mask.


Output: Menggambar kotak (Bounding Box) dan label nama warna pada objek yang terdeteksi.

6. Tugas_Blurring.py (Filtering & Smoothing)
Eksperimen dengan berbagai jenis filter kernel untuk memproses ketajaman citra.

Fungsi: Mengubah mode filter menggunakan input keyboard:

1: Average Blur 5x5.

2: Average Blur 9x9.


3: Gaussian Blur (menggunakan kernel Gaussian kustom).


4: Sharpening Filter (mempertajam tepi objek).

Bagian 2: Motion Capture & VTubing
# Python VTuber Motion Capture (MediaPipe to VSeeFace) 📸➡️💃

**Python VTuber Motion Capture** adalah sistem pelacakan gerak (Motion Capture) berbasis AI yang mengubah webcam standar menjadi alat pelacak tubuh penuh (*Upper Body* + *Hands*) untuk kebutuhan VTuber.

Sistem ini menggunakan **Google MediaPipe Holistic** untuk mendeteksi wajah, pose tubuh, dan tangan, kemudian mengonversi data tersebut menjadi protokol VMC (Virtual Motion Capture) dan mengirimkannya ke aplikasi **VSeeFace** melalui protokol OSC (UDP).

![Python to VSeeFace Visualization](https://via.placeholder.com/800x400?text=Screenshot+Visualisasi+Tracking+Anda)
*(Ganti gambar di atas dengan screenshot hasil tracking program Anda)*

## 🌟 Fitur Utama

* **Full Upper Body Tracking:** Melacak rotasi kepala, bahu, siku, dan pergelangan tangan dengan koreksi rotasi tulang (Quaternion).
* **High Precision Hand Tracking:** Deteksi pergerakan 10 jari secara individual.
* **Advanced Mouth Tracking (MAR):** Menggunakan *Mouth Aspect Ratio* (MAR) sehingga bukaan mulut tetap akurat meskipun jarak wajah ke kamera berubah-ubah (maju/mundur).
* **Smart Smoothing:** Menggunakan **One Euro Filter** untuk menghilangkan getaran (jitter) saat diam, namun tetap responsif saat bergerak cepat.
* **Visualisasi Debugging:** Dilengkapi GUI yang menampilkan:
    * Jaring wajah (Tesselation).
    * Indikator kepercayaan tracking (Titik Hijau/Merah di bahu).
    * Garis bantu deteksi mulut (Kuning).
* **Eye Tracking:** Deteksi kedipan dan pergerakan iris mata.

## 🛠️ Prasyarat (Requirements)

Pastikan Anda telah menginstal **Python 3.8** atau versi yang lebih baru. Proyek ini membutuhkan pustaka berikut:

* `opencv-python` (Pengolahan citra)
* `mediapipe` (AI Tracking)
* `python-osc` (Komunikasi ke VSeeFace)
* `numpy` (Kalkulasi Matematika)

## 📦 Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
    cd nama-repo-anda
    ```

2.  **Instal dependensi:**
    Salin dan jalankan perintah ini di terminal/CMD:
    ```bash
    pip install opencv-python mediapipe python-osc numpy
    ```

## ⚙️ Konfigurasi VSeeFace

Agar karakter VTuber Anda bergerak mengikuti data dari Python, lakukan pengaturan berikut di **VSeeFace**:

1.  Buka VSeeFace dan muat avatar Anda.
2.  Masuk ke **Settings** > **General Settings**.
3.  Cari bagian **OSC / VMC Receiver**.
4.  Centang **Enable**.
5.  Pastikan konfigurasi berikut sesuai:
    * **Port:** `39539` (Default)
    * **IP Address:** `127.0.0.1` (Localhost)
6.  **Penting:** Pada bagian pemilihan kamera di VSeeFace, pilih **(None)** atau matikan kamera bawaan VSeeFace agar tidak bentrok dengan skrip Python.

## 🚀 Cara Penggunaan

1.  Sambungkan webcam ke komputer.
2.  Jalankan skrip utama:
    ```bash
    python vseeface_sender.py
    ```
    *(Sesuaikan nama file jika Anda mengubahnya)*
3.  Akan muncul jendela visualisasi.
    * **Garis Kuning di Mulut:** Menandakan deteksi bibir aktif.
    * **Titik Hijau di Bahu:** Menandakan tracking tubuh stabil.
    * **Titik Merah:** Menandakan tracking hilang (sistem akan menahan gerakan agar avatar tidak *glitch*).
4.  Buka VSeeFace, dan avatar Anda akan mulai bergerak!
5.  Tekan tombol **`q`** pada keyboard di jendela Python untuk keluar.

## 🔧 Kustomisasi & Tuning

Anda dapat mengubah variabel di bagian atas kode (`vseeface_sender.py`) untuk menyesuaikan sensitivitas:

### 1. Sensitivitas Mulut
Jika mulut avatar tidak mau menutup rapat atau tidak terbuka lebar:
```python
MOUTH_MIN_RATIO = 0.05  # Rasio saat mulut diam/tutup
MOUTH_MAX_RATIO = 0.35  # Rasio saat mulut terbuka lebar ('A')
