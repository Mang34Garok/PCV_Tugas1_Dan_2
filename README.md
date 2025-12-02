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
7. vseeface_sender.py (MediaPipe to OSC)
Sistem pelacakan gerak (Motion Capture) tanpa marker yang mengubah webcam biasa menjadi pengendali karakter 3D (VTuber).

Teknologi: Menggunakan MediaPipe Holistic untuk melacak Wajah, Pose (Tubuh), dan Tangan secara bersamaan.

Fitur Utama:

Smoothing: Menggunakan One Euro Filter untuk mengurangi getaran (jitter) pada gerakan karakter.

Matematika 3D: Mengonversi koordinat landmark menjadi Rotasi Quaternion.

Komunikasi: Mengirim data gerakan via protokol OSC (Open Sound Control) ke aplikasi seperti VSeeFace atau Unity.

Pelacakan: Mencakup kedipan mata, gerakan bola mata (iris tracking), mulut, leher, tulang belakang, tangan, dan jari-jari.
