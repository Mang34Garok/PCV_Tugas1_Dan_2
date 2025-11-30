import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Poin 2: Tentukan Rentang Warna (Biru)
lower_bound = np.array([100, 150, 50])
upper_bound = np.array([130, 255, 255])

# --- POIN 3 (Definisi): Buat Kernel Morfologi ---
# Kernel adalah matriks kecil (5x5) yang digunakan untuk operasi.
# Ukuran ini bisa diubah-ubah untuk noise yang lebih besar/kecil.
kernel = np.ones((5, 5), np.uint8)
# --- Akhir Poin 3 (Definisi) ---

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    # POIN 1: Konversi ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # POIN 2: Thresholding
    mask_awal = cv2.inRange(hsv, lower_bound, upper_bound)

    # --- POIN 3 (Aplikasi): Pembersihan Mask ---
    # 1. Opening: (Erosi diikuti Dilasi).
    #    Ini efektif menghapus bintik-bintik noise kecil (false positives).
    mask_opening = cv2.morphologyEx(mask_awal, cv2.MORPH_OPEN, kernel)
    
    # 2. Closing: (Dilasi diikuti Erosi).
    #    Ini efektif menutup lubang-lubang kecil di dalam objek utama.
    mask_bersih = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel)
    # --- Akhir Poin 3 (Aplikasi) ---

    # Tampilkan perbandingan mask
    cv2.imshow("Mask Awal (Berderau)", mask_awal)
    cv2.imshow("Mask Bersih (Hasil Morfologi)", mask_bersih)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()