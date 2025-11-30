import cv2
import numpy as np

# --- Inisialisasi Kernel (Tugas 1) ---

# 2. Kernel Gaussian (Wajib menggunakan filter2D) [cite: 35]
#    Kita buat kernel 9x9 sebagai contoh
ksize = 9
sigma = 1.5
# Dapatkan kernel 1D
gaussian_kernel_x = cv2.getGaussianKernel(ksize, sigma)
gaussian_kernel_y = cv2.getGaussianKernel(ksize, sigma)
# Buat kernel 2D dari perkalian kernel 1D
gaussian_kernel_2d = gaussian_kernel_x * gaussian_kernel_y.T

# 3. Kernel Sharpening [cite: 37]
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# --- Variabel Status ---
filter_mode = '0'  # Mode filter awal
filter_text = "Normal"

# Buka Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

print("Kontrol Keyboard (Tugas 1):")
print("[0] Normal")
print("[1] Average Blur (5x5)")
print("[2] Average Blur (9x9)")
print("[3] Gaussian Blur (9x9)")
print("[4] Sharpen Filter")
print("[q] Keluar")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Balik frame secara horizontal (efek cermin)
    frame = cv2.flip(frame, 1)
    
    # Salin frame asli untuk difilter
    display_frame = frame.copy()

    # --- Kontrol Keyboard  ---
    key = cv2.waitKey(5) & 0xFF

    if key == ord('0'):
        filter_mode = '0'
        filter_text = "Normal"
    elif key == ord('1'):
        filter_mode = '1'
        filter_text = "Average Blur (5x5)"
    elif key == ord('2'): # Tombol tambahan untuk kernel kedua 
        filter_mode = '2'
        filter_text = "Average Blur (9x9)"
    elif key == ord('3'):
        filter_mode = '3'
        filter_text = "Gaussian Blur (9x9)"
    elif key == ord('4'):
        filter_mode = '4'
        filter_text = "Sharpen Filter"
    elif key == ord('q'):
        break # Keluar dari loop

    # --- Terapkan Filter Sesuai Mode ---
    if filter_mode == '1':
        # 1. Average Blurring 5x5 
        display_frame = cv2.blur(frame, (5, 5))
    elif filter_mode == '2':
        # 1. Average Blurring 9x9 
        display_frame = cv2.blur(frame, (9, 9))
    elif filter_mode == '3':
        # 2. Gaussian Blurring (Wajib menggunakan filter2D) [cite: 35]
        display_frame = cv2.filter2D(frame, -1, gaussian_kernel_2d)
    elif filter_mode == '4':
        # 3. Sharpening Filter [cite: 36]
        display_frame = cv2.filter2D(frame, -1, kernel_sharpen)
    # else (filter_mode == '0'): 'display_frame' tetap frame asli

    # Tampilkan teks status di layar
    cv2.putText(display_frame, f"Filter: {filter_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Tampilkan hasil
    cv2.imshow("Tugas 1 - Smoothing dan Blurring (Tekan 'q' untuk keluar)", display_frame)

# Selesai
cap.release()
cv2.destroyAllWindows()