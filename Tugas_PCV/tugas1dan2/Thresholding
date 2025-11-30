import cv2
import numpy as np

# Fungsi ini diperlukan untuk createTrackbar, tapi kita tidak melakukan apa-apa di dalamnya
def nothing(x):
    pass

# Buka webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam")
    exit()

# Buat jendela baru untuk menampung trackbar
cv2.namedWindow("Pengaturan Warna")

# --- Buat 6 Trackbar untuk rentang HSV ---
# cv2.createTrackbar("Nama", "Jendela", Nilai_Awal, Nilai_Maks, Fungsi_Callback)

# 1. Hue (Warna)
# Catatan: Hue di OpenCV adalah 0-179 (bukan 255)
cv2.createTrackbar("H Min", "Pengaturan Warna", 0, 179, nothing)
cv2.createTrackbar("H Max", "Pengaturan Warna", 179, 179, nothing)

# 2. Saturation (Kepekatan)
cv2.createTrackbar("S Min", "Pengaturan Warna", 0, 255, nothing)
cv2.createTrackbar("S Max", "Pengaturan Warna", 255, 255, nothing)

# 3. Value (Kecerahan)
cv2.createTrackbar("V Min", "Pengaturan Warna", 0, 255, nothing)
cv2.createTrackbar("V Max", "Pengaturan Warna", 255, 255, nothing)

# --- Atur posisi awal (Opsional, tapi membantu) ---
# Mari kita setel ke nilai 'Biru' dari contoh sebelumnya
cv2.setTrackbarPos("H Min", "Pengaturan Warna", 100)
cv2.setTrackbarPos("H Max", "Pengaturan Warna", 130)
cv2.setTrackbarPos("S Min", "Pengaturan Warna", 150)
cv2.setTrackbarPos("S Max", "Pengaturan Warna", 255)
cv2.setTrackbarPos("V Min", "Pengaturan Warna", 50)
cv2.setTrackbarPos("V Max", "Pengaturan Warna", 255)


while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)

    # --- POIN 1: Konversi ke HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Ambil nilai dari 6 trackbar ---
    h_min = cv2.getTrackbarPos("H Min", "Pengaturan Warna")
    h_max = cv2.getTrackbarPos("H Max", "Pengaturan Warna")
    s_min = cv2.getTrackbarPos("S Min", "Pengaturan Warna")
    s_max = cv2.getTrackbarPos("S Max", "Pengaturan Warna")
    v_min = cv2.getTrackbarPos("V Min", "Pengaturan Warna")
    v_max = cv2.getTrackbarPos("V Max", "Pengaturan Warna")

    # --- POISTAS 2: Thresholding Warna (Dinamis) ---
    # Buat array lower dan upper bound berdasarkan nilai trackbar
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # Buat mask biner
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # (Opsional) Tampilkan hasil warna yang terdeteksi saja
    hasil_warna = cv2.bitwise_and(frame, frame, mask=mask)

    # Tampilkan gambar
    cv2.imshow("Frame Asli", frame)
    cv2.imshow("Mask Hasil", mask)
    cv2.imshow("Hasil Warna", hasil_warna) # Tampilkan ini untuk melihat warna apa yg terambil

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()