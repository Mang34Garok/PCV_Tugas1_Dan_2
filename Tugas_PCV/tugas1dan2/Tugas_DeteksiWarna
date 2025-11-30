import cv2
import numpy as np

# --- 1. DEFINISI WARNA ---
# Kita akan membuat daftar (list) dari warna yang ingin dideteksi.
# Setiap warna adalah dictionary yang berisi:
# 'name': Nama warna (untuk label)
# 'lower': Rentang HSV bawah
# 'upper': Rentang HSV atas
# 'color_bgr': Warna BGR untuk kotak (misal, Biru = (255, 0, 0))
#
# !! GANTI NILAI DI BAWAH INI DENGAN HASIL KALIBRASI ANDA !!

colors_to_detect = [
    {
        "name": "BIRU",
        # Ganti dengan nilai kalibrasi Anda
        "lower": np.array([100, 150, 50]),
        "upper": np.array([140, 255, 255]),
        "color_bgr": (255, 0, 0) # Warna BGR untuk kotak
    },
    {
        "name": "HIJAU",
        # Ganti dengan nilai kalibrasi Anda
        "lower": np.array([35, 100, 50]),
        "upper": np.array([85, 255, 255]),
        "color_bgr": (0, 255, 0)
    },
    # CATATAN KHUSUS UNTUK MERAH:
    # Warna Merah ada di dua ujung rentang Hue (0-10 dan 170-179).
    # Jadi kita perlu mendefinisikannya sedikit berbeda (lihat di loop).
    {
        "name": "MERAH",
        # Rentang Merah #1 (ujung bawah)
        "lower": np.array([0, 150, 50]),
        "upper": np.array([10, 255, 255]),
        # Rentang Merah #2 (ujung atas)
        "lower2": np.array([170, 150, 50]), 
        "upper2": np.array([179, 255, 255]),
        "color_bgr": (0, 0, 255)
    }
]


# --- 2. PENGATURAN AWAL ---

# Kernel untuk operasi morfologi (membersihkan mask)
kernel_morph_open = np.ones((5, 5), np.uint8)
kernel_morph_close = np.ones((20, 20), np.uint8)

# Minimal area kontur agar dianggap sebagai objek
MIN_AREA = 3000

# Buka Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Balik frame (efek cermin)
    frame = cv2.flip(frame, 1)
    
    # Konversi frame ke HSV (cukup sekali)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- 3. LOOP DETEKSI WARNA ---
    
    # Loop untuk setiap warna yang kita definisikan di atas
    for color in colors_to_detect:
        
        # --- A. Buat Mask ---
        if color['name'] == "MERAH":
            # Jika Merah, gabungkan 2 mask
            mask1 = cv2.inRange(hsv, color['lower'], color['upper'])
            mask2 = cv2.inRange(hsv, color['lower2'], color['upper2'])
            mask = mask1 + mask2 # Gabungkan kedua mask
        else:
            # Untuk warna lain, cukup 1 mask
            mask = cv2.inRange(hsv, color['lower'], color['upper'])
        
        # --- B. Bersihkan Mask ---
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph_open)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_morph_close)
        
        # --- C. Temukan Kontur ---
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop semua kontur yang ditemukan untuk warna ini
        for c in contours:
            area = cv2.contourArea(c)
            
            # Jika areanya cukup besar
            if area > MIN_AREA:
                # Dapatkan kotak
                x, y, w, h = cv2.boundingRect(c)
                
                # Ambil warna BGR dari definisi
                box_color = color['color_bgr']
                
                # Gambar kotak
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
                
                # Tulis label nama warna
                cv2.putText(frame, color['name'], (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2, cv2.LINE_AA)

    # --- 4. Tampilkan Hasil ---
    cv2.imshow("Deteksi 3 Warna (Tekan 'q' untuk keluar)", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Selesai
cap.release()
cv2.destroyAllWindows()