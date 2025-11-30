import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam")
    exit()

print("Tekan 'q' untuk keluar...")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Balik gambar agar seperti cermin
    frame = cv2.flip(frame, 1)

    # 1. Konversi frame dari BGR ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Pecah gambar HSV menjadi 3 channel terpisah
    h, s, v = cv2.split(hsv)

    # --- Menggabungkan Jadi Satu Layar ---
    
    # 3. Konversi H, S, dan V (1-channel, grayscale) kembali ke BGR (3-channel)
    #    agar bisa digabung dengan 'frame' asli.
    h_bgr = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    s_bgr = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_bgr = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    # 4. Buat label untuk setiap video
    cv2.putText(frame, "Asli (BGR)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(h_bgr, "Hue (Warna)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(s_bgr, "Saturation (Kepekatan)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(v_bgr, "Value (Kecerahan)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. Susun gambarnya dalam grid 2x2
    # Tumpuk baris atas (Asli dan Hue) secara horizontal
    baris_atas = cv2.hconcat([frame, h_bgr])
    
    # Tumpuk baris bawah (Saturation dan Value) secara horizontal
    baris_bawah = cv2.hconcat([s_bgr, v_bgr])
    
    # Tumpuk baris atas dan bawah secara vertikal
    combined_view = cv2.vconcat([baris_atas, baris_bawah])

    # Tampilkan satu jendela gabungan
    cv2.imshow("HSV Channels (Satu Layar)", combined_view)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()