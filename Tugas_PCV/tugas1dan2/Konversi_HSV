import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Balik gambar agar seperti cermin
    frame = cv2.flip(frame, 1)

    # --- POIN 1: Konversi ke HSV ---
    # Konversi frame dari BGR (standar OpenCV) ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # --- Akhir Poin 1 ---

    # Tampilkan frame asli dan frame HSV secara berdampingan
    cv2.imshow("Frame Asli (BGR)", frame)
    cv2.imshow("Frame HSV", hsv)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()