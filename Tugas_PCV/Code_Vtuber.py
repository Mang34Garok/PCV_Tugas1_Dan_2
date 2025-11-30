import cv2
import mediapipe as mp
import numpy as np
import math

# --- Fungsi Bantuan ---

def get_distance(p1, p2):
    """Menghitung jarak antara dua titik landmark (float)."""
    if not p1 or not p2: return 0
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_pixel_distance(p1, p2):
    """Menghitung jarak piksel antara dua titik (tuple int)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def overlay_transparent(background, overlay, x, y):
    """
    Menempelkan gambar 'overlay' (dengan transparansi) ke 'background'
    pada koordinat x, y.
    """
    bg_h, bg_w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape

    # Dapatkan channel alpha
    overlay_image = overlay[:, :, 0:3] # Gambar BGR
    overlay_mask = overlay[:, :, 3]    # Channel Alpha

    # Hitung batas di mana gambar akan ditempatkan
    h, w = overlay_h, overlay_w
    
    # Pastikan overlay tidak keluar dari batas atas/kiri
    if y < 0:
        h = h + y
        y = 0
        overlay_mask = overlay_mask[-h:]
        overlay_image = overlay_image[-h:]
    if x < 0:
        w = w + x
        x = 0
        overlay_mask = overlay_mask[:, -w:]
        overlay_image = overlay_image[:, -w:]

    # Hitung batas bawah/kanan
    y_end = y + h
    x_end = x + w
    
    # Pastikan overlay tidak keluar dari batas bawah/kanan
    if y_end > bg_h:
        h = bg_h - y
        overlay_mask = overlay_mask[:h]
        overlay_image = overlay_image[:h]
    if x_end > bg_w:
        w = bg_w - x
        overlay_mask = overlay_mask[:, :w]
        overlay_image = overlay_image[:, :w]

    # Ambil Region of Interest (ROI) dari background
    roi = background[y:y+h, x:x+w]

    # Buat mask 3-channel
    overlay_mask_3ch = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    
    # Invert mask (bagian transparan jadi 1, bagian gambar jadi 0)
    mask_inv = cv2.bitwise_not(overlay_mask_3ch)
    
    # Hitamkan area avatar di ROI
    bg_masked = cv2.bitwise_and(roi, mask_inv)
    
    # Ambil hanya bagian gambar dari overlay
    overlay_masked = cv2.bitwise_and(overlay_image, overlay_mask_3ch)

    # Gabungkan keduanya dan tempatkan kembali ke background
    try:
        dst = cv2.add(bg_masked, overlay_masked)
        background[y:y+h, x:x+w] = dst
    except Exception as e:
        print(f"Error overlay: {e}")
        pass
    
    return background


# --- Inisialisasi MediaPipe Holistic & Webcam ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# --- Muat Gambar Avatar ---
try:
    avatar_img_original = cv2.imread('karakter.png', cv2.IMREAD_UNCHANGED)
    if avatar_img_original is None:
        raise FileNotFoundError
    avatar_h_orig, avatar_w_orig, _ = avatar_img_original.shape
except FileNotFoundError:
    print("="*50)
    print("ERROR: Gagal memuat 'karakter.png'.")
    print("Pastikan file ada di folder yang sama dengan skrip ini.")
    print("Membuat gambar placeholder...")
    print("="*50)
    avatar_img_original = np.zeros((200, 200, 4), dtype=np.uint8)
    cv2.putText(avatar_img_original, 'PNG GAGAL', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
    avatar_h_orig, avatar_w_orig, _ = avatar_img_original.shape


# --- Loop Utama ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    # Buat kanvas kosong
    avatar_canvas = np.full((h, w, 3), (200, 230, 180), dtype=np.uint8) # Warna mint

    # Proses gambar
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark

        # Dapatkan landmark yang diperlukan
        left_shoulder = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]

        # Konversi ke piksel
        ls_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        rs_pos = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        nose_pos = (int(nose.x * w), int(nose.y * h))
        
        # Cek jika landmark terlihat
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            
            # --- Logika Skala Avatar ---
            # Hitung lebar bahu di layar
            shoulder_width_px = get_pixel_distance(ls_pos, rs_pos)
            
            # Tentukan skala avatar berdasarkan lebar bahu
            # Angka '1.5' adalah faktor skala, bisa Anda ubah
            scale_factor = (shoulder_width_px * 1.5) / avatar_w_orig
            
            # Ubah ukuran avatar
            new_w = int(avatar_w_orig * scale_factor)
            new_h = int(avatar_h_orig * scale_factor)
            
            if new_w > 0 and new_h > 0:
                avatar_resized = cv2.resize(avatar_img_original, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # --- Logika Posisi Avatar ---
                # Posisikan avatar di atas bahu, berpusat di hidung
                # 'new_h // 2' agar bagian tengah avatar ada di hidung
                pos_x = nose_pos[0] - (new_w // 2)
                pos_y = nose_pos[1] - (new_h // 2)

                # Panggil fungsi overlay
                avatar_canvas = overlay_transparent(avatar_canvas, avatar_resized, pos_x, pos_y)


    # Tampilkan webcam kecil di pojok
    image_resized = cv2.resize(image, (w // 4, h // 4))
    avatar_canvas[20:20+h//4, w-20-w//4:w-20, :] = image_resized

    cv2.imshow('Avatar 2D Sederhana', avatar_canvas)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()