import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time

# ==============================================================================
# 1. KONFIGURASI
# ==============================================================================
OSC_IP = "127.0.0.1"
OSC_PORT = 39539
WEBCAM_ID = 0
TARGET_FPS = 30

# --- SENSITIVITAS MULUT (BISA DIATUR DI SINI) ---
MOUTH_MIN_RATIO = 0.05  # Rasio saat mulut diam/tutup (jika mulut Anda tebal, naikkan sedikit)
MOUTH_MAX_RATIO = 0.35  # Rasio saat mulut terbuka lebar (mengucap 'A')

# --- SMOOTHING ---
HEAD_MIN_CUTOFF = 0.05 ; HEAD_BETA = 0.5
BODY_MIN_CUTOFF = 0.05 ; BODY_BETA = 0.5
ARM_MIN_CUTOFF  = 0.1  ; ARM_BETA  = 0.6
FINGER_MIN_CUTOFF = 0.5; FINGER_BETA = 1.0

# CONFIDENCE
CONFIDENCE_THRESHOLD = 0.6 

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

def get_finger_quat(angle, axis_idx):
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    if axis_idx == 0:   return [s, 0, 0, c]
    elif axis_idx == 1: return [0, s, 0, c]
    elif axis_idx == 2: return [0, 0, s, c]
    return [0, 0, 0, 1]

def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1]
    v_curr = v_curr / norm
    v_rest = np.array(rest_vector)
    v_rest = v_rest / np.linalg.norm(v_rest)
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    axis = axis / axis_len
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)
    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    curl = (ratio - 1.9) / (0.8 - 1.9)
    return max(0.0, min(1.0, curl)) * 1.3 

def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

# === FUNGSI BARU: MOUTH ASPECT RATIO (MAR) ===
def calculate_mar(face_landmarks, img_w, img_h):
    lm = face_landmarks.landmark
    # 13 = Bibir Atas, 14 = Bibir Bawah
    # 61 = Ujung Kiri, 291 = Ujung Kanan
    top = np.array([lm[13].x * img_w, lm[13].y * img_h])
    bottom = np.array([lm[14].x * img_w, lm[14].y * img_h])
    left = np.array([lm[61].x * img_w, lm[61].y * img_h])
    right = np.array([lm[291].x * img_w, lm[291].y * img_h])
    
    # Hitung Jarak Vertikal (Buka Mulut)
    vertical_dist = np.linalg.norm(top - bottom)
    # Hitung Jarak Horizontal (Lebar Mulut)
    horizontal_dist = np.linalg.norm(left - right)
    
    # Rasio = Vertikal / Horizontal
    ratio = vertical_dist / (horizontal_dist + 1e-6)
    return ratio, top, bottom, left, right

# ==============================================================================
# 3. CLASS FILTER
# ==============================================================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# ==============================================================================
# 4. INIT
# ==============================================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6, 
    refine_face_landmarks=True,
    model_complexity=1
)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

t_start = time.time()
filter_pitch = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_yaw   = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_roll  = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)

filter_l_sh = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_l_el = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_l_wr = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_sh = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_el = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_wr = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]

filters_fingers_L = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]
filters_fingers_R = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]

# Filter KHUSUS MULUT (Cepat, beta tinggi agar responsif)
filter_mouth = OneEuroFilter(t_start, 0, min_cutoff=0.01, beta=2.0)

model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)] 
BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]
THUMB_SIGN_L, THUMB_SIGN_R = -1.0, 1.0
blink_l_state, blink_r_state = 0.0, 0.0
prev_time = 0

cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

print("=== TRACKING SYSTEM: READY ===")
print("Perhatikan garis KUNING di mulut.")
print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 and (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image.flags.writeable = False
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True

    # 1. VISUALISASI UTAMA
    if results.face_landmarks:
        # Tesselation (Jaring Wajah)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 2. LOGIKA UTAMA
    if results.face_landmarks:
        fl = results.face_landmarks
        
        # --- A. MULUT (LOGIKA BARU - MAR) ---
        
        # Menggunakan Landmark 13 (Atas), 14 (Bawah), 61 (Kiri), 291 (Kanan)
        
        raw_mar, pt_top, pt_bot, pt_left, pt_right = calculate_mar(fl, img_w, img_h)
        
        # Smooth nilai MAR agar tidak bergetar
        smooth_mar = filter_mouth(curr_time, raw_mar)

        # Mapping: Ubah rasio (0.05 - 0.35) menjadi (0.0 - 1.0) untuk VSeeFace
        mouth_open = (smooth_mar - MOUTH_MIN_RATIO) / (MOUTH_MAX_RATIO - MOUTH_MIN_RATIO)
        mouth_open = max(0.0, min(1.0, mouth_open)) # Clamp nilai agar tidak minus atau > 1

        # VISUALISASI KHUSUS MULUT (GARIS KUNING)
        # Garis Vertikal (Bukaan)
        cv2.line(image, (int(pt_top[0]), int(pt_top[1])), (int(pt_bot[0]), int(pt_bot[1])), (0, 255, 255), 2)
        # Garis Horizontal (Lebar)
        cv2.line(image, (int(pt_left[0]), int(pt_left[1])), (int(pt_right[0]), int(pt_right[1])), (0, 255, 255), 2)
        
        # Debugging: Tampilkan nilai bukaan mulut di layar
        cv2.putText(image, f"Mouth: {mouth_open:.2f}", (int(pt_right[0])+10, int(pt_right[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Kirim data ke VSeeFace
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])


        # --- B. KEPALA ---
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)
        
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        s_pitch = filter_pitch(curr_time, angles[0])
        s_yaw   = filter_yaw(curr_time, angles[1])
        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        s_roll  = filter_roll(curr_time, math.degrees(math.atan2(dY, dX)))

        neck_ratio = 0.5
        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(s_pitch*neck_ratio), math.radians(s_yaw*neck_ratio), math.radians(s_roll*neck_ratio))
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(s_pitch*(1-neck_ratio)), math.radians(s_yaw*(1-neck_ratio)), math.radians(s_roll*(1-neck_ratio)))
        
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nqx), float(nqy), float(nqz), float(nqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hqx), float(hqy), float(hqz), float(hqw)])
        
        # MATA
        raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
        raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)
        if raw_ear_l < 0.15: blink_l_state = 1.0 
        elif raw_ear_l > 0.25: blink_l_state = 0.0 
        if raw_ear_r < 0.15: blink_r_state = 1.0
        elif raw_ear_r > 0.25: blink_r_state = 0.0
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r_state)])

    # --- C. BADAN ---
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def get_filtered_vec(idx, filters):
            raw_x, raw_y, raw_z = lm[idx].x, lm[idx].y, lm[idx].z
            fx = filters[0](curr_time, raw_x)
            fy = filters[1](curr_time, raw_y)
            fz = filters[2](curr_time, raw_z)
            return [fx, fy, fz], lm[idx].visibility
        def to_unity_vec(vec):
            return np.array([vec[0]*1.2, vec[1]*1.2, vec[2]*0.4])

        l_sh_vec, l_sh_vis = get_filtered_vec(11, filter_l_sh)
        l_el_vec, l_el_vis = get_filtered_vec(13, filter_l_el)
        l_wr_vec, l_wr_vis = get_filtered_vec(15, filter_l_wr)
        r_sh_vec, r_sh_vis = get_filtered_vec(12, filter_r_sh)
        r_el_vec, r_el_vis = get_filtered_vec(14, filter_r_el)
        r_wr_vec, r_wr_vis = get_filtered_vec(16, filter_r_wr)

        if l_sh_vis > CONFIDENCE_THRESHOLD and l_el_vis > CONFIDENCE_THRESHOLD:
            start, end = to_unity_vec(l_sh_vec), to_unity_vec(l_el_vec)
            q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0]) 
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
            if l_wr_vis > CONFIDENCE_THRESHOLD:
                start, end = to_unity_vec(l_el_vec), to_unity_vec(l_wr_vec)
                q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])

        if r_sh_vis > CONFIDENCE_THRESHOLD and r_el_vis > CONFIDENCE_THRESHOLD:
            start, end = to_unity_vec(r_sh_vec), to_unity_vec(r_el_vec)
            q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
            if r_wr_vis > CONFIDENCE_THRESHOLD:
                start, end = to_unity_vec(r_el_vec), to_unity_vec(r_wr_vec)
                q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])

    # --- D. JARI ---
    if results.left_hand_landmarks:
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0)
            curl = filters_fingers_L[i](curr_time, raw_curl)
            angle = curl * (math.pi / 2.0) * -1.0 if name == "Thumb" else curl * (math.pi / 1.5) * 1.0
            axis = 1 if name == "Thumb" else 2
            fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
            client.send_message(f"/VMC/Ext/Bone/Pos", [f"Left{name}Proximal", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])
            client.send_message(f"/VMC/Ext/Bone/Pos", [f"Left{name}Intermediate", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    if results.right_hand_landmarks:
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0)
            curl = filters_fingers_R[i](curr_time, raw_curl)
            angle = curl * (math.pi / 2.0) * 1.0 if name == "Thumb" else curl * (math.pi / 1.5) * -1.0
            axis = 1 if name == "Thumb" else 2
            fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
            client.send_message(f"/VMC/Ext/Bone/Pos", [f"Right{name}Proximal", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])
            client.send_message(f"/VMC/Ext/Bone/Pos", [f"Right{name}Intermediate", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Tracking + Mouth Fix', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
