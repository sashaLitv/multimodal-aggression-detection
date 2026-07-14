from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import inspect
import glob
from scipy.signal import savgol_filter

from src.config import YOLO_PATH, SEQUENCE_LENGTH, HISTORY_LEN, STEP, OUTPUTS_DIR, VIDEO_DATA_DIR, KINETIC_CLIP_BOUND, GLOBAL_VEL_STD, GLOBAL_ACC_STD

NORMAL_RANGE = (
    [x for x in range(1, 41) if x != 24] + 
    [x for x in range(61, 103) if x not in [65, 93, 102]] + 
    [53, 54, 55, 56, 58, 59, 60, 112, 113, 114, 115, 116, 117, 118, 119, 120]
)

ANOMALY_RANGE = [50, 51, 52, 106, 107, 108]

yolo_model = YOLO(YOLO_PATH)

def read_ntu_skeleton(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if not lines: return []
    
    n_frames = int(lines[0].strip())
    bodies_data = {}
    cursor = 1

    for f_idx in range(n_frames):
        if cursor >= len(lines): break
        body_count = int(lines[cursor].strip())
        cursor +=1
        for b in range(body_count):
            body_id = lines[cursor].strip().split()[0]
            cursor += 1
            joint_count = int(lines[cursor].strip())
            cursor += 1
            if body_id not in bodies_data:
                bodies_data[body_id] = np.zeros((n_frames, 25, 2))
            for j in range(joint_count):
                joint_info = lines[cursor].strip().split()
                if j < 25:
                    bodies_data[body_id][f_idx, j, :] = [float(joint_info[0]), float(joint_info[1])]
                cursor += 1

    return list(bodies_data.values())

def get_yolo_mapping(keypoints):
    pts = np.zeros((13, 2))
    
    # 0: Голова 
    pts[0] = np.mean(keypoints[0:5, :2], axis=0)
    pts[1:13] = keypoints[5:17, :2]
    
    # ФІКС ОСІ Y: Перевертаємо скелет, щоб він збігався з 3D-камерою Kinect
    pts[:, 1] = -pts[:, 1]
    
    return pts

def get_ntu_mapping(landmarks):
    lm = landmarks
    pts = np.zeros((13, 2))
    
    def get_xy(idx): 
        return np.array([lm[idx][0], lm[idx][1]])

    # 0: Голова (NTU: 4 - head)
    pts[0] = get_xy(3) 

    # ТОРС (Плечі та Стегна)
    pts[1] = get_xy(4)  # 1: Ліве плече 
    pts[2] = get_xy(8)  # 2: Праве плече 
    pts[7] = get_xy(12) # 7: Ліве стегно 
    pts[8] = get_xy(16) # 8: Праве стегно 

    # РУКИ (Лікті та Зап'ястя)
    pts[3] = get_xy(5)  # 3: Лівий лікоть 
    pts[4] = get_xy(9)  # 4: Правий лікоть 
    pts[5] = get_xy(6)  # 5: Ліве зап'ястя 
    pts[6] = get_xy(10) # 6: Праве зап'ястя 

    # НОГИ (Коліна та Щиколотки)
    pts[9]  = get_xy(13) # 9: Ліве коліно 
    pts[10] = get_xy(17) # 10: Праве коліно 
    pts[11] = get_xy(14) # 11: Ліва щиколотка 
    pts[12] = get_xy(18) # 12: Права щиколотка 

    return pts

def normalize_sequence(sequence_points):
    pts = np.array(sequence_points) # Форма (T, 13, 2)
    
    # Індекси: 1(Л.плече), 2(П.плече), 7(Л.стегно), 8(П.стегно)
    torso_center = (pts[:, 1:2] + pts[:, 2:3] + pts[:, 7:8] + pts[:, 8:9]) / 4.0
    pts -= torso_center # Тепер центр тулуба завжди в (0,0)
    
    # Відстань від центру плечей до центру стегон
    shoulder_mid = (pts[:, 1] + pts[:, 2]) / 2.0
    hip_mid = (pts[:, 7] + pts[:, 8]) / 2.0
    
    torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=1, keepdims=True) # (T, 1)
    torso_length = np.maximum(torso_length, 0.05)
    torso_length = torso_length[:, np.newaxis, :] # Форма (T, 1, 1) для броадкастингу
    
    pts /= torso_length
        
    return pts 

def compute_kinematics(sequence):
    # sequence має форму (T, 13, 2)
    if sequence.shape[0] >= 5:
        smooth_seq = savgol_filter(sequence, window_length=5, polyorder=2, axis=0)
    else:
        smooth_seq = sequence
        
    vel = np.gradient(smooth_seq, axis=0)
    acc = np.gradient(vel, axis=0)

    np.clip(vel, -KINETIC_CLIP_BOUND, KINETIC_CLIP_BOUND, out=vel)
    np.clip(acc, -KINETIC_CLIP_BOUND, KINETIC_CLIP_BOUND, out=acc)
    
    vel /= GLOBAL_VEL_STD
    acc /= GLOBAL_ACC_STD
    
    # Склеюємо по останній осі (T, 13, 6)
    return np.concatenate([smooth_seq, vel, acc], axis=-1)

def prepare_sequences(features, seq_len=SEQUENCE_LENGTH, step=STEP):
    sequences = []
    num_frames = len(features)

    if num_frames < seq_len:
        return np.array(sequences)
    
    for i in range(0, num_frames - seq_len + 1, step):
        window = features[i : i + seq_len]
        sequences.append(window)
        
    return np.array(sequences)


def preprocess_dataset(skeleton_folder=VIDEO_DATA_DIR):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    files_with_paths = glob.glob(os.path.join(skeleton_folder, "**/*.skeleton"), recursive=True)
    
    batch_size = 20000  
    batch_count = 0
    
    for i in range(0, len(files_with_paths), batch_size):
        batch_files = files_with_paths[i : i + batch_size]
        norm_x, norm_y = [], []
        anom_x, anom_y = [], []
        
        batch_count += 1
        print(f"\nОбробка партії №{batch_count} ({i} - {i + batch_size})...")

        for filepath in tqdm(batch_files):
            file_name = os.path.basename(filepath)
            try:
                action_code = int(file_name.split("A")[1][:3])
                if action_code not in NORMAL_RANGE + ANOMALY_RANGE:
                    continue

                all_bodies = read_ntu_skeleton(filepath)
                for raw_data in all_bodies:
                    raw_mapped_seq = [get_ntu_mapping(p) for p in raw_data]
                    processed_seq = normalize_sequence(raw_mapped_seq)
                    full_features = compute_kinematics(processed_seq)
                    windows = prepare_sequences(full_features)

                    if len(windows) > 0:
                        x_part = windows[:, :HISTORY_LEN, :].astype(np.float32)
                        y_part = windows[:, HISTORY_LEN:SEQUENCE_LENGTH, :].astype(np.float32)
                        
                        if action_code in NORMAL_RANGE:
                            norm_x.append(x_part)
                            norm_y.append(y_part)
                        else:
                            anom_x.append(x_part)
                            anom_y.append(y_part)
            except:
                continue

        if norm_x:
            np.savez_compressed(os.path.join(OUTPUTS_DIR, f"norm_part_{batch_count}.npz"), 
                                x=np.concatenate(norm_x, axis=0), 
                                y=np.concatenate(norm_y, axis=0))
        if anom_x:
            np.savez_compressed(os.path.join(OUTPUTS_DIR, f"anomaly_part_{batch_count}.npz"), 
                                x=np.concatenate(anom_x, axis=0), 
                                y=np.concatenate(anom_y, axis=0))
        
        del norm_x, norm_y, anom_x, anom_y

if __name__ == "__main__":
    preprocess_dataset()