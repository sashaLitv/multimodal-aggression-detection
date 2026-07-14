from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import seaborn as sns
import queue
from matplotlib import rc
from src.modules.video.video_preprocess import video_preprocess
from src.modules.video import video_train
from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.config import HISTORY_LEN, SEQUENCE_LENGTH, STEP, CONSECUTIVE
# from src.config import THRESHOLD_FILE, YOLO_PATH, OUTPUT_PATH_FIGURES, DIST_PLOT_FILE, CM_PLOT_FILE
from src.config import font_small, font_main
rc('animation', html='jshtml')
sns.set_style("whitegrid")

class VideoDashboardWorker(QThread):
    buffer_ready_signal = Signal() 
    finished_signal = Signal(str)
    error_signal = Signal(str)
    change_pixmap_signal = Signal(QImage)
    metadata_signal = Signal(float, float, bool, float, int) 
    video_pred_signal = Signal(float)

    def __init__(self, video_path, model, yolo_model):
        super().__init__()
        
        self.video_path = video_path
        self.tracks_history = {}
        self.individual_losses = {}
        self.consecutive_anomalies = {}
        self.frame_buffer = queue.Queue(maxsize=100)
        self.video_fps = HISTORY_LEN

        self._is_running = True
        self._is_paused = False
        self._seek_request = None 
        
        self.THRESHOLD = video_train.load_threshold() 

        self.MAX_DISPLAY_ERROR = self.THRESHOLD * 3
        
        self.model = model
        self.yolo_model = yolo_model

        self.alarm_frames_left = 0
        self.ALARM_HOLD_DURATION = 30  
        self.display_loss = 0.0


    def set_paused(self, paused):
        self._is_paused = paused

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit("Не вдалося відкрити відео")
            return

        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0 or np.isnan(self.video_fps): 
            self.video_fps = 30

        TARGET_HEIGHT = 600 
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width_orig / height_orig if height_orig > 0 else 1.77
        
        TARGET_WIDTH = int(TARGET_HEIGHT * aspect_ratio)
        
        TOTAL_WIDTH = int(TARGET_HEIGHT * 2.8)

        self.tracks_history = {} 
        self.individual_losses = {}
        self.consecutive_anomalies = {}

        self.last_frame_time = time.time()
        self.history_buffer = deque(maxlen=30)
        frame_index = 0
        iou = 0.0
        num_people = 0
        boxes_all, track_ids, kpts_all = [], [], []

        while self._is_running and cap.isOpened():
            frame_index += 1

            if self._is_paused:
                time.sleep(0.05) 
                continue
                
            if self._seek_request is not None:
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                delta_frames = int(self._seek_request * self.video_fps)
                new_frame_pos = max(0, min(current_frame + delta_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
                
                self.tracks_history.clear()
                self.individual_losses.clear()
                self.consecutive_anomalies.clear()
                self._seek_request = None 

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                if hasattr(self, 'session_start_real_time'):
                    self.session_start_real_time = time.time()
                
                self._is_paused = True
                break
            try:
                results = self.yolo_model.track(
                    frame, persist=True, verbose=False, stream=True, 
                    conf=0.25, iou=0.45, classes=[0], tracker="bytetrack.yaml", 
                    imgsz=768, max_det=20, half=False
                )
                res = next(results)
                
                frame_main_raw = res.plot(labels=False, boxes=False)
                max_frame_loss = 0.0
                current_frame_ids = []
                alarm_triggered = False

                if res.keypoints is not None and len(res.keypoints.data) > 0 and res.boxes.id is not None:
                    boxes_all = res.boxes.xyxy.cpu().numpy()
                    track_ids = res.boxes.id.int().cpu().tolist()
                    kpts_all = res.keypoints.data.cpu().numpy()

                    real_height, real_width = frame.shape[:2] 

                    max_frame_loss, alarm_triggered, current_frame_ids, iou,  = self._analyze_frame_anomalies(
                        kpts_all, track_ids, frame_index, boxes_all, real_width, real_height
                    )
                    frame_main_raw = self._draw_detections(frame_main_raw, boxes_all, track_ids)

                self.tracks_history = {tid: h for tid, h in self.tracks_history.items() if tid in current_frame_ids}
                self.consecutive_anomalies = {tid: c for tid, c in self.consecutive_anomalies.items() if tid in current_frame_ids}

                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                self._update_alarm_state(max_frame_loss, alarm_triggered)
                is_bullying_now = self.alarm_frames_left > 0

                num_people = len(boxes_all)
                    
                self.metadata_signal.emit(float(self.display_loss), float(current_time_sec), bool(is_bullying_now), float(iou), int(num_people))
                self.video_pred_signal.emit(float(self.display_loss))

                frame_main = cv2.resize(frame_main_raw, (TARGET_WIDTH, TARGET_HEIGHT))
                canvas = np.full((TARGET_HEIGHT, TOTAL_WIDTH, 3), 255, dtype=np.uint8)
                canvas[:, :TARGET_WIDTH] = frame_main
                    
                img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                if is_bullying_now:
                    self.put_text_pil(img_pil, "АГРЕСІЯ!", (20, 20), font_main, (255, 0, 0))
                    
                self.draw_live_bar(img_pil, self.display_loss, self.THRESHOLD, TARGET_HEIGHT, self.MAX_DISPLAY_ERROR, TARGET_WIDTH)
                    
                final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                target_delay = (STEP + 1) / self.video_fps
                
                processing_time = time.time() - self.last_frame_time
                time_to_wait = target_delay - processing_time

                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                self.last_frame_time = time.time()

                final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                self._emit_frame(final_frame)

            except Exception as e:
                print(f"Помилка обробки кадру відео: {e}")
                self.error_signal.emit(f"Помилка обробки кадру: {str(e)}")
                break

        cap.release()
        self.tracks_history.clear()
        self.consecutive_anomalies.clear()
        self.finished_signal.emit(self.video_path)

    def _calculate_iou(self,box1, box2, depth_thresh=0.5):
        y_bottom1 = box1[3]
        y_bottom2 = box2[3]
        
        h_max = max(box1[3] - box1[1], box2[3] - box2[1])
        
        if abs(y_bottom1 - y_bottom2) > h_max * depth_thresh:
            return 0.0

        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top: return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection_area / float(box1_area + box2_area - intersection_area)
    
    def _analyze_frame_anomalies(self, kpts_all, track_ids, frame_index, boxes_all, frame_width, frame_height): 
        EDGE_MARGIN_WIDTH = frame_width * 0.15
        EDGE_MARGIN_HEIGHT = frame_height * 0.15
        global_max_iou = 0.0

        max_frame_loss = 0.0 
        alarm_triggered = False
        current_frame_ids = []

        global_max_iou = 0.0
        if len(boxes_all) > 1:
            for i in range(len(boxes_all)):
                for j in range(i + 1, len(boxes_all)):
                    current_iou = self._calculate_iou(boxes_all[i], boxes_all[j])
                    if current_iou > global_max_iou:
                        global_max_iou = current_iou

        inference_tracks = []
        inputs_list = []
        targets_list = []

        for person_kpts, track_id, box in zip(kpts_all, track_ids, boxes_all):
            x1, y1, x2, y2 = box
            
            is_on_edge = (x1 < EDGE_MARGIN_WIDTH) or (y1 < EDGE_MARGIN_HEIGHT) or \
                         (x2 > frame_width - EDGE_MARGIN_WIDTH) or (y2 > frame_height - EDGE_MARGIN_HEIGHT)
            
            valid_points_counts =  sum(1 for pt in person_kpts if pt[0]!= 0.0 or pt[1] != 0.0)
            missing_keypoints = valid_points_counts < len(person_kpts) * 0.8

            if(is_on_edge and missing_keypoints) or (valid_points_counts < 8):
                if track_id in self.tracks_history:
                    del self.tracks_history[track_id]
                    del self.consecutive_anomalies[track_id]
                continue

            if track_id not in self.tracks_history:
                self.tracks_history[track_id] = []
                self.consecutive_anomalies[track_id] = 0

            custom_pts = video_preprocess.get_yolo_mapping(person_kpts)
            self.tracks_history[track_id].append(custom_pts)
            current_frame_ids.append(track_id)

            if len(self.tracks_history[track_id]) > SEQUENCE_LENGTH:
                self.tracks_history[track_id].pop(0)
                    
            track_frames_alive = getattr(self, f'_track_age_{track_id}', 0) + 1
            setattr(self, f'_track_age_{track_id}', track_frames_alive)

            if len(self.tracks_history[track_id]) == SEQUENCE_LENGTH:
                if (track_frames_alive - SEQUENCE_LENGTH) % STEP == 0:
                    seq = np.array(self.tracks_history[track_id])
                    norm_seq = video_preprocess.normalize_sequence(seq)
                    full_feat = video_preprocess.compute_kinematics(norm_seq)
                                    
                    inputs_list.append(full_feat[:HISTORY_LEN])
                    targets_list.append(full_feat[HISTORY_LEN:])
                    inference_tracks.append(track_id)

        if inputs_list:
            batch_x = np.array(inputs_list, dtype=np.float32) 
            batch_y = np.array(targets_list, dtype=np.float32) 
            
            preds = self.model(batch_x, training=False)

            raw_loss_tensor = video_train.get_loss_top3_joints(batch_y, preds)
            raw_loss_batch = raw_loss_tensor.numpy() 

            is_anomaly_batch = video_train.detect_sequence_anomaly(raw_loss_batch, self.THRESHOLD, consecutive=CONSECUTIVE)

            for i, track_id in enumerate(inference_tracks):
                raw_loss = raw_loss_batch[i]
                is_anomaly = is_anomaly_batch[i]
                max_loss_val = float(np.max(raw_loss))

                if is_anomaly == 1:
                    print(f"\n[Кадр {frame_index}] АВТОЕНКОДЕР ПІДНЯВ ТРИВОГУ!")
                    print(f"Підозріла особа ID: {track_id}, Втрата (Loss): {max_loss_val:.2f} (Поріг: {self.THRESHOLD:.2f})")
                    
                    person_max_iou = 0.0
                    contact_id = None
                    track_idx_in_boxes = track_ids.index(track_id)
                    current_box = boxes_all[track_idx_in_boxes]

                    if len(boxes_all) == 1:
                        print("YOLO: В кадрі лише одна людина. Перекриття неможливе.")
                    else:
                        for j, other_box in enumerate(boxes_all):
                            if j != track_idx_in_boxes: 
                                other_id = track_ids[j]
                                iou_temp = self._calculate_iou(current_box, other_box)
                                
                                if iou_temp > person_max_iou:
                                    person_max_iou = iou_temp
                                    contact_id = other_id

                    if person_max_iou < 0.005:
                        print(f"Максимальне перекриття {person_max_iou:.4f} < 0.005.")
                        print(f"Це безпечний різкий рух (присідання/спорт). Глушимо тривогу.")
                        is_anomaly = 0
                        max_loss_val = min(max_loss_val, self.THRESHOLD - 0.001) 
                    else:
                        print(f"Є тісний контакт з ID {contact_id} (IoU = {person_max_iou:.4f}).")

                if is_anomaly == 1:
                    alarm_triggered = True
                    self.individual_losses[track_id] = max_loss_val
                    self.consecutive_anomalies[track_id] += 1 
                else:
                    self.individual_losses[track_id] = min(max_loss_val, self.THRESHOLD - 0.001)
                    self.consecutive_anomalies[track_id] = 0

        for track_id in current_frame_ids:
            current_loss = self.individual_losses.get(track_id, 0.0)
            if current_loss > max_frame_loss:
                max_frame_loss = current_loss

        return max_frame_loss, alarm_triggered, current_frame_ids, global_max_iou
    def _draw_detections(self, frame_main_raw, boxes_all, track_ids):
        for box, track_id in zip(boxes_all, track_ids):
            x1, y1, x2, y2 = box.astype(int)
            person_loss = self.individual_losses.get(track_id, 0.0)
                        
            if person_loss > self.THRESHOLD:
                color = (0, 0, 255)  # Червоний
            else:
                color = (0, 255, 0)  # Зелений
                
            label = f"ID: {track_id}"
                        
            cv2.rectangle(frame_main_raw, (x1, y1), (x2, y2), color, 4)
            
            cv2.putText(frame_main_raw, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(frame_main_raw, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
        return frame_main_raw

    def _update_alarm_state(self, max_frame_loss, alarm_triggered):
        if alarm_triggered:
            self.alarm_frames_left = self.ALARM_HOLD_DURATION
            self.display_loss = max_frame_loss
        elif self.alarm_frames_left > 0:
            self.alarm_frames_left -= 1
            self.display_loss = max(self.display_loss * 0.98, self.THRESHOLD + 0.0001)
        else:
            self.display_loss = self.display_loss * 0.8 + max_frame_loss * 0.2
    def _emit_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_data = rgb.tobytes()
        q_img = QImage(bytes_data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.change_pixmap_signal.emit(q_img)
    
    def stop(self):
        self._is_running = False
        self.wait()
    def toggle(self):
        self._is_paused = not self._is_paused
        return self._is_paused
    def seek_seconds(self, seconds):
        self._seek_request = seconds
    def put_text_pil(self, img, text, pos, font, color):
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, font=font, fill=color, stroke_width=2, stroke_fill='black')
    def draw_live_bar(self, img, value, threshold, height, max_error, x_offset):
        draw = ImageDraw.Draw(img)
        val_clamped = min(max(value, 0), max_error)
        
        bar_h = int((val_clamped / max_error) * height)
        color = (0, 255, 0) if value < threshold else (255, 0, 0)
        
        x_bar_left = x_offset + 10
        x_bar_right = x_bar_left + 50
        y_bar_top = height - bar_h
        y_bar_bottom = height
        
        draw.rectangle([x_bar_left, y_bar_top, x_bar_right, y_bar_bottom], fill=color)
        
        thresh_h = int((threshold / max_error) * height)
        thresh_y = height - thresh_h
        
        draw.line([x_offset, thresh_y, img.width, thresh_y], fill="yellow", width=6)
        
        if val_clamped >= threshold:
            percent_loss = 50.0 + ((val_clamped - threshold) / (max_error - threshold)) * 50.0
        else:
            percent_loss = (val_clamped / threshold) * 50.0
            
        text_x = x_bar_right + 20 
        
        text_thresh = "Поріг:\n50%" 
        pos_thresh = (text_x, thresh_y - 45)
        draw.multiline_text(pos_thresh, text_thresh, fill="yellow", font=font_small, stroke_width=2, stroke_fill='black')

        text_loss = f"Впевненість:\n{percent_loss:.1f}%"
        y_text_loss = max(10, y_bar_top - 20) 
        pos_loss = (text_x, y_text_loss)
        
        bbox_l = draw.multiline_textbbox(pos_loss, text_loss, font=font_small)
        draw.rectangle([bbox_l[0]-10, bbox_l[1]-10, bbox_l[2]+5, bbox_l[3]+10], fill='white')
        draw.multiline_text(pos_loss, text_loss, fill="black", font=font_small)


# def plot_training_history(history, save_path = OUTPUT_PATH_FIGURES + "/video_model_training_plot.png"):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.figure(figsize=(20, 8))
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Навчання (Train)')
#     plt.plot(history.history['val_loss'], label='Перевірка (Val)')
#     plt.title('Помилка (Loss)')
#     plt.xlabel('Епохи')
#     plt.ylabel('Помилка')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()

# def plot_confusion_matrix(cm, save_path=CM_PLOT_FILE, title=""):
#     plt.figure(figsize=(20, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=range(2), yticklabels=range(2))
#     plt.xlabel("Передбачений клас")
#     plt.ylabel("Cправжній клас")
#     plt.title(title)
#     plt.savefig(save_path)
#     print(f"Зображення збережено: {save_path}")
#     plt.show()
#     plt.close()

# def plot_real_distribution(errors_norm, errors_anom, threshold, save_path=DIST_PLOT_FILE):
#     plt.figure(figsize=(20, 8))
#     plt.style.use('bmh') 

#     data_min = min(errors_norm.min(), errors_anom.min())
#     data_max = max(errors_norm.max(), errors_anom.max())
    
#     x_grid = np.linspace(data_min, data_max, 1000)

#     kde_norm = gaussian_kde(errors_norm)
#     kde_anom = gaussian_kde(errors_anom)

#     y_norm_real = kde_norm(x_grid)
#     y_anom_real = kde_anom(x_grid)

#     plt.plot(x_grid, y_norm_real, label='Норма', color="#1fb447", lw=2)
#     plt.plot(x_grid, y_anom_real, label='Фізична агресія', color='#d62728', lw=2)

#     plt.hist(errors_norm, bins=50, density=True, alpha=0.3, color="#1fb447", label='Гістограма норми')
#     plt.hist(errors_anom, bins=50, density=True, alpha=0.3, color='#d62728', label='Гістограма фізичної агресії')

#     plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Поріг ({threshold:.4f})')


#     plt.fill_between(x_grid, y_norm_real, where=(x_grid >= threshold), 
#                      color="#1fb447", alpha=0.3, hatch='//', label='FP: Хибна тривога')
#     plt.fill_between(x_grid, y_anom_real, where=(x_grid <= threshold), 
#                      color='#d62728', alpha=0.3, hatch='\\\\', label='FN: Пропуск аномалії')

#     plt.title('Розподіл помилок реконструкції (KDE)', fontsize=14)
#     plt.xlabel('Середньоквадратична помилка (MSE)', fontsize=12)
#     plt.ylabel('Щільність ймовірності', fontsize=12)

#     plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
#     plt.grid(True, linestyle=':', alpha=0.6)

#     plt.savefig(save_path)
#     print(f"Зображення збережено: {save_path}")
#     plt.show()
#     plt.close()

# def run_full_eda(file_path, output_dir="outputs/figures"):
#     KEYPOINTS = [
#         "Ніс", "Л.Око", "П.Око", "Л.Вухо", "П.Вухо",
#         "Л.Плече", "П.Плече", "Л.Лікоть", "П.Лікоть",
#         "Л.Зап'ястя", "П.Зап'ястя", "Л.Стегно", "П.Стегно",
#         "Л.Коліно", "П.Коліно", "Л.Щиколотка", "П.Щиколотка"
#     ]
    
#     if not os.path.exists(file_path):
#         print(f"Помилка: Файл {file_path} не знайдено.")
#         return

#     data = np.load(file_path)
    
#     if 'X_norm' in data:
#         X_norm = data['X_norm']
#         X_anom = data['X_anom']
#     else:
#         print(f"Ключі у файлі: {list(data.files)}")
#         return
    
#     print(f"   Нормальні семпли: {X_norm.shape}")
#     print(f"   Аномальні семпли: {X_anom.shape}")

#     os.makedirs(output_dir, exist_ok=True)

#     def get_body_part_speed(X_data):
#         mean_vector = np.mean(np.abs(X_data), axis=(0, 1)) 
        
#         if mean_vector.shape[0] == 102:
#             velocity_part = mean_vector[34:68] 
#         elif mean_vector.shape[0] == 68:
#              velocity_part = mean_vector[34:68]
#         elif mean_vector.shape[0] == 34:
#             velocity_part = mean_vector
#         else:
#             velocity_part = mean_vector[:34]

#         if velocity_part.shape[0] == 34:
#             reshaped = velocity_part.reshape(17, 2)
         
#             speed = np.sqrt(reshaped[:, 0]**2 + reshaped[:, 1]**2)
#             return speed
        
#         return np.zeros(17)

#     norm_speed = get_body_part_speed(X_norm)
#     anom_speed = get_body_part_speed(X_anom)

#     x = np.arange(len(KEYPOINTS))
#     width = 0.35

#     plt.figure(figsize=(14, 6))
#     plt.bar(x - width/2, norm_speed, width, label='Нормальна поведінка', color='green', alpha=0.7)
#     plt.bar(x + width/2, anom_speed, width, label='Фізичний агресія', color='red', alpha=0.7)
    
#     plt.title("Середня кінетична активність частин тіла", fontsize=14)
#     plt.ylabel("Середня швидкість руху", fontsize=12)
#     plt.xticks(x, KEYPOINTS, rotation=45, ha="right")
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
    
#     save_path_1 = os.path.join(output_dir, "eda_1_body_activity.png")
#     plt.savefig(save_path_1)
#     plt.close() 

#     def get_sample_energy(sample_data):
#         if sample_data.shape[1] == 102:
#             dynamic_part = sample_data[:, 34:] 
#         else:
#             dynamic_part = sample_data
            
#         energy_per_frame = np.sum(np.abs(dynamic_part), axis=1) 
#         return energy_per_frame

#     if len(X_norm) > 0 and len(X_anom) > 0:
#         total_energy_anom = np.sum(np.abs(X_anom[:, :, 34:]), axis=(1, 2))
#         idx_anom = np.argmax(total_energy_anom)
        
#         idx_norm = np.random.randint(0, len(X_norm))
        
#         sample_norm = X_norm[idx_norm]
#         sample_anom = X_anom[idx_anom]

#         energy_norm = get_sample_energy(sample_norm)
#         energy_anom = get_sample_energy(sample_anom)

#         plt.figure(figsize=(12, 6))
        
#         plt.plot(energy_anom, label=f'Фізична агресія (Sample {idx_anom})', color='red', linewidth=2)
#         plt.plot(energy_norm, label=f'Норма (Sample {idx_norm})', color='green', linestyle='--', linewidth=2)
        
#         plt.title("Динаміка енергії руху в часі (один приклад)", fontsize=14)
#         plt.xlabel("Кадр (t)", fontsize=12)
#         plt.ylabel("Сумарна енергія (швидкість + прискорення)", fontsize=12)
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
        
#         save_path_2 = os.path.join(output_dir, "eda_2_single_sample_dynamics.png")
#         plt.savefig(save_path_2)
#         plt.close()
