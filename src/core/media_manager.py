import os
import numpy as np
import subprocess
import tempfile
from PySide6.QtCore import QObject, Signal
from tensorflow.config import list_physical_devices
from ultralytics import YOLO
from src.modules.video import video_train
from tensorflow.keras.models import load_model
from PySide6.QtMultimedia import QMediaPlayer

from src.config import JOINT_COUNT, SAMPLE_RATE, VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, YOLO_PATH, DURATION
from src.core.fusion import MultimodalFusionMLP

class Task:
    def __init__(self, file_path, task_type, original_name):
        self.file_path = file_path
        self.task_type = task_type       
        self.original_name = original_name

class SmartMediaManager(QObject):
    # Сигнал готовності окремого аудіо-сегмента: (назва_файлу, індекс_сегмента, результат_моделі, аудіо_дані)
    audio_segment_ready = Signal(str, int, float, object)

    # Сигнал проміжного результату аналізу моделі: (назва_файлу, тип_моделі_video/audio, значення_втрат_loss)
    raw_pred_signal = Signal(str, str, float)

    # Сигнал передачі обробленого кадру відео в інтерфейс: (назва_файлу, об'єкт_QImage)
    video_frame_signal = Signal(str, object)

    # Сигнал передачі метаданих відео-аналізу: (назва_файлу, поточний_loss, час_у_сек, чи_є_тривога, iou, num_people)
    video_metadata_signal = Signal(str, float, float, bool, float, int)

    # Сигнал прийняття фінального рішення Fusion-модулем: (назва_файлу, комбінований_loss, статус_тривоги)
    fusion_decision_signal = Signal(str, float, bool)

    # Сигнал про виникнення помилки під час обробки: (назва_джерела, повідомлення_про_помилку)
    error_signal = Signal(str, str)

    # Сигнал про завершення всіх задач у черзі та зупинку активних воркерів
    all_tasks_finished = Signal()

    # Сигнал для глобальної синхронізації перемотки всіх медіа-потоків: (час_зміщення_у_секундах)
    global_seek_signal = Signal(float)

     # Сигнал для глобальної синхронізації паузи всіх медіа-потоків: (пауза/продовження)
    global_pause_signal = Signal(bool)
    
    def __init__(self):
        super().__init__()
        self.active_workers = {}
        self.task_queue = []
        self.sync_buffer = {}
        self.registered_audio_players = {}
        self.number_of_processes = self._determine_process_count()
        self._is_globally_paused = False

        self.fusion_model = MultimodalFusionMLP()
        self.fusion_model.load_model()

        try:
            self.yolo_model = YOLO(YOLO_PATH)
            A_norm = video_train.get_adjacency_matrix(JOINT_COUNT)
            self.video_model = video_train.build_model(A_norm)
            self.video_model.load_weights(VIDEO_MODEL_PATH)
            self.audio_model = load_model(AUDIO_MODEL_PATH, compile=False)
            self._warmup_all_models()
        except Exception as e:
            print(f"Помилка завантаження моделей: {e}")

        self.TRUE_LABEL = 0
        self.CSV_PATH = "fusion_dataset.csv"
        
        self.audio_memory = {}
        self.video_memory = {}

        if not os.path.exists(self.CSV_PATH):
            with open(self.CSV_PATH, "w") as f:
                f.write("video_name,time_sec,audio_prob,video_prob,label,iou,num_people,type\n")


    def _warmup_all_models(self):
        try:
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.yolo_model.predict(dummy_image, verbose=False)

            from src.config import HISTORY_LEN, FEATURE_DIM, JOINT_COUNT 
            dummy_video_data = np.zeros((1, HISTORY_LEN, JOINT_COUNT, FEATURE_DIM), dtype=np.float32)
            
            self.video_model.predict(dummy_video_data, verbose=0)

            dummy_audio_data = np.zeros((1, 128, 130, 1), dtype=np.float32)
            
            self.audio_model.predict(dummy_audio_data, verbose=0)

        except Exception as e:
            print(f"Помилка під час розігріву: {e}")

    def _determine_process_count(self):
        gpus = list_physical_devices('GPU')
        count = 4 if gpus else 2
        print(f"[{'GPU' if gpus else 'CPU'}] Виявлено прискорення. Потоків обробки: {count}")
        return count

    def has_audio(self, video_path):
        try:
            cmd = ['ffprobe', '-i', video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
            return bool(subprocess.check_output(cmd))
        except: return False

    def has_video(self, video_path):
        return video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    def _extract_audio(self, video_path):
        try:
            temp_audio = os.path.join(tempfile.gettempdir(), f"ext_{os.path.basename(video_path)}.wav")
            cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', str(SAMPLE_RATE), temp_audio]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return temp_audio
        except: return None

    def _dispatch(self):
        while len(self.active_workers) < self.number_of_processes and self.task_queue:
            task = self.task_queue.pop(0)
            self._run_worker(task)

    def _run_worker(self, task):
        try:
            name = task.original_name
            if task.task_type == 'video':
                from src.modules.video.video_visualization import VideoDashboardWorker
                worker = VideoDashboardWorker(task.file_path, self.video_model, self.yolo_model)
                
                worker.change_pixmap_signal.connect(
                    lambda img, n=name: self.video_frame_signal.emit(n, img)
                )
                worker.video_pred_signal.connect(
                    lambda s, n=name: self._process_video_pred(n, s)
                )
                worker.metadata_signal.connect(
                    lambda loss, time_sec, ok, iou, num_people, n=name: self.video_metadata_signal.emit(n, loss, time_sec, ok, iou, num_people)
                )
                worker.metadata_signal.connect(
                    lambda loss, time_sec, ok, iou, num_people, n=name: self._save_video_memory(n, loss, time_sec, iou, num_people)
                )
                worker.finished_signal.connect(
                    lambda n=name: self._export_fusion_dataset(n)
                )

            else:
                from src.modules.audio.audio_visualization import AudioAnalysisWorker
                worker = AudioAnalysisWorker(task.file_path, self.audio_model)

                worker.segment_finished_signal.connect(
                    lambda idx, pred, chunk, n=name: self.audio_segment_ready.emit(n, idx, pred, chunk)
                )
                worker.segment_finished_signal.connect(
                    lambda idx, pred, chunk, n=name: self._save_audio_memory(n, idx, pred)
                )
                worker.segment_finished_signal.connect(
                    lambda idx, pred, chunk, n=name: self._process_audio_pred(n, pred)
                )


            worker.finished.connect(lambda w_id=id(worker): self._on_worker_finished(w_id))
            worker.error_signal.connect(lambda err, n=name: self.error_signal.emit(n, err))

            self.active_workers[id(worker)] = worker
            worker.start()

        except Exception as e:
            self.error_signal.emit(task.original_name, f"Помилка запуску задачі: {str(e)}")
    
    def _process_video_pred(self, name, pred):
        self.raw_pred_signal.emit(name, "video", pred)
        
        if name not in self.sync_buffer:
            self.sync_buffer[name] = {"v": pred, "a": 0.0, "iou": 0.0, "num_people": 0}
        else:
            self.sync_buffer[name]["v"] = pred
        
        self._check_and_run_fusion(name)
    def _process_audio_pred(self, name, pred):
        if pred == -1.0:
            return
        self.raw_pred_signal.emit(name, "audio", pred)
        
        if name not in self.sync_buffer:
            self.sync_buffer[name] = {"v": 0.0, "a": pred}
        else:
            self.sync_buffer[name]["a"] = pred
        
        self._check_and_run_fusion(name)

    def _check_and_run_fusion(self, name):
        buf = self.sync_buffer.get(name)
        if buf and "v" in buf and "a" in buf and "iou" in buf and "num_people" in buf:
            
            prob, alert_label = self.fusion_model.predict(
                audio_prob=buf["a"],
                video_loss=buf["v"],
                iou=buf["iou"],
                num_people=buf["num_people"]
            )
            
            self.fusion_decision_signal.emit(name, float(prob), bool(alert_label == 1))

    def ingest_files(self, file_paths):
        if len(file_paths) == 0:
            self.error_signal.emit("Завантаження файлів", "Немає вибраних файлів.")
            return
            
        if len(file_paths) > self.number_of_processes: 
            self.error_signal.emit("Завантаження файлів", "Максимальна кількість файлів - 4.")
                
            file_paths = file_paths[:self.number_of_processes]
        
        for path in file_paths:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.task_queue.append(Task(path, 'video', path))
            
            if self.has_audio(path):
                audio_temp_path = self._extract_audio(path)
                if audio_temp_path:
                    self.task_queue.append(Task(audio_temp_path, 'audio', path))
            
            self._dispatch()

    def _on_worker_finished(self, worker_id):
        if worker_id in self.active_workers:
            worker = self.active_workers.pop(worker_id)
            
            if hasattr(worker, 'file_path') and 'ext_' in worker.file_path:
                try:
                    os.remove(worker.file_path)
                    print(f"Тимчасовий файл видалено: {worker.file_path}")
                except: pass
                
            worker.deleteLater()
        
        if not self.active_workers and not self.task_queue:
            self.all_tasks_finished.emit()
        else:
            self._dispatch()

    def set_global_volume(self, volume_level):
        for player in self.registered_audio_players.values():
            output = player.audioOutput()
            if output:
                output.setVolume(volume_level)
    def get_global_volume(self):
        for player in self.registered_audio_players.values():
            output = player.audioOutput()
            if output:
                return output.volume()
        return 1.0
    
    def register_audio_player(self, source_path, player):
        self.registered_audio_players[source_path] = player

    def seek_all(self, seconds):
        for worker in self.active_workers.values():
            worker.seek_seconds(seconds)
        self.global_seek_signal.emit(seconds)

        for player in self.registered_audio_players.values():
            current_pos_ms = player.position()
            new_pos_ms = max(0, current_pos_ms + int(seconds * 1000))
            if player.duration() > 0:
                new_pos_ms = min(new_pos_ms, player.duration() - 100)
            player.setPosition(new_pos_ms)
    def toggle_all(self):
        self._is_globally_paused = not self._is_globally_paused
        
        for worker in self.active_workers.values():
            if hasattr(worker, 'set_paused'):
                worker.set_paused(self._is_globally_paused)
        
        for player in self.registered_audio_players.values():
            if self._is_globally_paused:
                player.pause()
            else:
                if player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
                    player.setPosition(max(0, player.position() - 100))
                player.play()
        
        self.global_pause_signal.emit(self._is_globally_paused)
        return self._is_globally_paused
    def stop_all(self):
        self.task_queue.clear()
        for worker in self.active_workers.values():
            if hasattr(worker, 'stop'): worker.stop()
            worker.terminate()
        self.active_workers.clear()
        
        for player in self.registered_audio_players.values():
            player.stop()
        self.registered_audio_players.clear()


    def _save_audio_memory(self, name, idx, pred):
        if pred == -1.0:
            return
        if name not in self.audio_memory:
            self.audio_memory[name] = {}
        self.audio_memory[name][idx] = pred
    def _save_video_memory(self, name, video_loss, time_sec, iou=0.0, num_people=0): 
        if name not in self.sync_buffer:
            self.sync_buffer[name] = {"v": video_loss, "a": 0.0, "iou": iou, "num_people": num_people}
        else:
            self.sync_buffer[name].update({"v": video_loss, "iou": iou, "num_people": num_people})

        if name not in self.video_memory:
            self.video_memory[name] = {'last_log': 0.0, 'data': [], 'window': []}

        self.video_memory[name]['window'].append({
            'loss': video_loss,
            'iou': iou,
            'num_people': num_people
        })
        last_time = self.video_memory[name]['last_log']

        if time_sec - last_time >= 1.0:
            self.video_memory[name]['last_log'] = time_sec
            window = self.video_memory[name]['window']

            max_loss = max(item['loss'] for item in window)
            max_iou = max(item['iou'] for item in window)
            max_people = max(item['num_people'] for item in window)

            threshold = video_train.load_threshold()
            max_error = threshold * 3.0

            val_clamped = min(max(max_loss, 0), max_error)
            
            if val_clamped >= threshold:
                video_prob = 0.5 + ((val_clamped - threshold) / (max_error - threshold)) * 0.5
            else:
                video_prob = (val_clamped / threshold) * 0.5

            self.video_memory[name]['data'].append((time_sec, video_prob, max_iou, max_people))
            
            self.video_memory[name]['window'] = []

    def _export_fusion_dataset(self, name):
        print(f"\n[FUSION] Збереження датасету для {name}...")
        
        v_data = self.video_memory.get(name, {}).get('data', [])
        a_data = self.audio_memory.get(name, {})

        base_name = os.path.basename(name)
        suffix = "_anomal" if self.TRUE_LABEL == 1 else "_normal"
        final_video_name = f"{base_name}{suffix}"

        with open(self.CSV_PATH, "a") as f:
            for time_sec, video_prob, iou, num_people in v_data: 
                audio_idx = int(time_sec // DURATION)
                audio_prob = a_data.get(audio_idx, 0.0) 
                
                f.write(f"{final_video_name},{time_sec:.1f},{audio_prob:.4f},{video_prob:.4f},{self.TRUE_LABEL},{iou:.4f},{(num_people/20):.4f},-\n")

        print(f"[FUSION] Успішно додано {len(v_data)} рядків до {self.CSV_PATH}")