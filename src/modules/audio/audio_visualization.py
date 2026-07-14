import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
from tqdm import tqdm
from IPython.display import Audio, display 
from PySide6.QtCore import QThread, Signal
from tensorflow.keras.models import load_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src.config import DATASET_DIR, OUTPUTS_DIR, SAMPLE_RATE, AUDIO_THRESHOLD

try:
    from src.modules.audio import audio_preprocess
except ImportError:
    pass

def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Навчання (Train)')
    plt.plot(history.history['val_loss'], label='Перевірка (Val)')
    plt.title('Помилка (Loss)')
    plt.xlabel('Епохи')
    plt.ylabel('Помилка')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Навчання (Train)')
    plt.plot(history.history['val_accuracy'], label='Перевірка (Val)')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"Графік найкращої моделі збережено як '{save_path}'")
    plt.close()

def get_loudest_file(folder_path, duration=3.0):
    files = [f for f in os.listdir(folder_path) if f.endswith('.m4a')]
    
    if not files:
        print(f"У папці {folder_path} немає .m4a файлів!")
        return None

    max_energy = -1
    best_file = None

    
    for f in tqdm(files):
        path = os.path.join(folder_path, f)
        try:
            y, sr = librosa.load(path, duration=duration)
            energy = np.mean(librosa.feature.rms(y=y))
            
            if energy > max_energy:
                max_energy = energy
                best_file = f
        except Exception as e:
            print(f"Помилка читання {f}: {e}")
            continue

    return best_file

def get_typical_file(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.m4a')]
    
    if not files:
        print(f"У папці {folder_path} немає .m4a файлів!")
        return None
    
    selected_file = random.choice(files) 
    return selected_file

def visualize_spectrograms():
    normal_dir = os.path.join(DATASET_DIR, "normal")
    anomal_dir = os.path.join(DATASET_DIR, "anomal")
    
    best_norm = get_typical_file(normal_dir)
    best_anom = get_loudest_file(anomal_dir)
    
    if not best_norm or not best_anom:
        return

    path_norm = os.path.join(normal_dir, best_norm)
    path_anom = os.path.join(anomal_dir, best_anom)

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    y_norm, sr_norm = librosa.load(path_norm)
    S = librosa.feature.melspectrogram(y=y_norm, sr=sr_norm, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=1.0)
    
    librosa.display.specshow(S_dB, sr=sr_norm, x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f'Норма')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    y_anom, sr_anom = librosa.load(path_anom)
    S = librosa.feature.melspectrogram(y=y_anom, sr=sr_anom, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=1.0)
    
    librosa.display.specshow(S_dB, sr=sr_anom, x_axis='time', y_axis='mel', fmax=8000)
    plt.title(f'ВЕРБАЛЬНА АГРЕСІЯ')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUTS_DIR, "best_audio_spectrograms.png")
    plt.savefig(save_path)
    print(f"Графік збережено: {save_path}")
    plt.show()

    print(f"\nНОРМА")
    display(Audio(data=y_norm, rate=sr_norm))
    
    print(f"\nВЕРБАЛЬНА АГРЕСІЯ")
    display(Audio(data=y_anom, rate=sr_anom))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()
        self.current_segment_idx = -1
        self.last_drawn_pred = None  
        self.user_role = "Аналітик"

    def init_plot(self):
        self.ax.clear()
        
        if self.user_role == "Аналітик":
            self.ax.set_xlim(0, 3.0)
            self.ax.set_ylim(0, 8000)
        else:
            self.ax.axis('off')
            
        self.line = self.ax.axvline(x=0, color='cyan', linewidth=3)
        self.draw()

    def sync_ui(self, current_time, segment_signal, pred):
        if segment_signal is None:
            return

        seg_idx = int(current_time / 3.0)
        
        is_new_segment = (seg_idx != self.current_segment_idx)
        is_new_prediction = (pred != self.last_drawn_pred)

        if is_new_segment or is_new_prediction:
            self.current_segment_idx = seg_idx
            self.last_drawn_pred = pred
            
            self.ax.clear()
            
            if self.user_role == "Аналітик":
                self.plot_spectrogram(segment_signal) 
            else:
                self.plot_waveform(segment_signal, pred)
            
            self.line = self.ax.axvline(x=current_time % 3.0, color='cyan', linewidth=3)
            self.draw()
            
        self.update_cursor(current_time)
        
    def plot_waveform(self, signal, pred):
        if pred == -1.0:
            color = "#f7f7f7"  
        else:
            color = '#ff4757' if pred >= AUDIO_THRESHOLD else '#2ed573'

        self.ax.set_facecolor(color)
        self.ax.patch.set_alpha(0.1) 
        
        time_axis = np.linspace(0, 3.0, len(signal[::100]))
        self.ax.plot(time_axis, signal[::100], color='#2c3e50', linewidth=1)
        
        self.ax.set_xlim(0, 3.0)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.axis('off')

    def plot_spectrogram(self, signal):
        S = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=1.0)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=self.ax, cmap='magma')
        
        self.ax.set_title("Аналіз сегмента", fontsize=10)
        self.ax.set_xlim(0, 3.0) 

    def update_cursor(self, time_pos):
        relative_pos = time_pos % 3.0 
        
        if hasattr(self, 'line') and self.line is not None:
            self.line.set_xdata([relative_pos, relative_pos])
            self.draw_idle() 

class AudioAnalysisWorker(QThread):
    finished_signal = Signal(object, object, object, object, float) 
    error_signal = Signal(str)
    segment_finished_signal = Signal(int, float, object)
    metadata_signal = Signal(dict)
    
    def __init__(self, file_path, model):
        super().__init__()
        self.file_path = file_path
        self._is_running = True
        self._is_paused = False
        self._seek_request = None
        self.model = model
        
        
    def set_paused(self, paused):
        self._is_paused = paused
    def seek_seconds(self, seconds):
        self._seek_request = seconds

    def run(self):
        segment_duration = 3.0 
        try:
            full_signal, sr = librosa.load(self.file_path, sr=SAMPLE_RATE)
            duration = librosa.get_duration(y=full_signal, sr=sr)
            
            self.metadata_signal.emit({
                "name": os.path.basename(self.file_path), 
                "duration": duration
            })

            X_raw, _, Signals_raw = audio_preprocess.process_signals(
                full_signal, sr, label="unknown", X=[], y=[], Signals=[]
            )
            X = np.array(X_raw)
            X = (X + 80) / 80.0 
            audio_segments = np.array(Signals_raw)

            all_preds = []
            i = 0 
            num_segments = len(X)

            session_start_real_time = time.time()

            while i < num_segments and self._is_running:

                if self._seek_request is not None:
                    current_time = i * segment_duration
                    new_time = max(0, min(current_time + self._seek_request, duration))
                    
                    i = int(new_time / segment_duration)
                    self._seek_request = None
                    
                    session_start_real_time = time.time() - (i * segment_duration)
                
                if self._is_paused:
                    session_start_real_time = time.time() - (i * segment_duration)
                    time.sleep(0.05) 
                    continue

                self.segment_finished_signal.emit(i, -1.0, audio_segments[i])
                chunk_pred = self.model.predict(X[i:i+1], verbose=0)[0][0]
                self.segment_finished_signal.emit(i, float(chunk_pred), audio_segments[i])

                if i < len(all_preds):
                    all_preds[i] = chunk_pred
                else:
                    all_preds.append(chunk_pred)

                i += 1 
            
                expected_time_from_start = i * segment_duration
                elapsed_real_time = time.time() - session_start_real_time
                
                sleep_time = expected_time_from_start - elapsed_real_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.finished_signal.emit(
                np.array(all_preds), 
                audio_segments, 
                full_signal, 
                sr,
                duration
            )

        except Exception as e:
            print(f"Критична помилка в AudioWorker: {e}")
            self.error_signal.emit(f"Помилка: {str(e)}")