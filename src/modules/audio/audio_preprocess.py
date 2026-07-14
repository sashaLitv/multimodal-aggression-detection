import math  
import numpy as np
import librosa
import os
from tqdm import tqdm
from src.config import SAMPLES_PER_TRACK, SAMPLE_RATE, PROCESSED_AUDIO_PATH, DATASET_DIR

# --- АУГМЕНТАЦІЯ ТИМЧАСОВО ВИМКНЕНА ---
# def augment_audio(signal, sr):
#     augmented_signals = []
#     
#     try:
#         signal_pitch = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=2.0)
#         augmented_signals.append(signal_pitch)
#     except Exception as e: 
#         print(f"Помилка Pitch Shift: {e}")
#         pass
#
#     try:
#         noise = np.random.normal(0, 1, len(signal)) 
#         noise_amplitude = 0.005 * np.max(np.abs(signal)) 
#         signal_noise = signal + noise_amplitude * noise
#         augmented_signals.append(signal_noise)
#     except Exception as e: 
#         print(f"Помилка додавання шуму: {e}")
#         pass
#     
#     return augmented_signals

def process_signals(current_signal, sr, label, X, y, Signals, groups=None, group_id=None):
    if len(current_signal) == 0:
        return X, y, Signals
        
    MIN_REMAINING_SAMPLES = 2 * sr 
    
    start_sample = 0
    
    while start_sample < len(current_signal):
        chunk = current_signal[start_sample : start_sample + SAMPLES_PER_TRACK]
        
        if len(chunk) < SAMPLES_PER_TRACK:
            if len(chunk) >= MIN_REMAINING_SAMPLES:
                padding_needed = SAMPLES_PER_TRACK - len(chunk)
                chunk = np.pad(chunk, (0, padding_needed), mode='constant')
            else:
                break
                        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=128, fmax=8000
        )
             
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=1.0)
        data_point = mel_spectrogram_db[..., np.newaxis]

        X.append(data_point)
        y.append(label)
        Signals.append(chunk)
        if groups is not None:
            groups.append(group_id)
            
        start_sample += SAMPLES_PER_TRACK
        
    return X, y, Signals


def run(): 
    X_clean, y_clean, groups_clean = [], [], []
    X_pitch, y_pitch = [], []
    X_noise, y_noise = [], []

    label_map = {'normal': 0, 'anomal': 1} 
    output_folder = os.path.dirname(PROCESSED_AUDIO_PATH)
    os.makedirs(output_folder, exist_ok=True)

    for category, label in label_map.items():
        folder_path = os.path.join(DATASET_DIR, "audio", category)
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не знайдена")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.m4a', '.wav', '.mp3'))] 
        print(f"Клас '{category}': знайдено {len(files)} файлів ")


        for fname in tqdm(files):
            file_path = os.path.join(folder_path, fname)
            try:
                orig_signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                duration = len(orig_signal) / sr
                if duration <= 1.0:
                    continue
                
                group_id = f"{category}/{fname}"
                X_clean, y_clean, _ = process_signals(orig_signal, sr, label, X_clean, y_clean, [], groups=groups_clean, group_id=group_id)
                
                pitch_sig = librosa.effects.pitch_shift(y=orig_signal, sr=sr, n_steps=2.0)
                X_pitch, y_pitch, _ = process_signals(pitch_sig, sr, label, X_pitch, y_pitch, [])
                
                noise_amp = 0.3 * np.max(np.abs(orig_signal))
                noise_sig = orig_signal + noise_amp * np.random.normal(0, 1, len(orig_signal))
                X_noise, y_noise, _ = process_signals(noise_sig, sr, label, X_noise, y_noise, [])

            except Exception as e:
                print(f"Помилка {fname}: {e}")

    X_clean = (np.array(X_clean) + 80) / 80.0
    X_pitch = (np.array(X_pitch) + 80) / 80.0
    X_noise = (np.array(X_noise) + 80) / 80.0
    
    np.savez(PROCESSED_AUDIO_PATH, 
             X_clean=X_clean, y_clean=np.array(y_clean), groups=np.array(groups_clean),
             X_pitch=X_pitch, y_pitch=np.array(y_pitch),
             X_noise=X_noise, y_noise=np.array(y_noise))
    
    print(f"\nЗбережено: {len(X_clean)} чистих, {len(X_pitch)} пітч, {len(X_noise)} шумів.")

if __name__ == "__main__":
    run()