import json
import os
from moviepy.editor import VideoFileClip, AudioFileClip

from src.config import COUNTS_FILE, DATASET_DIR

def get_total_count(dataset_type):
    dataset_path = os.path.join(DATASET_DIR, dataset_type)
    if not os.path.exists(dataset_path):
        return 0
    total_files = 0
    for _, _, files in os.walk(dataset_path):
        valid_files = [f for f in files if not f.startswith('.')]
        total_files += len(valid_files)
    return total_files
def get_saved_counts():
    if os.path.exists(COUNTS_FILE):
        try:
            with open(COUNTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}
def save_counts(counts_data):
    with open(COUNTS_FILE, 'w') as f:
        json.dump(counts_data, f)

def get_file_duration(file_path):
    try:
        if file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            clip = VideoFileClip(file_path)
        else:
            clip = AudioFileClip(file_path)
        
        duration = clip.duration
        clip.close() 
        return duration
    
    except Exception as e:
        print(f"Помилка з файлом {os.path.basename(file_path)}: {e}")
        return 0

def count_dataset_stats(name_dataset='audio'):
    total_dataset_time = 0
    classes_stats = {}

    if name_dataset == 'video':
        dataset_path = os.path.join(DATASET_DIR, "video")
    else:
        dataset_path = os.path.join(DATASET_DIR, "audio")


    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        class_time = 0
        file_count = 0
        
        files = [f for f in os.listdir(class_path) if not f.startswith('.')] 
        
        for file_name in files:
            file_path = os.path.join(class_path, file_name)
            duration = get_file_duration(file_path)
            
            class_time += duration
            file_count += 1
            
        classes_stats[class_name] = {
            "time_sec": class_time,
            "count": file_count
        }
        total_dataset_time += class_time

    for cls, stats in classes_stats.items():
        minutes = int(stats['time_sec'] // 60)
        seconds = int(stats['time_sec'] % 60)
        print(f"Клас: {cls.upper()}")
        print(f"  - Кількість файлів: {stats['count']}")
        print(f"  - Загальний час:    {minutes} хв {seconds} сек ({stats['time_sec']:.2f} с)")

    total_min = int(total_dataset_time // 60)
    total_sec = int(total_dataset_time % 60)
    print(f"ВСЬОГО ДАНИХ: {total_min} хв {total_sec} сек")