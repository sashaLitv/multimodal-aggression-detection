# def fake_fusion(video_loss, audio_loss, context_data=None):
#     weight_v = 0.5
#     weight_a = 0.5

#     loss = (video_loss * weight_v) + (audio_loss * weight_a)
#     pred = 1 if loss > 0.5 else 0
    
#     return loss, pred

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping

from src.config import FUSION_DATA_DIR, FUSION_MODEL_PATH


class MultimodalFusionMLP:
    def __init__(self, csv_path=os.path.join(FUSION_DATA_DIR, "fusion_dataset.csv"), model_path=FUSION_MODEL_PATH):
        self.CSV_PATH = csv_path
        self.MODEL_PATH = model_path
        
        self.scaler = StandardScaler()
        self.model = self._build_mlp()

    def _build_mlp(self):
        model = Sequential([
            InputLayer(shape=(4,), name='fusion_input'),

            Dense(16, activation='relu', name='hidden_1'),
            Dropout(0.4),
            
            Dense(8, activation='relu', name='hidden_2'),
            
            Dense(1, activation='sigmoid', name='fusion_output')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        if not os.path.exists(self.CSV_PATH):
            print(f"Помилка: Файл {self.CSV_PATH} не знайдено!")
            return

        df = pd.read_csv(self.CSV_PATH, low_memory=False)

        if len(df) < 20:
            print("Увага: Дуже мало даних для навчання. Зберіть більше зразків у fusion_dataset.csv.")
            return

        df = df[df['label'] != 'label']
        df = df[df['label'].notna()]
        df = df[df['video_name'].notna()]

        num_cols = ['audio_prob', 'video_prob', 'iou', 'num_people', 'label']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=num_cols)

        df['label'] = df['label'].astype(int)

        X = df[['audio_prob', 'video_prob', 'iou', 'num_people']].values
        y = df['label'].values
        groups = df['video_name'].values.astype(str)

        has_types = 'type' in df.columns

        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = kf.split(X, y, groups)

        accuracies, precisions, recalls, f1_scores = [], [], [], []
        audio_accs, audio_precs, audio_recs, audio_f1s = [], [], [], []
        video_accs, video_precs, video_recs, video_f1s = [], [], [], []
        
        weights_0, weights_1 = [], []
        test_videos_count, test_frames_count = [], []
        
        all_types, all_labels  = [], []
        all_audio_preds, all_video_preds, all_mlp_preds = [], [], []
        fold_no = 1
        best_loss = float('inf')
        total_videos = len(np.unique(groups))

        for train_index, val_index in splits:
            print(f"\n--- Fold {fold_no} / 5 ---")

            X_train, X_val = X[train_index],  X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            val_groups = groups[val_index]

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            weights_0.append(class_weights[0])
            weights_1.append(class_weights[1])

            test_videos_count.append(len(np.unique(val_groups)))
            test_frames_count.append(len(y_val))

            # Оцінка Аудіо
            y_pred_audio = (X_val[:, 0] > 0.5).astype(int)
            a_prec = precision_score(y_val, y_pred_audio, zero_division=0)
            a_rec = recall_score(y_val, y_pred_audio, zero_division=0)
            a_f1 = 2 * (a_prec * a_rec) / (a_prec + a_rec) if (a_prec + a_rec) > 0 else 0
            audio_accs.append(accuracy_score(y_val, y_pred_audio))
            audio_precs.append(a_prec); audio_recs.append(a_rec); audio_f1s.append(a_f1)

            # Оцінка Відео
            y_pred_video = (X_val[:, 1] > 0.5).astype(int)
            v_prec = precision_score(y_val, y_pred_video, zero_division=0)
            v_rec = recall_score(y_val, y_pred_video, zero_division=0)
            v_f1 = 2 * (v_prec * v_rec) / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0
            video_accs.append(accuracy_score(y_val, y_pred_video))
            video_precs.append(v_prec); video_recs.append(v_rec); video_f1s.append(v_f1)

            # Навчання Злиття (MLP)
            model = self._build_mlp()
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=16,
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                callbacks=[early_stopping],
                verbose=0
            )

            loss, acc = model.evaluate(X_val, y_val, verbose=0)
            y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            accuracies.append(acc); precisions.append(prec); recalls.append(rec); f1_scores.append(f1)

            print(f"Fusion Точність: {acc*100:.2f}% | Precision: {prec:.2f} | Recall: {rec:.2f}")

            if has_types:
                all_types.extend(df['type'].values[val_index])
            all_labels.extend(y_val)
            all_audio_preds.extend(y_pred_audio)
            all_video_preds.extend(y_pred_video)
            all_mlp_preds.extend(y_pred)

            if loss < best_loss:
                best_loss = loss
                model.save_weights(self.MODEL_PATH)
                print(f"   [Збережено найкращу модель злиття на Fold {fold_no}]")
                
            fold_no += 1
            if fold_no > 5:
                break

        print("\n" + "-"*80)
        print("ПАРАМЕТРИ ДАТАСЕТУ ТА НАВЧАННЯ (УСЕРЕДНЕНО)")
        print("-"*80)
        print(f"Загальний розмір датасету: {total_videos} унікальних відео ({len(df)} кадрів)")
        print(f"Середній розмір тестової вибірки (на 1 фолд): ~{int(np.mean(test_videos_count))} відео (~{int(np.mean(test_frames_count))} кадрів)")
        print(f"Динамічні ваги класів: Норма(0) ≈ {np.mean(weights_0):.2f} | Агресія(1) ≈ {np.mean(weights_1):.2f}")
        
        if has_types:
            aggression_frames = len(df[df['label'] == 1])
            types_count = df[df['label'] == 1]['type'].value_counts()
            print("\nСтруктура агресії у датасеті:")
            for t, count in types_count.items():
                if t != '-':
                    print(f" - {t:<10}: {count} кадрів ({count/aggression_frames*100:.1f}%)")

        print("\n" + "-"*80)
        print("ПОРІВНЯЛЬНИЙ АНАЛІЗ МОДУЛІВ (СЕРЕДНЄ ПО 5 ФОЛДАХ)")
        print("-"*80)
        print(f"{'Метрика':<15} | {'Тільки Аудіо':<15} | {'Тільки Відео':<15} | {'Злиття (MLP)':<15}")
        print("-" * 80)
        print(f"{'Accuracy':<15} | {np.mean(audio_accs)*100:6.2f}% ±{np.std(audio_accs)*100:5.2f}% | {np.mean(video_accs)*100:6.2f}% ±{np.std(video_accs)*100:5.2f}% | {np.mean(accuracies)*100:6.2f}% ±{np.std(accuracies)*100:5.2f}%")
        print(f"{'Precision':<15} | {np.mean(audio_precs)*100:6.2f}% ±{np.std(audio_precs)*100:5.2f}% | {np.mean(video_precs)*100:6.2f}% ±{np.std(video_precs)*100:5.2f}% | {np.mean(precisions)*100:6.2f}% ±{np.std(precisions)*100:5.2f}%")
        print(f"{'Recall':<15} | {np.mean(audio_recs)*100:6.2f}% ±{np.std(audio_recs)*100:5.2f}% | {np.mean(video_recs)*100:6.2f}% ±{np.std(video_recs)*100:5.2f}% | {np.mean(recalls)*100:6.2f}% ±{np.std(recalls)*100:5.2f}%")
        print(f"{'F1-Score':<15} | {np.mean(audio_f1s)*100:6.2f}% ±{np.std(audio_f1s)*100:5.2f}% | {np.mean(video_f1s)*100:6.2f}% ±{np.std(video_f1s)*100:5.2f}% | {np.mean(f1_scores)*100:6.2f}% ±{np.std(f1_scores)*100:5.2f}%")
        
        if has_types:
            print("\n" + "-"*80)
            print("ДЕТАЛІЗОВАНА ПОВНОТА (RECALL) ЗА ТИПАМИ АГРЕСІЇ ДЛЯ КОЖНОГО МОДУЛЯ")
            print("-"*80)
            print(f"{'Тип агресії':<7} | {'Кадрів':<8} | {'Тільки Відео':<15} | {'Тільки Аудіо':<15} | {'Злиття (MLP)':<15}")
            print("-" * 80)
            
            df_res = pd.DataFrame({
                'type': all_types,
                'label': all_labels,
                'audio_pred': all_audio_preds,
                'video_pred': all_video_preds,
                'mlp_pred': all_mlp_preds
            })
            
            aggression_types = ['Фізична', 'Вербальна', 'Змішана']
            
            for agg_type in aggression_types:
                subset = df_res[df_res['type'] == agg_type]
                if len(subset) == 0:
                    continue
                    
                total_sub_frames = len(subset)
                
                rec_video = (subset['video_pred'].sum() / total_sub_frames) * 100
                rec_audio = (subset['audio_pred'].sum() / total_sub_frames) * 100
                rec_mlp = (subset['mlp_pred'].sum() / total_sub_frames) * 100
                
                print(f"{agg_type:<10} | {total_sub_frames:<8} | {rec_video:>10.1f}%      | {rec_audio:>10.1f}%      | {rec_mlp:>10.1f}%")
            print("-"*80)

    def load_model(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                dummy_input = np.zeros((1, 4), dtype=np.float32)
                self.model(dummy_input)
                
                self.model.load_weights(self.MODEL_PATH)
                print(f"Модель злиття завантажена з {self.MODEL_PATH}")
                return True
            except Exception as e:
                print(f"Помилка завантаження моделі: {e}")
                return False
        else:
            print(f"Файл моделі {self.MODEL_PATH} не знайдено. Використовується простий фьюжн.")
            return False

    def predict(self, audio_prob, video_loss, iou, num_people):
        features = np.array([[audio_prob, video_loss, iou, num_people]])
        
        if hasattr(self.model, 'weights') and len(self.model.weights) > 0:
                probability = float(self.model.predict(features, verbose=0)[0][0])
        else:
            probability = (float(audio_prob) * 0.45) + (float(video_loss) * 0.55)
            probability = min(max(probability, 0.0), 1.0)
        
        raw_label = 1 if probability >= 0.5 else 0
        
        return probability, raw_label
    
    def debug_anomalous_videos(self):
        print("\n" + "-"*80)
        print("ВІЗУАЛЬНИЙ ДЕБАГ АНОМАЛЬНИХ ВІДЕО (ПОКАДРОВО)")
        print("Легенда: '.' - Норма (0), 'X' - Тривога (1)")
        print("-"*80)

        if not hasattr(self, 'model') or not hasattr(self.model, 'estimators_'):
            if not self.load_model():
                print("Помилка: Модель не знайдена. Спочатку навчіть її!")
                return

        df = pd.read_csv(self.CSV_PATH)
    
        videos = df['video_name'].unique()
        videos_tested = 0

        for video in videos:
            video_data = df[df['video_name'] == video]
            true_labels = video_data['label'].values
                
            videos_tested += 1
            video_data_smoothed = df[df['video_name'] == video]

            pred_labels = []

            for _, row in video_data_smoothed.iterrows():
                _, raw = self.predict(
                    row['audio_prob'], row['video_prob'], row['iou'], row['num_people']
                )
                pred_labels.append(raw)

            true_str = "".join(['X' if val == 1 else '.' for val in true_labels])
            pred_str = "".join(['X' if val == 1 else '.' for val in pred_labels])

            print(f"\nВідео: {video} ({len(true_labels)} сек)")
            print(f"Реальність : {true_str}")
            print(f"Прогноз    : {pred_str}")

            if 'X' not in pred_str and 'X' in true_str:
                print("   -> [КРИТИЧНО]")
            elif true_str != pred_str:
                print("   -> [УВАГА] Є розбіжності")

        print("\nДебаг завершено. Перевірено відео з агресією:", videos_tested)

if __name__ == "__main__":
    fusion = MultimodalFusionMLP()
    fusion.train()

    fusion.debug_anomalous_videos()
    