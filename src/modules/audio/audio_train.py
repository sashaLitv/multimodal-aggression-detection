import csv
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, InputLayer, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import class_weight

from src.config import AUDIO_MODEL_PATH, PROCESSED_AUDIO_PATH

def build_model(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        GaussianNoise(0.05),
        Conv2D(16, (3, 3), use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(16, (3, 3), use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(32),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.6), 
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def save_audio_report(y_test, y_pred, groups, output_file="audio_evaluation.csv"):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "prediction", "is_correct"])
        
        for i in range(len(y_test)):
            is_correct = 1 if y_test[i] == y_pred[i] else 0
            writer.writerow([groups[i], y_test[i], y_pred[i], is_correct])
            
    print(f"Звіт збережено у {output_file}")

def run(model_file=AUDIO_MODEL_PATH, fold_max=5):
    data = np.load(PROCESSED_AUDIO_PATH)
    
    X_clean = data['X_clean']
    X_pitch = data['X_pitch']
    X_noise = data['X_noise']
    y_clean = data['y_clean']
    y_pitch = data['y_pitch']
    y_noise = data['y_noise']
    groups = data['groups'] if 'groups' in data.files else None
    
   
    # X_clean = data['X_clean']
    # y_clean = data['y_clean']
    
    if 'groups' in data:
        groups = data['groups']
    else:
        groups = np.arange(len(y_clean))
    
    input_shape = X_clean.shape[1:]
    
    if groups is not None:
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = kf.split(X_clean, y_clean, groups)
        print("Валідація: split по файлах, без змішування сегментів одного аудіо.")
    else:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = kf.split(X_clean, y_clean)
        print("Увага: у processed-файлі немає groups, split іде по сегментах.")
    
    accuracies, losses = [], []
    fold_no = 1
    best_loss = float('inf')

    for train_index, val_index in splits:
        print(f"\nFold {fold_no} / 5 ...")
        
        X_val, y_val = X_clean[val_index], y_clean[val_index]
        
        X_train = np.concatenate([
            X_clean[train_index], 
            X_pitch[train_index], 
            X_noise[train_index]
        ])
        y_train = np.concatenate([
            y_clean[train_index], 
            y_pitch[train_index], 
            y_noise[train_index]
        ])

        # X_train = X_clean[train_index]
        # y_train = y_clean[train_index]
        
        
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        model = build_model(input_shape)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)


        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(weights))
        
        if fold_no == 1:
            print(f"Ваги для навчання: Норма(0)={class_weights_dict[0]:.2f}, Аномалія(1)={class_weights_dict[1]:.2f}")

        model = build_model(input_shape)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
        model.fit(
            X_train, y_train,
            epochs=100, 
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights_dict, 
            
            verbose=1 
        )
        
        preds = model.evaluate(X_val, y_val, verbose=0)
        y_pred_probs = model.predict(X_clean) 
        y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
        save_audio_report(y_clean, y_pred_classes, groups)

        current_loss = preds[0]
        current_acc = preds[1] * 100
        print(f"Точність: {current_acc:.2f}%, Loss: {current_loss:.4f}")

        accuracies.append(current_acc)
        losses.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            model.save(model_file)
            print(f"   [Збережено найкращу модель на Fold {fold_no}]")
            
        fold_no += 1
        if fold_no > fold_max:
            break

    print("СТАТИСТИКА ПО ВСІХ ФОЛДАХ - валідаційний набір")
    print(f"Точність (mean ± std): {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    print(f"Втрата Loss (mean ± std): {np.mean(losses):.4f} ± {np.std(losses):.4f}")
