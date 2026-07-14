import os
import gc
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Lambda, Layer, Input, RepeatVector,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.config import HORIZON, JOINT_COUNT, HISTORY_LEN, FEATURE_DIM, THRESHOLD_FILE, OUTPUTS_DIR, VIDEO_MODEL_PATH, PROJECT_ROOT, CONSECUTIVE, W_POSS, W_VELL, W_ACC

class GraphConvolution(Layer):
    def __init__(self, units, adjacency_matrix=None, activation='relu', use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        # print(f"\n[INIT] Ініціалізація шару GraphConvolution(units={units}, use_bias={use_bias})")
        self.units = units
        self.use_bias = use_bias
        
        if adjacency_matrix is not None:
            self.A = tf.constant(adjacency_matrix, dtype=tf.float32)
            # print(f"[INIT] Отримано матрицю суміжності A з формою: {self.A.shape}")
        else:
            self.A = None
            # print("[INIT] Матрицю суміжності не передано (очікується в методі call)")
            
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # print(f"\n[BUILD] Виклик методу build(). Вхідна форма (input_shape): {input_shape}")
        feature_dim = input_shape[-1]
        # print(f"[BUILD] Визначено кількість вхідних фічей (остання вісь): {feature_dim}")
        
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel', 
            trainable=True
        )
        # print(f"[BUILD] Створено ваги (kernel) з формою: {self.kernel.shape}")
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias', 
                trainable=True
            )
            # print(f"[BUILD] Створено зміщення (bias) з формою: {self.bias.shape}")
        else:
            self.bias = None
            # print("[BUILD] Зміщення (bias) вимкнено.")
            
        super(GraphConvolution, self).build(input_shape)
        # print("[BUILD] Метод build() успішно завершено.")

    def call(self, inputs, training=None, adjacency_matrix=None):
        # print(f"\n[CALL] ---> Початок прямого проходу (forward pass) <---")
        # print(f"[CALL] 0. Форма вхідного тензора (inputs): {inputs.shape}")
        
        A = adjacency_matrix if adjacency_matrix is not None else self.A
        if A is None:
            raise ValueError("Матрицю суміжності потрібно передати в __init__ або в call.")
        
        # Шаг 1: Лінійна трансформація
        # print(f"[CALL] 1. Виконуємо tensordot (множення фічей на ваги)...")
        x = tf.tensordot(inputs, self.kernel, axes=[[-1], [0]])
        # print(f"[CALL]    Результат після tensordot: {x.shape}")
        # print(f"[CALL]    (Очікується: Batch, Time, Joints, Units={self.units})")
        
        # Шаг 2: Агрегація по графу
        # print(f"[CALL] 2. Виконуємо агрегацію сусідів (einsum) з матрицею A {A.shape}...")
        output = tf.einsum('ij,btjc->btic', A, x)
        # print(f"[CALL]    Результат після einsum: {output.shape}")
        # print(f"[CALL]    (Очікується: Batch, Time, Joints, Units={self.units})")
        
        # Шаг 3: Зміщення та активація
        if self.use_bias:
            output += self.bias
            # print(f"[CALL] 3. Додано зміщення (bias)")
            
        output = self.activation(output)
        # print(f"[CALL] 4. Застосовано функцію активації: {self.activation.__name__}")
        # print(f"[CALL] ---> Завершення прямого проходу, вихідна форма: {output.shape} <---")
        
        return output

    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], input_shape[1], input_shape[2], self.units)
        # print(f"\n[COMPUTE_SHAPE] Обчислення вихідної форми: вхід {input_shape} -> вихід {out_shape}")
        return out_shape

    def get_config(self):
        # print("\n[CONFIG] Виклик get_config() для збереження архітектури шару.")
        config = super(GraphConvolution, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
        })
        return config


def get_adjacency_matrix(num_nodes=JOINT_COUNT):
    edges = [
        (0, 1), (0, 2),
        (1, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6),
        (1, 7), (2, 8),
        (7, 8),
        (7, 9),  (9, 11),
        (8, 10), (10, 12),
    ]

    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    degree = np.sum(A, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = np.dot(d_mat_inv_sqrt, np.dot(A, d_mat_inv_sqrt))
    return A_norm

def build_model(adjacency_matrix):
    inputs = Input(shape=(HISTORY_LEN, JOINT_COUNT, FEATURE_DIM), name="skeleton_input")

    gcn_out = GraphConvolution(
        units=64, 
        adjacency_matrix=adjacency_matrix, 
        activation='relu'
    )(inputs)

    flattened_time_steps = Reshape((HISTORY_LEN, JOINT_COUNT * 64))(gcn_out)

    encoded_state = LSTM(128, return_sequences=False, name="encoder_lstm")(flattened_time_steps)
    repeated_context = RepeatVector(HORIZON)(encoded_state)
    decoded_sequence = LSTM(128, return_sequences=True, name="decoder_lstm")(repeated_context)
    
    flat_predictions = TimeDistributed(Dense(JOINT_COUNT * FEATURE_DIM, activation='linear'))(decoded_sequence)

    outputs = Reshape((HORIZON, JOINT_COUNT, FEATURE_DIM), name="skeleton_output")(flat_predictions)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model
def get_weighted_error_map(y_true, y_pred):
    w_pos = W_POSS
    w_vel = W_VELL
    w_acc = W_ACC  

    err_pos = tf.square(y_true[:, :, 0:26]  - y_pred[:, :, 0:26])  * w_pos
    err_vel = tf.square(y_true[:, :, 26:52] - y_pred[:, :, 26:52]) * w_vel
    err_acc = tf.square(y_true[:, :, 52:78] - y_pred[:, :, 52:78]) * w_acc

    return tf.concat([err_pos, err_vel, err_acc], axis=-1)

def custom_weighted_loss(y_true, y_pred):
    return tf.reduce_mean(get_weighted_error_map(y_true, y_pred))

def get_loss_top3_joints(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    
    pos_true = y_true[:, :, :, 0:2]
    pos_pred = y_pred[:, :, :, 0:2]
    err_pos = tf.reduce_mean(tf.square(pos_true - pos_pred), axis=-1)
    
    vel_true = y_true[:, :, :, 2:4]
    vel_pred = y_pred[:, :, :, 2:4]
    err_vel = tf.reduce_mean(tf.square(vel_true - vel_pred), axis=-1)
    
    acc_true = y_true[:, :, :, 4:6]
    acc_pred = y_pred[:, :, :, 4:6]
    err_acc = tf.reduce_mean(tf.square(acc_true - acc_pred), axis=-1)
    
    w_pos = W_POSS
    w_vel = W_VELL
    w_acc = W_ACC  
    
    joint_err = (err_pos * w_pos) + (err_vel * w_vel) + (err_acc * w_acc)
    
    top3_err_vals, _ = tf.math.top_k(joint_err, k=3) 
    
    mean_top3_err = tf.reduce_mean(top3_err_vals, axis=-1) 
    
    return mean_top3_err

def detect_sequence_anomaly(frame_errors, threshold, consecutive=3):
    batch_size, seq_len = frame_errors.shape
    y_pred = np.zeros(batch_size)

    for i in range(batch_size):
        above = frame_errors[i] > threshold
        for j in range(seq_len - consecutive + 1):
            if np.all(above[j:j+consecutive]):
                y_pred[i] = 1
                break

    return y_pred

def calculate_threshold(model, X_val, y_val, percentile=98):
    print("Розрахунок порогу по батчах...")
    frame_errors_list = []
    batch_size = 64
    
    for i in range(0, len(X_val), batch_size):
        x_batch = X_val[i:i+batch_size]
        y_batch = y_val[i:i+batch_size]
        
        pred_batch = model.predict(x_batch, verbose=0)
        err_batch = get_loss_top3_joints(y_batch, pred_batch).numpy()
        
        window_errors = err_batch[:, 0]
        frame_errors_list.append(window_errors)

    frame_errors = np.concatenate(frame_errors_list, axis=0)
    
    threshold = float(np.percentile(frame_errors, percentile))
    print(f"Поріг ({percentile}-й перцентиль норми): {threshold:.6f}")

    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(threshold))
    return threshold

def evaluate_model(model, threshold, x_list, y_list, is_anomaly_list, consecutive=CONSECUTIVE):
    correct_anom = 0
    missed_anom = 0
    correct_norm = 0
    false_alarms = 0
    
    for i in range(len(x_list)):
        x_vid = x_list[i]
        y_vid = y_list[i]
        is_anomaly = is_anomaly_list[i]
        
        if len(x_vid) == 0: continue
        
        pred = model.predict(x_vid, batch_size=64, verbose=0)
        errors = get_loss_top3_joints(y_vid, pred).numpy()
        
        window_errors = errors[:, 0]
        above_threshold = window_errors > threshold

        trigger = False
        count = 0
        
        for val in above_threshold:
            if val:
                count += 1
                if count >= consecutive:
                    trigger = True
                    break
            else:
                count = 0
                
        if is_anomaly == 1:
            if trigger: correct_anom += 1
            else: missed_anom += 1
        else:
            if trigger: false_alarms += 1
            else: correct_norm += 1
                
    total_anoms = correct_anom + missed_anom
    total_norms = correct_norm + false_alarms
    
    
    print("\n" + "="*65)
    print("ОЦІНКА НА РІВНІ ПОДІЙ (EVENT-LEVEL EVALUATION)")
    print("="*65)
    
    print(f"Всього сцен з агресивною поведінкою (>= 32 кадри для спрацювання): {total_anoms}")
    if total_anoms > 0:
        recall_anom = (correct_anom / total_anoms) * 100  if total_anoms > 0 else 0
        precision_anom = (correct_anom / (correct_anom + false_alarms)) * 100 if (correct_anom + false_alarms) > 0 else 0
        print(f" Знайдено (True Positives): {correct_anom}")
        print(f" Пропущено (False Negatives): {missed_anom}")
        print(f" Повнота виявлення (Recall): {recall_anom:.1f}%")
        print(f" Точність тривоги (Precision): {precision_anom:.1f}%")
        
    print(f"\nВсього сцен з нормальною поведінкою (>= 32 кадри для спрацювання): {total_norms}")
    if total_norms > 0:
        recall_norm = (correct_norm / total_norms) * 100 if total_norms > 0 else 0
        precision_norm = (correct_norm / (correct_norm + missed_anom)) * 100 if (correct_norm + missed_anom) > 0 else 0
        print(f" Правильно проігноровано (True Negatives): {correct_norm}")
        print(f" Хибні тривоги (False Positives): {false_alarms}")
        print(f" Повнота виявлення  (Recall): {recall_norm:.1f}%")
        print(f" Точність на нормі (Precision): {precision_norm:.1f}%")
        
    return {"anom_detected": correct_anom, "false_alarms": false_alarms}


def load_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, 'r') as f:
            content = f.read().strip()
            return float(content) if content else 0.05
    return 15

class VideoDataManager:
    def __init__(self, file_list, batch_size=64):
        self.file_list = file_list
        self.batch_size = batch_size

    def generate_data(self, split='train'):

        if split == 'train':
            np.random.shuffle(self.file_list)

        for file in self.file_list:
            with np.load(file, mmap_mode='r') as data:
                x_full = data['x']
                y_full = data['y']

            n_samples = len(x_full)
            if n_samples == 0:
                continue

            indices = np.arange(n_samples)
            np.random.RandomState(42).shuffle(indices)

            train_end = int(n_samples * 0.80)
            val_end = train_end + int(n_samples * 0.1)

            if split == 'train':
                target_idx = indices[:train_end]
            else:
                target_idx = indices[train_end:val_end]

            x_target = x_full[target_idx]
            y_target = y_full[target_idx]

            if split == 'train':
                shuff_idx = np.random.permutation(len(x_target))
                x_target = x_target[shuff_idx]
                y_target = y_target[shuff_idx]

            for i in range(0, len(x_target), self.batch_size):
                yield (
                    x_target[i:i+self.batch_size].astype(np.float32),
                    y_target[i:i+self.batch_size].astype(np.float32)
                )

            del x_target
            del y_target
            gc.collect()

def run():
    BATCH_SIZE = 64
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints(Graph)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    DATA_DIR = os.path.join(OUTPUTS_DIR, "processed_video_parts(Graph)")

    all_norm = sorted(glob.glob(os.path.join(DATA_DIR, "norm_part_*.npz")))

    output_signature = (
        tf.TensorSpec(shape=(None, HISTORY_LEN, FEATURE_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(None, HORIZON,     FEATURE_DIM), dtype=tf.float32),
    )
    data_manager = VideoDataManager(all_norm, batch_size=BATCH_SIZE)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_manager.generate_data(split='train'), 
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_manager.generate_data(split='val'), 
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    input_shape = (HISTORY_LEN, FEATURE_DIM)

    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.weights.h5")))
    initial_epoch = 0
    
    if checkpoints:
        latest = checkpoints[-1]
        initial_epoch = int(latest.split("_")[-1].split(".")[0])
        model = build_model(input_shape)
        model.load_weights(latest)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_weighted_loss)
        print(f"Відновлено навчання з чекпоінту: {latest}")
    else:
        model = build_model(input_shape)

    
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_{epoch:02d}.weights.h5"),
        save_freq='epoch',
        save_weights_only=True,
        monitor='val_loss'
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        checkpoint_cb,
    ]

    total_train, total_val = 0, 0
    for f in all_norm:
        data_len = np.load(f, mmap_mode='r')['x'].shape[0]
        total_train += int(data_len * 0.80)
        total_val += int(data_len * 0.1)

    model.fit(
        train_dataset,
        steps_per_epoch=max(1, total_train // BATCH_SIZE),
        epochs=10, initial_epoch=initial_epoch,
        validation_data=val_dataset, validation_steps=max(1, total_val // BATCH_SIZE),
        callbacks=callbacks, verbose=1
    )

    weights_path = VIDEO_MODEL_PATH.replace('.keras', '.weights.h5')
    model.save_weights(weights_path, save_format='tf')
    print(f"Модель збережено у {weights_path}")

    return model

if __name__ == "__main__":
    DATA_DIR = os.path.join(OUTPUTS_DIR, "classes")
    all_norm = sorted(glob.glob(os.path.join(DATA_DIR, "norm_*.npz"))) 
    all_anom = sorted(glob.glob(os.path.join(DATA_DIR, "anom_*.npz")))

    test_x_list, test_y_list, test_paths = [], [], []
    anom_x_list, anom_y_list, anom_paths = [], [], []

    print("Збір нормальних даних...")
    for file in all_norm:
        file_name = os.path.basename(file) 
        with np.load(file) as data:  
            x_full, y_full = data['x'], data['y']
        
        n_samples = len(x_full)
        if n_samples == 0: continue
            
        val_end = int(n_samples * 0.90) 
        x_test_chunk = x_full[val_end:]
        y_test_chunk = y_full[val_end:]

        for i in range(0, len(x_test_chunk), 300):
            chunk_x = x_test_chunk[i:i+300]
            chunk_y = y_test_chunk[i:i+300]
            
            if len(chunk_x) >= 100:
                test_x_list.append(chunk_x.copy())
                test_y_list.append(chunk_y.copy())
                test_paths.append(file_name) 

    print("Збір аномальних даних...")
    for file in all_anom:
        file_name = os.path.basename(file)
        with np.load(file) as data:
            x_full, y_full = data['x'], data['y']
            
        if len(x_full) == 0: continue
        
        for i in range(0, len(x_full), 300):
            chunk_x = x_full[i:i+300]
            chunk_y = y_full[i:i+300]
            
            if len(chunk_x) >= 100:
                anom_x_list.append(chunk_x.copy())
                anom_y_list.append(chunk_y.copy())
                anom_paths.append(file_name) 

    print(f"Завантаження моделі ваг...")
    input_shape = (HISTORY_LEN, FEATURE_DIM)
    model = build_model(input_shape)
    weights_path = VIDEO_MODEL_PATH.replace('.keras', '.weights.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print(f"Файл ваг не знайдено: {weights_path}")

    eval_x_list = test_x_list + anom_x_list
    eval_y_list = test_y_list + anom_y_list
    eval_labels = [0] * len(test_x_list) + [1] * len(anom_x_list)
    
    eval_paths = test_paths + anom_paths 

    if len(eval_x_list) > 0:
        print(f"\nГотово до тестування! Всього сцен: {len(eval_x_list)}")
        
        print("Розраховуємо новий динамічний поріг...")
        x_norm_calib = np.concatenate(test_x_list)
        y_norm_calib = np.concatenate(test_y_list)
        
        threshold = calculate_threshold(model, x_norm_calib, y_norm_calib, percentile=95)
        print(f"Новий відкалібрований поріг: {threshold:.2f}")
        
        results = evaluate_model(model, threshold, eval_x_list, eval_y_list, eval_labels, consecutive=CONSECUTIVE)
        
        print("\n" + "="*60)
        print("ДЕТАЛЬНИЙ АНАЛІЗ ПО КЛАСАХ ДІЙ (EVENT-LEVEL)")
        print("="*60)
        print(f"{'Action':<8} | {'Сцен':<6} | {'Recall/TN Rate':<15} | {'Статус'}")
        
        from collections import defaultdict
        stats = defaultdict(lambda: {'tp': 0, 'fn': 0, 'tn': 0, 'fp': 0, 'type': ''})
        
        for i in range(len(eval_x_list)):
            try:
                code_str = eval_paths[i].split("A")[1].split(".")[0][:3]
                action_code = int(code_str)
            except:
                action_code = 999 
                
            pred = model.predict(eval_x_list[i], batch_size=64, verbose=0)
            errors = get_loss_top3_joints(eval_y_list[i], pred).numpy()[:, 0]
            
            trigger = False
            count = 0
            for val in errors:
                if val > threshold:
                    count += 1
                    if count >= 2: trigger = True; break
                else: count = 0
            
            is_anom = eval_labels[i] == 1
            if is_anom:
                stats[action_code]['type'] = 'anom'
                if trigger: stats[action_code]['tp'] += 1
                else: stats[action_code]['fn'] += 1
            else:
                stats[action_code]['type'] = 'norm'
                if trigger: stats[action_code]['fp'] += 1
                else: stats[action_code]['tn'] += 1

        for code, s in sorted(stats.items()):
            if s['type'] == 'anom':
                total = s['tp'] + s['fn']
                pred = (s['tp'] / total) * 100 if total > 0 else 0
                status = "OK" if pred > 70 else "Пропуски!"
                print(f"A{code:03d}    | {total:<6} | {pred:>6.1f}% (Recall) | {status}")
            else:
                total = s['tn'] + s['fp']
                pred = (s['tn'] / total) * 100 if total > 0 else 0
                status = "OK" if pred > 80 else "Хибні тривоги!"
                print(f"A{code:03d}    | {total:<6} | {pred:>6.1f}% (TN Rate)| {status}")